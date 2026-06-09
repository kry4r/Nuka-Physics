// ---------------------------------------------------------------------------
// nuka::collision::gpu -- BATCHED GRASP NARROWPHASE kernel + host launcher (P2.4c)
// ---------------------------------------------------------------------------
// See narrowphase_grasp.cuh for the design. This TU carries:
//   1. grasp_sphere_hull_kernel: one thread per (env x fingertip), wrapping the
//      HD-clean cvx::SphereHull (sphere=fingertip x ConvexHull=cup). The find_sdf_
//      contact_kernel template (sdf_contact.cu) applied to SphereHull -- slot-
//      indexed, fixed iteration caps -> D1.
//   2. LaunchGraspSphereHullNarrowphase: the HOST drop-in for the grasp narrowphase.
//      Uploads the per-slot sphere inputs + per-env cup frames + the SHARED cup
//      verts, launches ONE kernel over all (env x fingertip) slots, downloads the
//      ContactManifold buffer, and applies StampSides + the param merge EXACTLY as
//      contact_stream_driver.cpp::BuildContactManifolds does, so each non-empty slot
//      is byte-equivalent to the host BuildContactManifolds output (modulo the
//      ULP-scale SphereHull host-vs-device float delta the A2/diagnostic absorb).
// ---------------------------------------------------------------------------

#include "collision/gpu/narrowphase_grasp.cuh"

#include "collision/candidate_pair.hpp"        // CandidatePair / CollidableRef / MakeStableKey
#include "phi/buffer.hpp"                       // phi::Buffer
#include "phi/device_context.hpp"               // phi::DeviceContext
#include "scene/contact_filter.hpp"             // scene::MergeContactParams / ContactParamsIn

#include "collision/gpu/narrowphase_grasp.hpp"  // the host launcher declaration

#include <cuda_runtime.h>

#include <vector>

namespace nuka::collision::gpu {

using constraint::ContactManifold;
using math::Vec3;

// One thread per (env x fingertip) slot. Writes results[slot]. Slot-indexed: no
// atomics, no reduction, fixed order + fixed SphereHull iteration caps -> D1.
__global__ void grasp_sphere_hull_kernel(
    const GraspSphereInput* __restrict__ inputs,
    const amf::PrimFrame*   __restrict__ cup_frames,
    const float*            __restrict__ cup_verts,
    uint32_t                             cup_vcount,
    ContactManifold*        __restrict__ results,
    uint32_t                             slot_count) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= slot_count) {
        return;
    }
    const GraspSphereInput in = inputs[i];

    // The fingertip sphere prim (identity rotation -- a sphere is rotation-invariant;
    // mirrors FootSpherePrim: only frame.t (the world center) + radius matter).
    amf::PrimParams sphere;
    sphere.radius = in.radius;
    sphere.frame.t = in.center;

    // The cup hull view: SHARED mesh-local verts + this env's host-baked world frame.
    cvx::ConvexHullView hull;
    hull.verts = cup_verts;
    hull.vcount = cup_vcount;
    hull.frame = cup_frames[in.env_index];

    // sphere_is_a == true: the grasp pair is (fingertip=A=sphere, cup=B=hull), the
    // SAME ordering the host dispatch (narrowphase_dispatch.hpp) routes -> the emitted
    // normal is the separation dir for side A (the fingertip). Flipping it would push
    // the cup the wrong way (HOLD fails). SphereHull does Clear()+AddPoint, so a
    // separated fingertip writes point_count=0 (the no-contact early-return).
    ContactManifold m;
    cvx::SphereHull(sphere, hull, /*sphere_is_a=*/true, &m);
    results[i] = m;
}

// ---------------------------------------------------------------------------
// HOST LAUNCHER -- the grasp narrowphase drop-in.
// ---------------------------------------------------------------------------
void LaunchGraspSphereHullNarrowphase(
    const phi::DeviceContext& context,
    const std::vector<GraspSphereInput>& inputs,
    const std::vector<amf::PrimFrame>& cup_frames,
    const float* cup_verts,
    uint32_t cup_vcount,
    const std::vector<constraint::CollidableRef>& side_a,
    const std::vector<constraint::CollidableRef>& side_b,
    std::vector<ContactManifold>* out) {
    if (out == nullptr) {
        return;
    }
    out->clear();
    const uint32_t slot_count = static_cast<uint32_t>(inputs.size());
    if (slot_count == 0u || cup_vcount == 0u || cup_frames.empty()) {
        return;  // no fingertip pairs / no cup geometry -> no manifolds.
    }
    // SLOT-INDEXED output: one ContactManifold per INPUT slot, in slot order (env-major,
    // fingertip), so the caller recovers env/fingertip attribution from the slot index
    // (a no-contact slot keeps point_count==0; the caller skips it, mirroring the host
    // BuildContactManifolds non-empty append). This is what lets the batched world bucket
    // the flat output back into per-env manifold lists.
    out->assign(slot_count, ContactManifold{});

    // ----- ONE upload of the per-slot sphere inputs + per-env cup frames + the
    // SHARED cup verts. (The cup verts are env-invariant -- one upload across all N
    // envs. The frames are per-env -- one entry / env, host-baked from the live cup
    // pose because BuildPrimFrame/Quat::Rotate are host-only `inline`, NOT HD.) -----
    //
    // ★ NAMED P2.4-series DEBT (off the critical path -- deferred deliberately). These four
    // device buffers are allocated PER CALL, and the mesh-local cup verts (time- AND env-
    // invariant) are re-uploaded every step. The P2.4b articulation path pre-allocates its
    // scratch (world_pose_dev_/crba_*) at construction to avoid exactly this; the equivalent
    // here is for BatchedUnifiedWorld to OWN persistent narrowphase scratch (sizes are all
    // known at construction: env_count*nfinger slots, cup_vcount verts uploaded ONCE, env_count
    // frames) and pass it in -- a launcher SIGNATURE change (this free function is stateless
    // today). NOT done now because after P2.4c the step is ROW_SOLVER-bound (~79.7% at N=32;
    // this whole narrowphase tag is ~10.6%), so the per-call alloc + constant re-upload is not
    // measurable. Named downstream consumer = the P2.4d/P2.4e device-resident-rows increment
    // (which already restructures this region); fold the persistent-scratch ownership there.
    phi::Buffer in_buf(static_cast<size_t>(slot_count) * sizeof(GraspSphereInput),
                       phi::MemoryKind::Device);
    phi::Buffer frame_buf(cup_frames.size() * sizeof(amf::PrimFrame),
                          phi::MemoryKind::Device);
    phi::Buffer vert_buf(static_cast<size_t>(cup_vcount) * 3u * sizeof(float),
                         phi::MemoryKind::Device);
    phi::Buffer res_buf(static_cast<size_t>(slot_count) * sizeof(ContactManifold),
                        phi::MemoryKind::Device);
    in_buf.CopyFromHost(inputs.data(), inputs.size() * sizeof(GraspSphereInput));
    frame_buf.CopyFromHost(cup_frames.data(), cup_frames.size() * sizeof(amf::PrimFrame));
    vert_buf.CopyFromHost(cup_verts,
                          static_cast<size_t>(cup_vcount) * 3u * sizeof(float));

    // ----- ONE batched launch over all (env x fingertip) slots. -----
    constexpr uint32_t kBlock = 128u;
    const uint32_t grid = (slot_count + kBlock - 1u) / kBlock;
    grasp_sphere_hull_kernel<<<grid, kBlock, 0, context.stream.Native()>>>(
        static_cast<const GraspSphereInput*>(in_buf.Data()),
        static_cast<const amf::PrimFrame*>(frame_buf.Data()),
        static_cast<const float*>(vert_buf.Data()), cup_vcount,
        static_cast<ContactManifold*>(res_buf.Data()), slot_count);
    context.stream.Synchronize();

    // ----- ONE download of the per-slot manifold buffer. -----
    std::vector<ContactManifold> slots(slot_count);
    res_buf.CopyToHost(slots.data(), slots.size() * sizeof(ContactManifold));

    // ----- Apply StampSides + the param merge, EXACTLY mirroring BuildContactManifolds
    // (contact_stream_driver.cpp), per NON-EMPTY slot, IN PLACE at the slot index. The grasp
    // resolver leaves both sides' MuJoCo per-geom params at the ResolvedShape defaults, so the
    // merge is MergeContactParams(default, default) -- computed once and reused. An empty slot
    // (point_count==0, the SphereHull separated early-return) is LEFT default (the caller skips
    // it), mirroring the host BuildContactManifolds non-empty append. The non-empty fields
    // (a/b/manifold_key + merged friction/solimp/solref) are byte-equivalent to what host
    // BuildContactManifolds produced for the SAME pair -- the only delta is the ULP-scale
    // SphereHull host-vs-device point geometry (position/normal/penetration). -----
    const scene::ContactParamsIn def_in{};  // default ResolvedShape -> default merge input.
    const scene::MergedContactParams merged = scene::MergeContactParams(def_in, def_in);
    for (uint32_t i = 0u; i < slot_count; ++i) {
        ContactManifold m = slots[i];
        if (m.point_count == 0u) {
            continue;  // no contact -> left default-constructed (caller skips it).
        }
        // StampSides: a/b from the candidate pair + the D1 manifold_key.
        m.a = side_a[i];
        m.b = side_b[i];
        m.manifold_key = MakeStableKey(side_a[i], side_b[i]);
        // Param merge (driver step 6): friction / solimp / per-point solref.
        m.friction = merged.friction_mu;
        for (int k = 0; k < 5; ++k) m.solimp[k] = merged.solimp[k];
        for (uint32_t p = 0u; p < m.point_count; ++p) {
            m.points[p].solref_timeconst = merged.solref[0];
            m.points[p].solref_dampratio = merged.solref[1];
        }
        (*out)[i] = m;
    }
}

}  // namespace nuka::collision::gpu
