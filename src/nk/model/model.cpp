// ---------------------------------------------------------------------------
// nk::Model implementation (plan §3.3).
// ---------------------------------------------------------------------------

#include "nk/model/model.hpp"

#include <cstring>
#include <utility>

namespace nuka::nk {

namespace {

constexpr uint64_t kAlign = 256;  // CUDA buffer-type alignment (BufferTypeAlignment).

uint64_t AlignUp(uint64_t v) { return (v + (kAlign - 1)) & ~(kAlign - 1); }

}  // namespace

uint32_t ModelCapacities::PerEnvCount(FieldPer per) const {
    switch (per) {
        case FieldPer::Env:            return 1u;
        case FieldPer::Dof:            return dofs_per_env;
        case FieldPer::Link:           return links_per_env;
        case FieldPer::Body:           return bodies_per_env;
        case FieldPer::ContactSlot:    return max_contacts_per_env;
        case FieldPer::RowSlot:        return max_rows_per_env;
        case FieldPer::SlotDof:        return max_contacts_per_env * dofs_per_env;
        case FieldPer::Particle:       return particles_per_env;
        case FieldPer::DistCon:        return dist_cons_per_env;
        case FieldPer::BendCon:        return bend_cons_per_env;
        case FieldPer::VolCon:         return vol_cons_per_env;
        case FieldPer::ShapeMatchSlot: return 0u;
        case FieldPer::EnvDof2:        return dofs_per_env * dofs_per_env;
        case FieldPer::Scalar:         return 0u;  // resolved by ElementCount
    }
    return 0u;
}

uint64_t ModelCapacities::ElementCount(FieldId id) const {
    const FieldLayout& lay = LayoutOf(id);
    if (lay.per == FieldPer::Scalar) {
        // The one symbolic per:scalar field in M3a is mat_buckets (num_buckets*8),
        // a GLOBAL table (not env-replicated). Any other scalar field defaults to
        // a single env-major row. Keyed by FieldId so the resolution is explicit.
        if (id == FieldId::MatBuckets) {
            return static_cast<uint64_t>(num_material_buckets) * 8ull;
        }
        if (id == FieldId::FootShape) {
            // GLOBAL foot table (shared by every env, base-relative indices):
            // 5 packed scalars per slot, max_contacts_per_env slots.
            return static_cast<uint64_t>(max_contacts_per_env) * 5ull;
        }
        return static_cast<uint64_t>(env_count);
    }
    const uint64_t per_env = PerEnvCount(lay.per);
    return per_env * static_cast<uint64_t>(env_count);
}

std::vector<Model::Segment> Model::ComputeModelSegments(uint64_t* total_bytes) const {
    std::vector<Segment> segs;
    uint64_t offset = 0;
    for (int i = 0; i < kFieldCount; ++i) {
        const FieldId id = static_cast<FieldId>(i);
        const FieldLayout& lay = LayoutOf(id);
        if (lay.owner != FieldOwner::Model) {
            continue;
        }
        const uint64_t count = capacities.ElementCount(id);
        const uint64_t bytes = count * static_cast<uint64_t>(lay.elem_size);
        const uint64_t aligned_off = AlignUp(offset);
        segs.push_back(Segment{id, aligned_off, bytes});
        offset = aligned_off + bytes;
    }
    if (total_bytes != nullptr) {
        *total_bytes = AlignUp(offset);
    }
    return segs;
}

namespace {

// memcpy one element repeatedly: dst[i] = src for i in [0, n). Used to stamp a
// per-link template value across replicas (env-major tiling, the EXACT layout
// ReplicateArticulationHostState's TileConcat produces).
template <typename T>
void StampPerLink(uint8_t* dst, const std::vector<T>& tpl, uint32_t links,
                  uint32_t envs, size_t elem_size) {
    for (uint32_t e = 0; e < envs; ++e) {
        for (uint32_t l = 0; l < links && l < tpl.size(); ++l) {
            std::memcpy(dst + (static_cast<size_t>(e) * links + l) * elem_size,
                        &tpl[l], elem_size);
        }
    }
}

}  // namespace

void Model::StageModelField(FieldId id, const Segment& seg,
                            std::vector<uint8_t>& host) const {
    // M3b: the REAL transcription of the cook-constant tables into the packed
    // section. Per-link tables tile env-major (replica e at [e*L, (e+1)*L)) —
    // value-identical to ReplicateArticulationHostState's TileConcat; the two
    // index tables (link_to_articulation, articulation_link_offset) get the
    // per-replica offset exactly as TileConcatOffset does. Sections this model
    // does not populate (solve-schedule M4, XPBD templates M6) stay zero.
    uint8_t* dst = host.data() + seg.offset;
    const ModelArticulation& a = articulation;
    const uint32_t L = capacities.links_per_env;
    const uint32_t E = capacities.env_count;
    if (seg.bytes == 0) {
        return;
    }
    switch (id) {
        case FieldId::LinkInertia: {
            for (uint32_t e = 0; e < E; ++e) {
                for (uint32_t l = 0; l < L; ++l) {
                    if ((static_cast<size_t>(l) + 1) * 36 <= a.link_inertia_spatial.size()) {
                        std::memcpy(dst + (static_cast<size_t>(e) * L + l) * 144,
                                    a.link_inertia_spatial.data() + static_cast<size_t>(l) * 36,
                                    144);
                    }
                }
            }
            break;
        }
        case FieldId::LinkLocalPose:
            StampPerLink(dst, a.link_local_pose, L, E, sizeof(math::Transform));
            break;
        case FieldId::LinkInertialFrame:
            StampPerLink(dst, a.link_inertial_frame, L, E, sizeof(math::Transform));
            break;
        case FieldId::JointAxis:
            StampPerLink(dst, a.joint_axis, L, E, sizeof(math::Vec3));
            break;
        case FieldId::ParentOffset:
            StampPerLink(dst, a.parent_offset, L, E, sizeof(math::Vec3));
            break;
        case FieldId::JointType:
            StampPerLink(dst, a.joint_type, L, E, sizeof(uint8_t));
            break;
        case FieldId::ParentLink:
            // articulation-LOCAL parent indices (the ~0u root sentinel included)
            // tile unchanged — the kernels compute global = offset + parent.
            StampPerLink(dst, a.parent_link, L, E, sizeof(uint32_t));
            break;
        case FieldId::LinkBody:
            StampPerLink(dst, a.link_body, L, E, sizeof(uint32_t));
            break;
        case FieldId::LinkToArticulation: {
            // replica e's links all belong to articulation e (1 artic / env).
            auto* p = reinterpret_cast<uint32_t*>(dst);
            for (uint32_t e = 0; e < E; ++e) {
                for (uint32_t l = 0; l < L; ++l) {
                    p[static_cast<size_t>(e) * L + l] = e;
                }
            }
            break;
        }
        case FieldId::JointDamping:
            StampPerLink(dst, a.joint_damping, L, E, sizeof(float));
            break;
        case FieldId::JointArmature:
            StampPerLink(dst, a.joint_armature, L, E, sizeof(float));
            break;
        case FieldId::ArticulationLinkCount: {
            auto* p = reinterpret_cast<uint32_t*>(dst);
            for (uint32_t e = 0; e < E; ++e) { p[e] = L; }
            break;
        }
        case FieldId::ArticulationLinkOffset: {
            auto* p = reinterpret_cast<uint32_t*>(dst);
            for (uint32_t e = 0; e < E; ++e) { p[e] = e * L; }
            break;
        }
        case FieldId::FootShape: {
            // GLOBAL table: 5 packed scalars per slot {link bits, off.xyz, radius}.
            auto* p = reinterpret_cast<float*>(dst);
            const uint32_t slots = capacities.max_contacts_per_env;
            for (uint32_t f = 0; f < feet.size() && f < slots; ++f) {
                uint32_t link_bits = feet[f].calf_local_link;
                std::memcpy(&p[static_cast<size_t>(f) * 5 + 0], &link_bits, 4);
                p[static_cast<size_t>(f) * 5 + 1] = feet[f].local_offset.x;
                p[static_cast<size_t>(f) * 5 + 2] = feet[f].local_offset.y;
                p[static_cast<size_t>(f) * 5 + 3] = feet[f].local_offset.z;
                p[static_cast<size_t>(f) * 5 + 4] = feet[f].radius;
            }
            break;
        }
        default:
            // Solve-schedule (M4) / XPBD-template (M6) sections: deterministic 0.
            break;
    }
}

namespace {

// Bind one segment's device pointer into the ModelView by FieldId. Centralized
// so the generated field list and the view members stay in lockstep (a missing
// case is a compile-time gap surfaced when a new model-owned field is added).
void BindModelPointer(phi::ModelView& v, FieldId id, void* p) {
    switch (id) {
        case FieldId::LinkInertia:           v.link_inertia = static_cast<Mat36*>(p); break;
        case FieldId::LinkLocalPose:         v.link_local_pose = static_cast<math::Transform*>(p); break;
        case FieldId::LinkInertialFrame:     v.link_inertial_frame = static_cast<math::Transform*>(p); break;
        case FieldId::JointAxis:             v.joint_axis = static_cast<math::Vec3*>(p); break;
        case FieldId::ParentOffset:          v.parent_offset = static_cast<math::Vec3*>(p); break;
        case FieldId::JointType:             v.joint_type = static_cast<uint8_t*>(p); break;
        case FieldId::ParentLink:            v.parent_link = static_cast<uint32_t*>(p); break;
        case FieldId::LinkBody:              v.link_body = static_cast<uint32_t*>(p); break;
        case FieldId::LinkToArticulation:    v.link_to_articulation = static_cast<uint32_t*>(p); break;
        case FieldId::JointDamping:          v.joint_damping = static_cast<float*>(p); break;
        case FieldId::JointArmature:         v.joint_armature = static_cast<float*>(p); break;
        case FieldId::ArticulationLinkCount: v.articulation_link_count = static_cast<uint32_t*>(p); break;
        case FieldId::ArticulationLinkOffset:v.articulation_link_offset = static_cast<uint32_t*>(p); break;
        case FieldId::FootShape:             v.foot_shape = static_cast<float*>(p); break;
        case FieldId::IslandRowOffsets:      v.island_row_offsets = static_cast<uint32_t*>(p); break;
        case FieldId::IslandColorSegments:   v.island_color_segments = static_cast<uint32_t*>(p); break;
        case FieldId::RowOrder:              v.row_order = static_cast<uint32_t*>(p); break;
        case FieldId::DistParticleA:         v.dist_particle_a = static_cast<uint32_t*>(p); break;
        case FieldId::DistParticleB:         v.dist_particle_b = static_cast<uint32_t*>(p); break;
        case FieldId::DistRestLength:        v.dist_rest_length = static_cast<float*>(p); break;
        case FieldId::DistCompliance:        v.dist_compliance = static_cast<float*>(p); break;
        case FieldId::BendParticles:         v.bend_particles = static_cast<uint32_t*>(p); break;
        case FieldId::BendGradients:         v.bend_gradients = static_cast<math::Vec3*>(p); break;
        case FieldId::BendCompliance:        v.bend_compliance = static_cast<float*>(p); break;
        case FieldId::VolParticles:          v.vol_particles = static_cast<uint32_t*>(p); break;
        case FieldId::VolRestTimes6:         v.vol_rest_times6 = static_cast<float*>(p); break;
        case FieldId::VolCompliance:         v.vol_compliance = static_cast<float*>(p); break;
        default: break;  // a data-owned field id: not a ModelView member.
    }
}

}  // namespace

phi::Status Model::UploadTo(phi::BufferType* bt, phi::ModelView* out_view) {
    if (bt == nullptr || out_view == nullptr) {
        return phi::Status::Failed;
    }
    uint64_t total = 0;
    const std::vector<Segment> segs = ComputeModelSegments(&total);

    phi::Buffer* buf = phi::BufferAlloc(bt, total == 0 ? kAlign : total);
    if (buf == nullptr) {
        return phi::Status::OutOfMemory;
    }
    // Zero the whole buffer first (deterministic; unimplemented sections stay 0).
    phi::BufferMemset(buf, 0, 0, total == 0 ? kAlign : total);

    // Stage the model-owned fields host-side, then ONE upload of the packed bytes.
    std::vector<uint8_t> host(total, 0u);
    for (const Segment& seg : segs) {
        StageModelField(seg.field, seg, host);
    }
    if (total > 0) {
        phi::BufferUpload(buf, host.data(), 0, total);
    }

    // Fill the view: each pointer = buffer base + section offset.
    auto* base = static_cast<uint8_t*>(phi::BufferBase(buf));
    *out_view = phi::ModelView{};
    for (const Segment& seg : segs) {
        BindModelPointer(*out_view, seg.field, base + seg.offset);
    }

    if (device_buffer_ != nullptr) {
        phi::BufferFree(device_buffer_);
    }
    device_buffer_ = buf;
    return phi::Status::Ok;
}

Model::~Model() {
    if (device_buffer_ != nullptr) {
        phi::BufferFree(device_buffer_);
        device_buffer_ = nullptr;
    }
}

Model::Model(Model&& other) noexcept
    : capacities(std::move(other.capacities)),
      articulation(std::move(other.articulation)),
      shapes(std::move(other.shapes)),
      material_buckets(std::move(other.material_buckets)),
      body_material_bucket(std::move(other.body_material_bucket)),
      feet(std::move(other.feet)),
      hold_drives(std::move(other.hold_drives)),
      ground_height(other.ground_height),
      friction_coefficient(other.friction_coefficient),
      baumgarte_max_velocity(other.baumgarte_max_velocity),
      filter_cross_env(other.filter_cross_env),
      device_buffer_(other.device_buffer_) {
    other.device_buffer_ = nullptr;
}

Model& Model::operator=(Model&& other) noexcept {
    if (this != &other) {
        if (device_buffer_ != nullptr) {
            phi::BufferFree(device_buffer_);
        }
        capacities = std::move(other.capacities);
        articulation = std::move(other.articulation);
        shapes = std::move(other.shapes);
        material_buckets = std::move(other.material_buckets);
        body_material_bucket = std::move(other.body_material_bucket);
        feet = std::move(other.feet);
        hold_drives = std::move(other.hold_drives);
        ground_height = other.ground_height;
        friction_coefficient = other.friction_coefficient;
        baumgarte_max_velocity = other.baumgarte_max_velocity;
        filter_cross_env = other.filter_cross_env;
        device_buffer_ = other.device_buffer_;
        other.device_buffer_ = nullptr;
    }
    return *this;
}

} // namespace nuka::nk
