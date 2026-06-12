#pragma once
// ---------------------------------------------------------------------------
// nk::Model — the immutable cook product (plan §3.3).
//
// Model is the cook-constant half of an nk::World: joint/link topology, link
// inertias + local poses, shape tables, XPBD constraint templates, PBF params,
// the physics-material bucket table initial values, the contact filter policy,
// the max_* capacities, and the field schema. It holds host-side std::vectors
// and a single UploadTo() that PACKS the model-owned tables into ONE device
// phi::Buffer (256B-aligned sections, deterministic layout) and fills a
// phi::ModelView of typed device pointers into that buffer.
//
// PURE C++ — zero CUDA tokens. All device memory via phi::BufferType/BufferI,
// no execution here (Model only allocates + uploads its constant tables).
// ---------------------------------------------------------------------------

#include <cstdint>
#include <vector>

#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "phi/backend.hpp"   // phi::ModelView, BufferType, Buffer (forward + wrappers)
#include "nk/model/generated/field_ids.hpp"
#include "nk/model/generated/arena_layout.hpp"
#include "nk/model/generated/views.hpp"

namespace nuka::nk {

// Fixed-capacity policy (plan §3.3): every per-env array is sized to its max
// slot count; runtime `*_count` watermarks mask the unused tail. Capacities are
// a Model property (deterministic for a given cook product + env_count).
struct ModelCapacities {
    uint32_t env_count       = 1;   // number of replicated envs (env-major).
    uint32_t dofs_per_env    = 0;   // articulation generalized DOF count / env.
    uint32_t links_per_env   = 0;   // articulation link count / env.
    uint32_t bodies_per_env  = 0;   // movable rigid body count / env.
    uint32_t max_contacts_per_env = 0;  // contact-slot capacity / env.
    uint32_t max_rows_per_env     = 0;  // row-slot capacity / env.
    uint32_t particles_per_env    = 0;  // XPBD/PBF particle count / env.
    uint32_t dist_cons_per_env    = 0;  // XPBD distance-constraint count / env.
    uint32_t bend_cons_per_env    = 0;  // XPBD bend-constraint count / env.
    uint32_t vol_cons_per_env     = 0;  // XPBD volume-constraint count / env.
    uint32_t num_material_buckets = 0;  // physics-material bucket table rows.
    uint32_t obs_width            = 64; // per-env observation export width.

    // Resolve a FieldPer count-unit to a concrete per-env element count using
    // these capacities (env-major; the env multiplier is applied by the Arena /
    // UploadTo packer, NOT folded in here -- this returns the PER-ENV count).
    uint32_t PerEnvCount(FieldPer per) const;
    // Total element count across all envs for a field (per-env x env_count, with
    // scalar fields counted once per env unless they are global). mat_buckets is
    // the one symbolic per:scalar field; its count = num_material_buckets*8.
    uint64_t ElementCount(FieldId id) const;
};

// A cook-time articulation template (one per env, replicated). The arrays are
// the SINGLE-ENV BuildArticulationHostState product transcribed 1:1 (M3b), so
// the staged Model bytes match the legacy UploadArticulationState bytes exactly
// (the byte-exact kernel-port contract).
struct ModelArticulation {
    // per-link topology (length == links_per_env)
    std::vector<uint32_t>         parent_link;       // articulation-LOCAL; ~0u at the root
    std::vector<uint8_t>          joint_type;        // ArticulationJointType (u8-backed)
    std::vector<math::Vec3>       joint_axis;
    std::vector<math::Vec3>       parent_offset;     // parent_frame.position (host-state build)
    std::vector<uint32_t>         link_body;         // owning rigid-body row
    std::vector<math::Transform>  link_local_pose;
    std::vector<math::Transform>  link_inertial_frame;
    std::vector<float>            link_inertia_spatial;  // 36 floats / link (flat)
    std::vector<float>            joint_damping;
    std::vector<float>            joint_armature;
    std::vector<float>            initial_q;             // per LINK (scalar slot per link)
    std::vector<math::Transform>  initial_link_pose;     // cook rest pose per link
    math::Transform               base_pose = math::Transform::Identity();  // root world pose
    uint32_t                      dof_count  = 0;        // generalized DOFs (floating root = 6)
    uint32_t                      link_count = 0;
    uint32_t                      root_link  = ~uint32_t(0);
};

// One cooked foot-sphere row (legacy articulation::FootShape 1:1: base-relative
// calf link + sphere center offset in the calf frame + radius). Staged into the
// foot_shape Model field as 5 packed scalars {link bits, off.xyz, radius}.
struct ModelFootShape {
    uint32_t   calf_local_link = 0;
    math::Vec3 local_offset{};
    float      radius = 0.0f;
};

// Cook-seeded PD hold-drive template (per template link; legacy
// BuildHoldDriveTargets / CookGo2 HoldDrives 1:1). Seeded into the Data
// drive_* persistent fields at World construction (replicated env-major).
struct ModelHoldDrives {
    std::vector<float> targets;       // = initial_q (hold the cooked stance)
    std::vector<float> stiffness;     // actuator gain (Position actuators)
    std::vector<float> damping;       // 2*sqrt(gain) (DefaultDriveDamping)
    std::vector<float> force_limits;
};

// A cooked shape row (primitive params + optional hull/SDF asset reference).
struct ModelShape {
    uint8_t          kind = 0;          // scene::ShapeType
    uint32_t         body_row = ~uint32_t(0);
    uint32_t         material_bucket = 0;
    math::Transform  local_transform = math::Transform::Identity();
    math::Vec3       half_extents{0.5f, 0.5f, 0.5f};
    float            radius = 0.5f;
    float            half_height = 0.5f;
    uint32_t         convex_geometry_index = ~uint32_t(0);
    uint32_t         sdf_index = ~uint32_t(0);
};

// A physics-material bucket row (8 floats: μs, μd, restitution, compliant_k,
// compliant_d, density, sdf_cell_size, reserved). The mat_buckets device field
// is num_buckets x 8 floats (plan §3.3 example).
struct ModelMaterialBucket {
    float values[8] = {0.5f, 0.5f, 0.0f, 0.0f, 0.0f, 1000.0f, 0.005f, 0.0f};
};

class Model {
public:
    Model() = default;

    // -- cook-product host tables (filled by CookToModel) -------------------
    ModelCapacities                 capacities;
    ModelArticulation               articulation;       // the per-env template.
    std::vector<ModelShape>         shapes;             // cooked shape rows (per env).
    std::vector<ModelMaterialBucket> material_buckets;  // (num_buckets) bucket table.
    std::vector<uint32_t>           body_material_bucket;  // per body-row bucket index.

    // -- articulation contact pipeline config (M3b foot sphere x ground) -----
    // feet: derived by CookToModel (every Sphere shape owned by an articulation
    // link, the T2/T6 derivation). ground_height/friction/baumgarte are scene/
    // run config a caller may override BEFORE World construction (the Model is
    // mutable until then). foot count of 0 disables contact generation (the
    // detection kernel then zero-fills every slot — the single-env oracle path).
    std::vector<ModelFootShape>     feet;
    ModelHoldDrives                 hold_drives;
    float ground_height          = 0.0f;
    float friction_coefficient   = 0.8f;   // legacy kContactFriction
    float baumgarte_max_velocity = 3.0e38f; // ~+inf (legacy default non-binding)
    // Convex hull geometry + SDF grids are referenced by ModelShape indices and
    // packed into the .nka by the cooker; the device upload of those large
    // assets is the collision milestone's (M5) business, not M3a's.

    // Filter policy (plan §3.3): cross-env collision flag + (future) excluded
    // pairs. M3a carries the flag; the baked pair lists ride the CookedBlob.
    bool filter_cross_env = false;

    // The field schema is the generated FieldId enum + arena_layout table; Model
    // exposes the capacities the layout multiplies by. (No per-field storage here
    // beyond the constant tables above.)

    // -- device upload ------------------------------------------------------
    // Pack ALL model-owned (owner==Model) fields into ONE device buffer via the
    // BufferType allocator, upload them, and FILL `out_view` with typed device
    // pointers into that buffer. 256B-aligned sections, deterministic layout
    // (iteration in FieldId order). The returned Buffer is owned by the Model and
    // freed in the dtor. Returns Status::Ok / OutOfMemory.
    phi::Status UploadTo(phi::BufferType* bt, phi::ModelView* out_view);

    // The packed device buffer (null until UploadTo). Exposed for World teardown.
    phi::Buffer* DeviceBuffer() const { return device_buffer_; }

    ~Model();
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    Model(Model&&) noexcept;
    Model& operator=(Model&&) noexcept;

    // -- deterministic segment layout (shared with the test's determinism gate)
    // One packed section per model-owned field, in FieldId order, each 256B
    // aligned. Computed purely from `capacities`; identical inputs => identical
    // table. Public so the acceptance test can assert two builds match.
    struct Segment {
        FieldId  field;
        uint64_t offset;   // byte offset into the device buffer
        uint64_t bytes;    // section size (element_count x elem_size)
    };
    std::vector<Segment> ComputeModelSegments(uint64_t* total_bytes) const;

private:
    // Fill the per-field host staging bytes for one model-owned field into `dst`
    // at the segment offset (transcribes the host tables above into the packed
    // layout). Unimplemented (zero-filled) sections are left as the buffer's
    // memset(0) default -- the kernel-port milestone (M3b) wires the real packing
    // as each op lands; M3a guarantees the LAYOUT + pointers + determinism.
    void StageModelField(FieldId id, const Segment& seg, std::vector<uint8_t>& host) const;

    phi::Buffer* device_buffer_ = nullptr;
};

} // namespace nuka::nk
