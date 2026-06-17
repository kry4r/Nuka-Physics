// GENERATED — do not edit; regen via tools/codegen/regen.py
// Source of truth: src/nk/model/fields.yaml

#pragma once

// The REAL definitions of nuka::phi::ModelView / nuka::phi::DataView that
// phi/backend.hpp only FORWARD-DECLARES. Ops receive these by const-ref and
// read typed device pointers, one per field. Clean owner split: ModelView
// carries the cook-constant model tables; DataView carries the mutable per-
// World state. Both are plain aggregates of raw device pointers (filled by
// nk::Model::UploadTo and nk::Data respectively).

#include <cstdint>

#include "math/vec3.hpp"
#include "math/quat.hpp"
#include "math/transform.hpp"

namespace nuka::nk {
// Spatial / matrix element types for the articulation device state
// (float[6] spatial vectors, float[36] 6x6 spatial matrices). Trivial
// aggregates so a typed pointer indexes one element per link/row.
struct Spatial6 { float v[6]; };
struct Mat36    { float m[36]; };
} // namespace nuka::nk

namespace nuka::phi {

// Model-owned, cook-constant tables. Pointers index into the ONE Model
// device buffer (nk::Model::UploadTo packs + 256B-aligns the sections).
struct ModelView {
    ::nuka::nk::Mat36* link_inertia = nullptr;  // per:link arena:persistent owner:model
    ::nuka::math::Transform* link_local_pose = nullptr;  // per:link arena:persistent owner:model
    ::nuka::math::Transform* link_inertial_frame = nullptr;  // per:link arena:persistent owner:model
    ::nuka::math::Vec3* joint_axis = nullptr;  // per:link arena:persistent owner:model
    ::nuka::math::Vec3* parent_offset = nullptr;  // per:link arena:persistent owner:model
    uint8_t* joint_type = nullptr;  // per:link arena:persistent owner:model
    uint32_t* parent_link = nullptr;  // per:link arena:persistent owner:model
    uint32_t* link_body = nullptr;  // per:link arena:persistent owner:model
    uint32_t* link_to_articulation = nullptr;  // per:link arena:persistent owner:model
    float* joint_damping = nullptr;  // per:link arena:persistent owner:model flags:[param]
    float* joint_armature = nullptr;  // per:link arena:persistent owner:model flags:[param]
    uint32_t* articulation_link_count = nullptr;  // per:articulation arena:persistent owner:model
    uint32_t* articulation_link_offset = nullptr;  // per:articulation arena:persistent owner:model
    float* foot_shape = nullptr;  // per:scalar arena:persistent owner:model count:max_contacts_per_env*5
    float* hull_verts = nullptr;  // per:scalar arena:persistent owner:model count:max_hull_verts*3
    uint32_t* dof_to_link = nullptr;  // per:dof arena:persistent owner:model
    uint32_t* dof_to_component = nullptr;  // per:dof arena:persistent owner:model
    float* shape_table = nullptr;  // per:scalar arena:persistent owner:model count:max_bodies_total*12
    uint64_t* excluded_pairs = nullptr;  // per:scalar arena:persistent owner:model count:max_excluded_pairs
    float* samp_points = nullptr;  // per:scalar arena:persistent owner:model count:max_samp_points*3
    uint32_t* samp_ranges = nullptr;  // per:scalar arena:persistent owner:model count:max_bodies_total*2
    float* sdf_headers = nullptr;  // per:scalar arena:persistent owner:model count:max_sdf_grids*8
    uint32_t* sdf_cell_count = nullptr;  // per:scalar arena:persistent owner:model count:max_sdf_grids
    uint64_t* sdf_cell_keys = nullptr;  // per:scalar arena:persistent owner:model count:max_sdf_cells
    float* sdf_cell_values = nullptr;  // per:scalar arena:persistent owner:model count:max_sdf_cells
    ::nuka::math::Vec3* sdf_cell_gradients = nullptr;  // per:scalar arena:persistent owner:model count:max_sdf_cells
    uint32_t* island_row_offsets = nullptr;  // per:row_slot arena:persistent owner:model elem:4
    uint32_t* island_color_segments = nullptr;  // per:row_slot arena:persistent owner:model elem:2
    uint32_t* row_order = nullptr;  // per:row_slot arena:persistent owner:model
    uint32_t* dist_particle_a = nullptr;  // per:dist_con arena:persistent owner:model
    uint32_t* dist_particle_b = nullptr;  // per:dist_con arena:persistent owner:model
    float* dist_rest_length = nullptr;  // per:dist_con arena:persistent owner:model
    float* dist_compliance = nullptr;  // per:dist_con arena:persistent owner:model flags:[param]
    uint32_t* bend_particles = nullptr;  // per:bend_con arena:persistent owner:model elem:4
    ::nuka::math::Vec3* bend_gradients = nullptr;  // per:bend_con arena:persistent owner:model elem:4
    float* bend_compliance = nullptr;  // per:bend_con arena:persistent owner:model flags:[param]
    uint32_t* vol_particles = nullptr;  // per:vol_con arena:persistent owner:model elem:4
    float* vol_rest_times6 = nullptr;  // per:vol_con arena:persistent owner:model
    float* vol_compliance = nullptr;  // per:vol_con arena:persistent owner:model flags:[param]
    uint32_t* sm_cluster_offset = nullptr;  // per:shape_match_slot arena:persistent owner:model
    uint32_t* sm_cluster_size = nullptr;  // per:shape_match_slot arena:persistent owner:model
    float* sm_stiffness = nullptr;  // per:shape_match_slot arena:persistent owner:model flags:[param]
    ::nuka::math::Vec3* sm_rest_centroid = nullptr;  // per:shape_match_slot arena:persistent owner:model
    uint32_t* sm_particles = nullptr;  // per:shape_match_member arena:persistent owner:model
    ::nuka::math::Vec3* sm_rest_q = nullptr;  // per:shape_match_member arena:persistent owner:model
    float* sm_mass = nullptr;  // per:shape_match_member arena:persistent owner:model
    uint32_t* link_geom_kind = nullptr;  // per:link arena:persistent owner:model
    float* link_geom_params = nullptr;  // per:link arena:persistent owner:model elem:4
    ::nuka::math::Transform* link_geom_local = nullptr;  // per:link arena:persistent owner:model
    uint32_t* body_to_link = nullptr;  // per:body arena:persistent owner:model
    uint32_t* body_to_articulation = nullptr;  // per:body arena:persistent owner:model
    float* heights = nullptr;  // per:scalar arena:persistent owner:model count:max_heightfield_cells
};

// Data-owned, mutable per-World state. Pointers index into the nk::Arena
// (Persistent / Scratch / Tape phi Buffers), env-major.
struct DataView {
    float* q = nullptr;  // per:link arena:persistent owner:data flags:[diff]
    float* qdot = nullptr;  // per:link arena:persistent owner:data flags:[diff]
    ::nuka::math::Transform* link_pose = nullptr;  // per:link arena:persistent owner:data
    ::nuka::math::Transform* base_pose = nullptr;  // per:articulation arena:persistent owner:data flags:[diff]
    ::nuka::math::Transform* body_pose = nullptr;  // per:body arena:persistent owner:data flags:[diff]
    float* body_inv_mass = nullptr;  // per:body arena:persistent owner:data flags:[param]
    uint32_t* contact_count = nullptr;  // per:env arena:scratch owner:data
    float* rows = nullptr;  // per:contact_slot arena:scratch owner:data elem:16
    float* lambda = nullptr;  // per:row_slot arena:persistent owner:data
    float* m_inv = nullptr;  // per:articulation_dof2 arena:scratch owner:data
    ::nuka::math::Vec3* particle_pos = nullptr;  // per:particle arena:persistent owner:data flags:[diff]
    float* mat_buckets = nullptr;  // per:scalar arena:persistent owner:data count:num_buckets*8 flags:[param]
    uint32_t* mat_index = nullptr;  // per:body arena:persistent owner:data flags:[param]
    ::nuka::nk::Spatial6* link_contact_wrench = nullptr;  // per:link arena:scratch owner:data flags:[readout]
    float* qddot = nullptr;  // per:link arena:persistent owner:data
    float* tau = nullptr;  // per:link arena:persistent owner:data
    float* joint_force = nullptr;  // per:link arena:scratch owner:data
    float* joint_diagonal = nullptr;  // per:link arena:scratch owner:data
    ::nuka::nk::Spatial6* link_velocity = nullptr;  // per:link arena:persistent owner:data flags:[diff]
    ::nuka::nk::Spatial6* link_acceleration = nullptr;  // per:link arena:scratch owner:data
    ::nuka::nk::Spatial6* link_velocity_bias = nullptr;  // per:link arena:scratch owner:data
    ::nuka::nk::Mat36* link_xup = nullptr;  // per:link arena:scratch owner:data
    ::nuka::nk::Mat36* link_articulated_I = nullptr;  // per:link arena:scratch owner:data
    ::nuka::nk::Spatial6* link_bias_force = nullptr;  // per:link arena:scratch owner:data
    ::nuka::nk::Spatial6* link_u_spatial = nullptr;  // per:link arena:scratch owner:data
    ::nuka::nk::Spatial6* joint_motion_subspace = nullptr;  // per:link arena:scratch owner:data
    float* drive_target = nullptr;  // per:link arena:persistent owner:data flags:[param]
    float* drive_stiffness = nullptr;  // per:link arena:persistent owner:data flags:[param]
    float* drive_damping = nullptr;  // per:link arena:persistent owner:data flags:[param]
    float* drive_force_limit = nullptr;  // per:link arena:persistent owner:data flags:[param]
    float* snapshot_q = nullptr;  // per:link arena:persistent owner:data
    float* snapshot_qdot = nullptr;  // per:link arena:persistent owner:data
    ::nuka::nk::Spatial6* snapshot_link_velocity = nullptr;  // per:link arena:persistent owner:data
    ::nuka::math::Transform* snapshot_base_pose = nullptr;  // per:articulation arena:persistent owner:data
    ::nuka::math::Transform* snapshot_body_pose = nullptr;  // per:body arena:persistent owner:data
    ::nuka::math::Vec3* snapshot_body_linear_velocity = nullptr;  // per:body arena:persistent owner:data
    ::nuka::math::Vec3* snapshot_body_angular_velocity = nullptr;  // per:body arena:persistent owner:data
    uint32_t* reset_env_ids = nullptr;  // per:env arena:scratch owner:data
    ::nuka::math::Vec3* body_linear_velocity = nullptr;  // per:body arena:persistent owner:data flags:[diff]
    ::nuka::math::Vec3* body_angular_velocity = nullptr;  // per:body arena:persistent owner:data flags:[diff]
    ::nuka::math::Vec3* body_force = nullptr;  // per:body arena:scratch owner:data
    ::nuka::math::Vec3* body_torque = nullptr;  // per:body arena:scratch owner:data
    ::nuka::math::Vec3* body_inv_inertia = nullptr;  // per:body arena:persistent owner:data flags:[param]
    ::nuka::math::Vec3* body_aabb_lo = nullptr;  // per:body arena:scratch owner:data
    ::nuka::math::Vec3* body_aabb_hi = nullptr;  // per:body arena:scratch owner:data
    uint32_t* pair_count = nullptr;  // per:env arena:scratch owner:data
    uint32_t* candidate_pairs = nullptr;  // per:contact_slot arena:scratch owner:data elem:2
    float* lbvh_nodes = nullptr;  // per:scalar arena:scratch owner:data count:lbvh_node_count
    uint32_t* lbvh_morton = nullptr;  // per:body arena:scratch owner:data
    uint32_t* lbvh_index = nullptr;  // per:body arena:scratch owner:data
    uint32_t* lbvh_visit = nullptr;  // per:body arena:scratch owner:data
    uint32_t* contact_link = nullptr;  // per:contact_slot arena:scratch owner:data
    ::nuka::math::Vec3* contact_point = nullptr;  // per:contact_slot arena:scratch owner:data
    ::nuka::math::Vec3* contact_normal = nullptr;  // per:contact_slot arena:scratch owner:data
    float* contact_depth = nullptr;  // per:contact_slot arena:scratch owner:data
    ::nuka::math::Vec3* contact_tangent1 = nullptr;  // per:contact_slot arena:scratch owner:data
    ::nuka::math::Vec3* contact_tangent2 = nullptr;  // per:contact_slot arena:scratch owner:data
    uint32_t* contact_material = nullptr;  // per:contact_slot arena:scratch owner:data
    float* jac_normal = nullptr;  // per:slot_dof arena:scratch owner:data
    float* jac_tangent1 = nullptr;  // per:slot_dof arena:scratch owner:data
    float* jac_tangent2 = nullptr;  // per:slot_dof arena:scratch owner:data
    float* contact_meff_normal = nullptr;  // per:contact_slot arena:scratch owner:data
    float* contact_meff_tangent1 = nullptr;  // per:contact_slot arena:scratch owner:data
    float* contact_meff_tangent2 = nullptr;  // per:contact_slot arena:scratch owner:data
    float* contact_force = nullptr;  // per:contact_slot arena:scratch owner:data elem:3 flags:[readout]
    uint32_t* ucontact_count = nullptr;  // per:contact_slot arena:scratch owner:data
    ::nuka::math::Vec3* ucontact_point = nullptr;  // per:contact_slot arena:scratch owner:data elem:4
    ::nuka::math::Vec3* ucontact_normal = nullptr;  // per:contact_slot arena:scratch owner:data elem:4
    float* ucontact_depth = nullptr;  // per:contact_slot arena:scratch owner:data elem:4
    uint32_t* row_count = nullptr;  // per:env arena:scratch owner:data
    float* urows = nullptr;  // per:row_slot arena:scratch owner:data elem:32
    float* chain_jacobian = nullptr;  // per:row_dof arena:scratch owner:data
    float* row_minv_jt = nullptr;  // per:row_dof arena:scratch owner:data
    float* row_meff = nullptr;  // per:row_slot arena:scratch owner:data
    uint32_t* row_cj_link = nullptr;  // per:row_slot arena:scratch owner:data
    ::nuka::math::Vec3* row_cj_point = nullptr;  // per:row_slot arena:scratch owner:data
    ::nuka::math::Vec3* row_cj_dir = nullptr;  // per:row_slot arena:scratch owner:data
    float* qdot_flat = nullptr;  // per:articulation_dof arena:scratch owner:data
    float* m = nullptr;  // per:articulation_dof2 arena:scratch owner:data
    ::nuka::nk::Mat36* link_composite_inertia = nullptr;  // per:link arena:scratch owner:data
    ::nuka::math::Vec3* particle_prev_pos = nullptr;  // per:particle arena:persistent owner:data
    ::nuka::math::Vec3* particle_vel = nullptr;  // per:particle arena:persistent owner:data flags:[diff]
    float* particle_inv_mass = nullptr;  // per:particle arena:persistent owner:data flags:[param]
    ::nuka::math::Vec3* particle_v_pre = nullptr;  // per:particle arena:scratch owner:data
    float* dist_lambda = nullptr;  // per:dist_con arena:persistent owner:data
    float* bend_lambda = nullptr;  // per:bend_con arena:persistent owner:data
    float* vol_lambda = nullptr;  // per:vol_con arena:persistent owner:data
    ::nuka::math::Vec3* pbf_predicted_pos = nullptr;  // per:particle arena:scratch owner:data
    ::nuka::math::Vec3* pbf_position_delta = nullptr;  // per:particle arena:scratch owner:data
    float* pbf_density = nullptr;  // per:particle arena:scratch owner:data
    float* pbf_lambda = nullptr;  // per:particle arena:scratch owner:data
    uint32_t* grid_cell_key = nullptr;  // per:particle arena:scratch owner:data
    uint32_t* grid_particle_idx = nullptr;  // per:particle arena:scratch owner:data
    uint32_t* grid_cell_start = nullptr;  // per:scalar arena:scratch owner:data count:max_grid_cells*env_count
    uint32_t* grid_cell_end = nullptr;  // per:scalar arena:scratch owner:data count:max_grid_cells*env_count
    uint32_t* grid_neighbor_offset = nullptr;  // per:particle arena:scratch owner:data
    uint32_t* grid_neighbor_count = nullptr;  // per:particle arena:scratch owner:data
    uint32_t* grid_neighbor_idx = nullptr;  // per:particle arena:scratch owner:data elem:32
    uint64_t* rng_state = nullptr;  // per:env arena:persistent owner:data flags:[param]
    uint32_t* env_status = nullptr;  // per:env arena:scratch owner:data flags:[readout]
    float* obs_buffer = nullptr;  // per:env arena:scratch owner:data elem:64 flags:[readout]
    uint32_t* env_terrain_type = nullptr;  // per:env arena:persistent owner:data
    float* env_terrain_difficulty = nullptr;  // per:env arena:persistent owner:data
    float* joint_f = nullptr;  // per:link arena:persistent owner:data flags:[param]
    uint32_t* ucontact_a = nullptr;  // per:contact_slot arena:scratch owner:data elem:4
    uint32_t* ucontact_b = nullptr;  // per:contact_slot arena:scratch owner:data elem:4
    uint32_t* ucontact_gen = nullptr;  // per:contact_slot arena:scratch owner:data elem:4
    ::nuka::math::Vec3* ucontact_tangent1 = nullptr;  // per:contact_slot arena:scratch owner:data elem:4
    ::nuka::math::Vec3* ucontact_tangent2 = nullptr;  // per:contact_slot arena:scratch owner:data elem:4
    float* chain_jacobian_b = nullptr;  // per:row_dof arena:scratch owner:data
    float* row_minv_jt_b = nullptr;  // per:row_dof arena:scratch owner:data
    uint32_t* row_cj_link_b = nullptr;  // per:row_slot arena:scratch owner:data
    ::nuka::math::Vec3* row_cj_point_b = nullptr;  // per:row_slot arena:scratch owner:data
    ::nuka::math::Vec3* row_cj_dir_b = nullptr;  // per:row_slot arena:scratch owner:data
};

} // namespace nuka::phi
