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
#include "math/symmetric_mat3.hpp"
#include "collision/mesh_surface_types.hpp"
#include "nk/solve/point_endpoint.hpp"

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
    float* shape_table = nullptr;  // per:scalar arena:persistent owner:model count:max_bodies_total*13
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
    float* bend_rest_angle = nullptr;  // per:bend_con arena:persistent owner:model
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
    uint32_t* aero_tri_verts = nullptr;  // per:aero_tri arena:persistent owner:model elem:3
    float* aero_tri_area = nullptr;  // per:aero_tri arena:persistent owner:model
    uint32_t* link_geom_kind = nullptr;  // per:link arena:persistent owner:model
    float* link_geom_params = nullptr;  // per:link arena:persistent owner:model elem:4
    ::nuka::math::Transform* link_geom_local = nullptr;  // per:link arena:persistent owner:model
    uint32_t* body_to_link = nullptr;  // per:body arena:persistent owner:model
    uint32_t* body_to_articulation = nullptr;  // per:body arena:persistent owner:model
    float* heights = nullptr;  // per:scalar arena:persistent owner:model count:max_heightfield_cells
    uint32_t* dist_color_segments = nullptr;  // per:scalar arena:persistent owner:model count:xpbd_dist_colors*2
    uint32_t* bend_color_segments = nullptr;  // per:scalar arena:persistent owner:model count:xpbd_bend_colors*2
    uint32_t* vol_color_segments = nullptr;  // per:scalar arena:persistent owner:model count:xpbd_vol_colors*2
    uint32_t* sm_color_segments = nullptr;  // per:scalar arena:persistent owner:model count:xpbd_sm_colors*2
    float* joint_frictionloss = nullptr;  // per:link arena:persistent owner:model flags:[param]
    uint32_t* body_collidable_link = nullptr;  // per:body arena:persistent owner:model
    ::nuka::math::Transform* body_collidable_local = nullptr;  // per:body arena:persistent owner:model
    uint32_t* body_collidable_body = nullptr;  // per:body arena:persistent owner:model
    float* joint_limit_lower = nullptr;  // per:link arena:persistent owner:model flags:[param]
    float* joint_limit_upper = nullptr;  // per:link arena:persistent owner:model flags:[param]
    uint8_t* joint_limit_flags = nullptr;  // per:link arena:persistent owner:model flags:[param]
    uint32_t* aero_particle_offset = nullptr;  // per:particle arena:persistent owner:model
    uint32_t* aero_particle_count = nullptr;  // per:particle arena:persistent owner:model
    uint32_t* aero_incident_tri = nullptr;  // per:aero_tri arena:persistent owner:model elem:3
    uint32_t* particle_topology_offsets = nullptr;  // per:scalar arena:persistent owner:model count:particle_topology_offsets
    uint32_t* particle_topology_elements = nullptr;  // per:scalar arena:persistent owner:model count:particle_topology_incidence_count
    ::nuka::math::Vec3* particle_contact_rest_pos = nullptr;  // per:scalar arena:persistent owner:model count:particle_contact_rest_positions
    ::nuka::collision::MeshSurfaceInfo* mesh_surface_info = nullptr;  // per:scalar arena:persistent owner:model count:mesh_surface_info_count
    uint32_t* mesh_triangles = nullptr;  // per:scalar arena:persistent owner:model count:max_mesh_triangles*3
    ::nuka::collision::MeshBvhNode* mesh_bvh_nodes = nullptr;  // per:scalar arena:persistent owner:model count:max_mesh_bvh_nodes
    ::nuka::collision::MeshSurfaceInfo* particle_surface_info = nullptr;  // per:scalar arena:persistent owner:model count:particle_surfaces_per_env
    uint32_t* particle_surface_triangles = nullptr;  // per:scalar arena:persistent owner:model count:particle_surface_triangles*3
    ::nuka::collision::MeshBvhNode* particle_surface_tree = nullptr;  // per:scalar arena:persistent owner:model count:particle_surface_nodes_per_env
    float* particle_surface_thickness = nullptr;  // per:scalar arena:persistent owner:model count:particle_surfaces_per_env
    float* particle_surface_friction = nullptr;  // per:scalar arena:persistent owner:model count:particle_surfaces_per_env
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
    float* mat_buckets = nullptr;  // per:scalar arena:persistent owner:data count:num_buckets*16 flags:[param]
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
    uint64_t* lbvh_sortkey = nullptr;  // per:body arena:scratch owner:data
    uint32_t* contact_link = nullptr;  // per:contact_slot arena:scratch owner:data
    ::nuka::math::Vec3* contact_point = nullptr;  // per:contact_slot arena:scratch owner:data
    ::nuka::math::Vec3* contact_normal = nullptr;  // per:contact_slot arena:scratch owner:data
    float* contact_depth = nullptr;  // per:contact_slot arena:scratch owner:data
    ::nuka::math::Vec3* contact_tangent1 = nullptr;  // per:contact_slot arena:scratch owner:data
    ::nuka::math::Vec3* contact_tangent2 = nullptr;  // per:contact_slot arena:scratch owner:data
    uint64_t* contact_material = nullptr;  // per:contact_slot arena:scratch owner:data
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
    float* row_damping = nullptr;  // per:row_slot arena:scratch owner:data
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
    uint32_t* grid_neighbor_count = nullptr;  // per:particle arena:scratch owner:data flags:[readout]
    uint32_t* grid_neighbor_idx = nullptr;  // per:scalar arena:scratch owner:data count:neighbor_pool_capacity*env_count
    uint64_t* rng_state = nullptr;  // per:env arena:persistent owner:data flags:[param]
    uint32_t* env_status = nullptr;  // per:env arena:scratch owner:data flags:[readout]
    float* obs_buffer = nullptr;  // per:scalar arena:scratch owner:data count:obs_width*env_count flags:[readout]
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
    uint32_t* pd_solve_scratch = nullptr;  // per:row_slot arena:scratch owner:data elem:3
    float* row_penetration = nullptr;  // per:row_slot arena:scratch owner:data
    float* qdot_pseudo = nullptr;  // per:link arena:scratch owner:data
    ::nuka::nk::Spatial6* link_velocity_pseudo = nullptr;  // per:link arena:scratch owner:data
    float* qdot_pseudo_flat = nullptr;  // per:articulation_dof arena:scratch owner:data
    ::nuka::math::Vec3* body_pseudo_linear_velocity = nullptr;  // per:body arena:scratch owner:data
    ::nuka::math::Vec3* body_pseudo_angular_velocity = nullptr;  // per:body arena:scratch owner:data
    ::nuka::math::Vec3* particle_pseudo_vel = nullptr;  // per:particle arena:scratch owner:data
    float* row_pseudo_lambda = nullptr;  // per:row_slot arena:scratch owner:data
    uint32_t* ucontact_a_kind = nullptr;  // per:contact_slot arena:scratch owner:data elem:4
    uint32_t* ucontact_b_kind = nullptr;  // per:contact_slot arena:scratch owner:data elem:4
    ::nuka::math::Vec3* snapshot_particle_pos = nullptr;  // per:particle arena:persistent owner:data
    ::nuka::math::Vec3* snapshot_particle_prev_pos = nullptr;  // per:particle arena:persistent owner:data
    ::nuka::math::Vec3* snapshot_particle_vel = nullptr;  // per:particle arena:persistent owner:data
    uint8_t* grid_sort_scratch = nullptr;  // per:scalar arena:scratch owner:data count:grid_sort_scratch_bytes
    float* grid_mass = nullptr;  // per:scalar arena:scratch owner:data count:mpm_grid_nodes_per_env*env_count
    ::nuka::math::Vec3* grid_momentum = nullptr;  // per:scalar arena:scratch owner:data count:mpm_grid_nodes_per_env*env_count
    ::nuka::math::Vec3* grid_velocity = nullptr;  // per:scalar arena:scratch owner:data count:mpm_grid_nodes_per_env*env_count
    ::nuka::math::Vec3* grid_force = nullptr;  // per:scalar arena:scratch owner:data count:mpm_grid_nodes_per_env*env_count
    float* particle_F = nullptr;  // per:particle arena:persistent owner:data elem:9 flags:[diff]
    float* particle_C = nullptr;  // per:particle arena:persistent owner:data elem:9
    float* particle_vol0 = nullptr;  // per:particle arena:persistent owner:data
    float* particle_plastic = nullptr;  // per:particle arena:persistent owner:data
    uint32_t* particle_material_id = nullptr;  // per:particle arena:persistent owner:data
    float* snapshot_particle_F = nullptr;  // per:particle arena:persistent owner:data elem:9
    float* snapshot_particle_C = nullptr;  // per:particle arena:persistent owner:data elem:9
    float* snapshot_particle_plastic = nullptr;  // per:particle arena:persistent owner:data
    uint32_t* mpm_grid_cell_key = nullptr;  // per:particle arena:scratch owner:data
    uint32_t* mpm_grid_part_idx = nullptr;  // per:particle arena:scratch owner:data
    uint8_t* mpm_sort_scratch = nullptr;  // per:scalar arena:scratch owner:data count:mpm_grid_sort_scratch_bytes
    float* mpm_particle_stress = nullptr;  // per:particle arena:scratch owner:data elem:9
    float* mpm_material_table = nullptr;  // per:scalar arena:persistent owner:data count:mpm_material_count*11
    ::nuka::math::Vec3* grid_body_dp = nullptr;  // per:scalar arena:scratch owner:data count:mpm_grid_nodes_per_env*env_count
    uint32_t* grid_body_owner = nullptr;  // per:scalar arena:scratch owner:data count:mpm_grid_nodes_per_env*env_count
    ::nuka::math::Vec3* mpm_body_reaction = nullptr;  // per:body arena:scratch owner:data flags:[readout]
    ::nuka::math::Vec3* mpm_body_ang_reaction = nullptr;  // per:body arena:scratch owner:data flags:[readout]
    uint32_t* cc_parent = nullptr;  // per:row_slot arena:scratch owner:data
    uint32_t* cc_root = nullptr;  // per:row_slot arena:scratch owner:data
    uint32_t* cc_artic_first = nullptr;  // per:articulation arena:scratch owner:data
    uint32_t* cc_body_first = nullptr;  // per:body arena:scratch owner:data
    uint32_t* cc_particle_first = nullptr;  // per:particle arena:scratch owner:data
    uint32_t* island_root_sorted = nullptr;  // per:row_slot arena:scratch owner:data
    uint32_t* island_rows = nullptr;  // per:row_slot arena:scratch owner:data
    uint32_t* island_quads = nullptr;  // per:row_slot arena:scratch owner:data elem:4
    uint32_t* island_count = nullptr;  // per:scalar arena:scratch owner:data count:1
    uint8_t* island_cub_temp = nullptr;  // per:scalar arena:scratch owner:data count:island_cub_temp_bytes
    uint64_t* ucontact_id_pair = nullptr;  // per:contact_slot arena:scratch owner:data elem:4
    uint64_t* ucontact_id_feature = nullptr;  // per:contact_slot arena:scratch owner:data elem:4
    uint64_t* contact_cache_pair = nullptr;  // per:contact_slot arena:persistent owner:data elem:4
    uint64_t* contact_cache_feature = nullptr;  // per:contact_slot arena:persistent owner:data elem:4
    float* contact_cache_lambda = nullptr;  // per:contact_slot arena:persistent owner:data elem:12
    ::nuka::math::Vec3* contact_cache_normal = nullptr;  // per:contact_slot arena:persistent owner:data elem:4
    ::nuka::math::Vec3* contact_cache_tangent1 = nullptr;  // per:contact_slot arena:persistent owner:data elem:4
    ::nuka::math::Vec3* contact_cache_tangent2 = nullptr;  // per:contact_slot arena:persistent owner:data elem:4
    uint64_t* contact_cache_material = nullptr;  // per:contact_slot arena:persistent owner:data elem:4
    uint32_t* contact_cache_age = nullptr;  // per:contact_slot arena:persistent owner:data elem:4
    uint64_t* contact_cache_snapshot_pair = nullptr;  // per:contact_slot arena:scratch owner:data elem:4
    uint64_t* contact_cache_snapshot_feature = nullptr;  // per:contact_slot arena:scratch owner:data elem:4
    float* contact_cache_snapshot_lambda = nullptr;  // per:contact_slot arena:scratch owner:data elem:12
    ::nuka::math::Vec3* contact_cache_snapshot_normal = nullptr;  // per:contact_slot arena:scratch owner:data elem:4
    ::nuka::math::Vec3* contact_cache_snapshot_tangent1 = nullptr;  // per:contact_slot arena:scratch owner:data elem:4
    ::nuka::math::Vec3* contact_cache_snapshot_tangent2 = nullptr;  // per:contact_slot arena:scratch owner:data elem:4
    uint64_t* contact_cache_snapshot_material = nullptr;  // per:contact_slot arena:scratch owner:data elem:4
    uint32_t* contact_cache_snapshot_age = nullptr;  // per:contact_slot arena:scratch owner:data elem:4
    uint32_t* contact_cache_current_owner = nullptr;  // per:contact_slot arena:scratch owner:data elem:4
    uint32_t* contact_cache_old_keep = nullptr;  // per:contact_slot arena:scratch owner:data elem:4
    float* joint_limit_impulse = nullptr;  // per:link arena:scratch owner:data elem:2 flags:[readout]
    float* actuator_effort_requested = nullptr;  // per:link arena:scratch owner:data flags:[readout]
    float* actuator_effort = nullptr;  // per:link arena:scratch owner:data flags:[readout]
    float* actuator_saturated = nullptr;  // per:link arena:scratch owner:data flags:[readout]
    uint8_t* pair_sort_scratch = nullptr;  // per:scalar arena:scratch owner:data count:pair_sort_scratch_bytes
    ::nuka::math::Vec3* task_target = nullptr;  // per:articulation arena:persistent owner:data flags:[param]
    ::nuka::math::Quat* task_rotation_target = nullptr;  // per:articulation arena:persistent owner:data flags:[param]
    ::nuka::math::Transform* task_local_pose = nullptr;  // per:articulation arena:persistent owner:data flags:[param]
    ::nuka::math::Transform* body_inertial_frame = nullptr;  // per:body arena:persistent owner:data flags:[param]
    ::nuka::math::SymmetricMat3* body_world_inv_inertia = nullptr;  // per:body arena:scratch owner:data
    float* step_qdot_flat = nullptr;  // per:articulation_dof arena:scratch owner:data
    ::nuka::math::Vec3* step_body_linear_velocity = nullptr;  // per:body arena:scratch owner:data
    ::nuka::math::Vec3* step_body_angular_velocity = nullptr;  // per:body arena:scratch owner:data
    ::nuka::math::Vec3* step_particle_velocity = nullptr;  // per:particle arena:scratch owner:data
    uint32_t* contact_side_a_kind = nullptr;  // per:contact_slot arena:scratch owner:data flags:[readout]
    uint32_t* contact_side_b_kind = nullptr;  // per:contact_slot arena:scratch owner:data flags:[readout]
    uint32_t* contact_side_a_index = nullptr;  // per:contact_slot arena:scratch owner:data flags:[readout]
    uint32_t* contact_side_b_index = nullptr;  // per:contact_slot arena:scratch owner:data flags:[readout]
    ::nuka::math::Vec3* aero_tri_impulse = nullptr;  // per:aero_tri arena:scratch owner:data
    float* body_gyro_residual = nullptr;  // per:body arena:scratch owner:data flags:[readout]
    uint32_t* body_gyro_iterations = nullptr;  // per:body arena:scratch owner:data flags:[readout]
    uint32_t* body_gyro_status = nullptr;  // per:body arena:scratch owner:data flags:[readout]
    uint32_t* grid_neighbor_attempted = nullptr;  // per:particle arena:scratch owner:data flags:[readout]
    uint64_t* grid_neighbor_scan_offset = nullptr;  // per:particle arena:scratch owner:data
    uint8_t* lbvh_sort_scratch = nullptr;  // per:scalar arena:scratch owner:data count:lbvh_sort_scratch_bytes
    uint8_t* contact_cache_scratch = nullptr;  // per:scalar arena:scratch owner:data count:contact_cache_scratch_bytes
    uint32_t* active_row_ids = nullptr;  // per:row_slot arena:scratch owner:data
    uint32_t* active_row_count = nullptr;  // per:env arena:scratch owner:data
    uint64_t* contact_endpoint_keys = nullptr;  // per:row_slot arena:scratch owner:data elem:2
    uint32_t* contact_endpoint_count = nullptr;  // per:env arena:scratch owner:data
    uint32_t* link_contact_begin = nullptr;  // per:link arena:scratch owner:data
    uint32_t* link_contact_end = nullptr;  // per:link arena:scratch owner:data
    uint8_t* contact_index_scratch = nullptr;  // per:scalar arena:scratch owner:data count:contact_index_scratch_bytes
    uint32_t* step_env_status = nullptr;  // per:env arena:scratch owner:data
    ::nuka::math::Vec3* step_mpm_body_impulse = nullptr;  // per:body arena:scratch owner:data
    ::nuka::math::Vec3* step_mpm_body_moment = nullptr;  // per:body arena:scratch owner:data
    ::nuka::math::Vec3* step_link_impulse = nullptr;  // per:link arena:scratch owner:data
    ::nuka::math::Vec3* step_link_moment = nullptr;  // per:link arena:scratch owner:data
    float* step_joint_limit_impulse = nullptr;  // per:link arena:scratch owner:data elem:2
    float* particle_plastic_F = nullptr;  // per:particle arena:persistent owner:data elem:9
    float* snapshot_particle_plastic_F = nullptr;  // per:particle arena:persistent owner:data elem:9
    float* grid_inv_mass = nullptr;  // per:scalar arena:scratch owner:data count:mpm_grid_nodes_per_env*env_count
    uint32_t* cc_grid_first = nullptr;  // per:scalar arena:scratch owner:data count:mpm_grid_nodes_per_env*env_count
    uint64_t* grid_contact_count = nullptr;  // per:scalar arena:scratch owner:data count:particles_per_env*env_count
    uint64_t* grid_contact_offset = nullptr;  // per:scalar arena:scratch owner:data count:particles_per_env*env_count
    uint32_t* ucontact_law = nullptr;  // per:contact_slot arena:scratch owner:data
    float* ucontact_friction = nullptr;  // per:contact_slot arena:scratch owner:data
    uint64_t* grid_contact_attempted = nullptr;  // per:env arena:scratch owner:data flags:[readout]
    uint32_t* grid_contact_retained = nullptr;  // per:env arena:scratch owner:data flags:[readout]
    uint64_t* grid_contact_peak = nullptr;  // per:env arena:persistent owner:data flags:[readout]
    uint64_t* grid_contact_overflow = nullptr;  // per:env arena:scratch owner:data flags:[readout]
    ::nuka::math::Vec3* mpm_boundary_impulse = nullptr;  // per:scalar arena:scratch owner:data count:mpm_boundary_count*env_count flags:[readout]
    ::nuka::math::Vec3* mpm_boundary_moment = nullptr;  // per:scalar arena:scratch owner:data count:mpm_boundary_count*env_count flags:[readout]
    ::nuka::math::Vec3* step_mpm_boundary_impulse = nullptr;  // per:scalar arena:scratch owner:data count:mpm_boundary_count*env_count
    ::nuka::math::Vec3* step_mpm_boundary_moment = nullptr;  // per:scalar arena:scratch owner:data count:mpm_boundary_count*env_count
    uint8_t* solver_velocity_scratch = nullptr;  // per:scalar arena:scratch owner:data count:solver_velocity_scratch_bytes
    float* velocity_target = nullptr;  // per:link arena:persistent owner:data flags:[param]
    float* acceleration_target = nullptr;  // per:link arena:persistent owner:data flags:[param]
    float* actuator_noload_speed = nullptr;  // per:link arena:persistent owner:data flags:[param]
    float* task_nullspace_stiffness = nullptr;  // per:articulation arena:persistent owner:data flags:[param]
    float* task_nullspace_damping = nullptr;  // per:articulation arena:persistent owner:data flags:[param]
    float* drive_command = nullptr;  // per:link arena:scratch owner:data
    float* drive_dissipation = nullptr;  // per:link arena:scratch owner:data
    float* drive_lower = nullptr;  // per:link arena:scratch owner:data
    float* drive_upper = nullptr;  // per:link arena:scratch owner:data
    float* control_mass = nullptr;  // per:articulation_dof2 arena:scratch owner:data
    float* control_factor = nullptr;  // per:articulation_dof2 arena:scratch owner:data
    float* control_jacobian = nullptr;  // per:articulation_dof arena:scratch owner:data elem:6
    float* control_response = nullptr;  // per:articulation_dof arena:scratch owner:data elem:6
    float* control_task_map = nullptr;  // per:articulation_dof arena:scratch owner:data elem:6
    ::nuka::collision::MeshBvhNode* particle_surface_nodes = nullptr;  // per:scalar arena:scratch owner:data count:particle_surface_nodes_per_env*env_count
    ::nuka::nk::PointEndpointRange* point_endpoint_ranges = nullptr;  // per:scalar arena:scratch owner:data count:point_endpoints_per_env*env_count flags:[readout]
    ::nuka::nk::PointEndpointTerm* point_endpoint_terms = nullptr;  // per:scalar arena:scratch owner:data count:point_endpoint_terms_per_env*env_count flags:[readout]
    ::nuka::math::Vec3* particle_projection_delta = nullptr;  // per:particle arena:scratch owner:data
    float* particle_surface_max_speed = nullptr;  // per:scalar arena:scratch owner:data count:particle_surfaces_per_env*env_count
    ::nuka::math::Vec3* grid_pseudo_vel = nullptr;  // per:scalar arena:scratch owner:data count:mpm_grid_nodes_per_env*env_count
    uint64_t* contact_solve_metrics = nullptr;  // per:env arena:scratch owner:data elem:8 flags:[readout]
    uint32_t* contact_solve_counts = nullptr;  // per:env arena:scratch owner:data elem:2 flags:[readout]
};

} // namespace nuka::phi
