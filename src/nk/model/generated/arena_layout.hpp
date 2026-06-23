// GENERATED — do not edit; regen via tools/codegen/regen.py
// Source of truth: src/nk/model/fields.yaml

#pragma once

// Per-field layout descriptor table, indexed by FieldId. The Arena and
// Model use this to compute deterministic 256B-aligned segment offsets
// (per-unit multiplier x env_count x capacity x elem_size). owner picks
// the buffer (Model device buffer vs Arena Persistent/Scratch/Tape).

#include <cstdint>

#include "nk/model/generated/field_ids.hpp"

namespace nuka::nk {

enum class FieldArena : uint8_t { Persistent, Scratch, Tape };
enum class FieldOwner : uint8_t { Model, Data };
// The count-unit a field is sized by. ArenaLayout/Model resolve each to a
// concrete element count from the Model capacities x env_count.
enum class FieldPer : uint8_t {
    Env, Dof, Link, Body, ContactSlot, RowSlot, SlotDof, RowDof, Particle,
    DistCon, BendCon, VolCon, ShapeMatchSlot, ShapeMatchMember, EnvDof2,
    Scalar,
    // Multi-articulation co-residence: per-articulation (= articulations_per_env)
    // and per-articulation M-tile (= articulations_per_env * max_dof^2).
    // APPENDED so the existing enumerator values never shift.
    Articulation, ArticulationDof2,
    // Per-articulation flat-DOF tile (= articulations_per_env * max_dof);
    // the general-contact per-artic qdot_flat tile (S3). APPENDED LAST.
    ArticulationDof
};

struct FieldLayout {
    FieldArena arena;
    FieldOwner owner;
    FieldPer   per;
    uint16_t   elem;       // packed elements per unit (e.g. rows=16)
    uint16_t   lanes;      // scalar lanes per element (vec3=3, mat36=36)
    uint16_t   elem_size;  // bytes per element (elem x lanes x sizeof scalar)
    uint32_t   scalar_count;  // for FieldPer::Scalar (0 otherwise); see note
};

// NOTE: per:scalar fields with a symbolic count (e.g. mat_buckets =
// num_buckets*8) carry scalar_count = 0 here; the Model resolves the real
// count from its bucket table at layout time (the symbolic expression is
// documented in fields.yaml and field_ids.hpp).

inline constexpr FieldLayout kFieldLayout[kFieldCount] = {
    {FieldArena::Persistent, FieldOwner::Data, FieldPer::Link, 1, 1, 4, 0},  // q
    {FieldArena::Persistent, FieldOwner::Data, FieldPer::Link, 1, 1, 4, 0},  // qdot
    {FieldArena::Persistent, FieldOwner::Data, FieldPer::Link, 1, 7, 28, 0},  // link_pose
    {FieldArena::Persistent, FieldOwner::Data, FieldPer::Articulation, 1, 7, 28, 0},  // base_pose
    {FieldArena::Persistent, FieldOwner::Data, FieldPer::Body, 1, 7, 28, 0},  // body_pose
    {FieldArena::Persistent, FieldOwner::Data, FieldPer::Body, 1, 1, 4, 0},  // body_inv_mass
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::Env, 1, 1, 4, 0},  // contact_count
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::ContactSlot, 16, 1, 64, 0},  // rows
    {FieldArena::Persistent, FieldOwner::Data, FieldPer::RowSlot, 1, 1, 4, 0},  // lambda
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::ArticulationDof2, 1, 1, 4, 0},  // m_inv
    {FieldArena::Persistent, FieldOwner::Data, FieldPer::Particle, 1, 3, 12, 0},  // particle_pos
    {FieldArena::Persistent, FieldOwner::Data, FieldPer::Scalar, 1, 1, 4, 0},  // mat_buckets
    {FieldArena::Persistent, FieldOwner::Data, FieldPer::Body, 1, 1, 4, 0},  // mat_index
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::Link, 1, 6, 24, 0},  // link_contact_wrench
    {FieldArena::Persistent, FieldOwner::Data, FieldPer::Link, 1, 1, 4, 0},  // qddot
    {FieldArena::Persistent, FieldOwner::Data, FieldPer::Link, 1, 1, 4, 0},  // tau
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::Link, 1, 1, 4, 0},  // joint_force
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::Link, 1, 1, 4, 0},  // joint_diagonal
    {FieldArena::Persistent, FieldOwner::Data, FieldPer::Link, 1, 6, 24, 0},  // link_velocity
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::Link, 1, 6, 24, 0},  // link_acceleration
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::Link, 1, 6, 24, 0},  // link_velocity_bias
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::Link, 1, 36, 144, 0},  // link_xup
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::Link, 1, 36, 144, 0},  // link_articulated_I
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::Link, 1, 6, 24, 0},  // link_bias_force
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::Link, 1, 6, 24, 0},  // link_u_spatial
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::Link, 1, 6, 24, 0},  // joint_motion_subspace
    {FieldArena::Persistent, FieldOwner::Data, FieldPer::Link, 1, 1, 4, 0},  // drive_target
    {FieldArena::Persistent, FieldOwner::Data, FieldPer::Link, 1, 1, 4, 0},  // drive_stiffness
    {FieldArena::Persistent, FieldOwner::Data, FieldPer::Link, 1, 1, 4, 0},  // drive_damping
    {FieldArena::Persistent, FieldOwner::Data, FieldPer::Link, 1, 1, 4, 0},  // drive_force_limit
    {FieldArena::Persistent, FieldOwner::Data, FieldPer::Link, 1, 1, 4, 0},  // snapshot_q
    {FieldArena::Persistent, FieldOwner::Data, FieldPer::Link, 1, 1, 4, 0},  // snapshot_qdot
    {FieldArena::Persistent, FieldOwner::Data, FieldPer::Link, 1, 6, 24, 0},  // snapshot_link_velocity
    {FieldArena::Persistent, FieldOwner::Data, FieldPer::Articulation, 1, 7, 28, 0},  // snapshot_base_pose
    {FieldArena::Persistent, FieldOwner::Data, FieldPer::Body, 1, 7, 28, 0},  // snapshot_body_pose
    {FieldArena::Persistent, FieldOwner::Data, FieldPer::Body, 1, 3, 12, 0},  // snapshot_body_linear_velocity
    {FieldArena::Persistent, FieldOwner::Data, FieldPer::Body, 1, 3, 12, 0},  // snapshot_body_angular_velocity
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::Env, 1, 1, 4, 0},  // reset_env_ids
    {FieldArena::Persistent, FieldOwner::Model, FieldPer::Link, 1, 36, 144, 0},  // link_inertia
    {FieldArena::Persistent, FieldOwner::Model, FieldPer::Link, 1, 7, 28, 0},  // link_local_pose
    {FieldArena::Persistent, FieldOwner::Model, FieldPer::Link, 1, 7, 28, 0},  // link_inertial_frame
    {FieldArena::Persistent, FieldOwner::Model, FieldPer::Link, 1, 3, 12, 0},  // joint_axis
    {FieldArena::Persistent, FieldOwner::Model, FieldPer::Link, 1, 3, 12, 0},  // parent_offset
    {FieldArena::Persistent, FieldOwner::Model, FieldPer::Link, 1, 1, 1, 0},  // joint_type
    {FieldArena::Persistent, FieldOwner::Model, FieldPer::Link, 1, 1, 4, 0},  // parent_link
    {FieldArena::Persistent, FieldOwner::Model, FieldPer::Link, 1, 1, 4, 0},  // link_body
    {FieldArena::Persistent, FieldOwner::Model, FieldPer::Link, 1, 1, 4, 0},  // link_to_articulation
    {FieldArena::Persistent, FieldOwner::Model, FieldPer::Link, 1, 1, 4, 0},  // joint_damping
    {FieldArena::Persistent, FieldOwner::Model, FieldPer::Link, 1, 1, 4, 0},  // joint_armature
    {FieldArena::Persistent, FieldOwner::Model, FieldPer::Articulation, 1, 1, 4, 0},  // articulation_link_count
    {FieldArena::Persistent, FieldOwner::Model, FieldPer::Articulation, 1, 1, 4, 0},  // articulation_link_offset
    {FieldArena::Persistent, FieldOwner::Model, FieldPer::Scalar, 1, 1, 4, 0},  // foot_shape
    {FieldArena::Persistent, FieldOwner::Data, FieldPer::Body, 1, 3, 12, 0},  // body_linear_velocity
    {FieldArena::Persistent, FieldOwner::Data, FieldPer::Body, 1, 3, 12, 0},  // body_angular_velocity
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::Body, 1, 3, 12, 0},  // body_force
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::Body, 1, 3, 12, 0},  // body_torque
    {FieldArena::Persistent, FieldOwner::Data, FieldPer::Body, 1, 3, 12, 0},  // body_inv_inertia
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::Body, 1, 3, 12, 0},  // body_aabb_lo
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::Body, 1, 3, 12, 0},  // body_aabb_hi
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::Env, 1, 1, 4, 0},  // pair_count
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::ContactSlot, 2, 1, 8, 0},  // candidate_pairs
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::Scalar, 1, 1, 4, 0},  // lbvh_nodes
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::Body, 1, 1, 4, 0},  // lbvh_morton
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::Body, 1, 1, 4, 0},  // lbvh_index
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::Body, 1, 1, 4, 0},  // lbvh_visit
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::Body, 1, 1, 8, 0},  // lbvh_sortkey
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::ContactSlot, 1, 1, 4, 0},  // contact_link
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::ContactSlot, 1, 3, 12, 0},  // contact_point
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::ContactSlot, 1, 3, 12, 0},  // contact_normal
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::ContactSlot, 1, 1, 4, 0},  // contact_depth
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::ContactSlot, 1, 3, 12, 0},  // contact_tangent1
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::ContactSlot, 1, 3, 12, 0},  // contact_tangent2
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::ContactSlot, 1, 1, 4, 0},  // contact_material
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::SlotDof, 1, 1, 4, 0},  // jac_normal
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::SlotDof, 1, 1, 4, 0},  // jac_tangent1
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::SlotDof, 1, 1, 4, 0},  // jac_tangent2
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::ContactSlot, 1, 1, 4, 0},  // contact_meff_normal
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::ContactSlot, 1, 1, 4, 0},  // contact_meff_tangent1
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::ContactSlot, 1, 1, 4, 0},  // contact_meff_tangent2
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::ContactSlot, 3, 1, 12, 0},  // contact_force
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::ContactSlot, 1, 1, 4, 0},  // ucontact_count
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::ContactSlot, 4, 3, 48, 0},  // ucontact_point
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::ContactSlot, 4, 3, 48, 0},  // ucontact_normal
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::ContactSlot, 4, 1, 16, 0},  // ucontact_depth
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::Env, 1, 1, 4, 0},  // row_count
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::RowSlot, 32, 1, 128, 0},  // urows
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::RowDof, 1, 1, 4, 0},  // chain_jacobian
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::RowDof, 1, 1, 4, 0},  // row_minv_jt
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::RowSlot, 1, 1, 4, 0},  // row_meff
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::RowSlot, 1, 1, 4, 0},  // row_cj_link
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::RowSlot, 1, 3, 12, 0},  // row_cj_point
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::RowSlot, 1, 3, 12, 0},  // row_cj_dir
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::ArticulationDof, 1, 1, 4, 0},  // qdot_flat
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::ArticulationDof2, 1, 1, 4, 0},  // m
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::Link, 1, 36, 144, 0},  // link_composite_inertia
    {FieldArena::Persistent, FieldOwner::Model, FieldPer::Scalar, 1, 1, 4, 0},  // hull_verts
    {FieldArena::Persistent, FieldOwner::Model, FieldPer::Dof, 1, 1, 4, 0},  // dof_to_link
    {FieldArena::Persistent, FieldOwner::Model, FieldPer::Dof, 1, 1, 4, 0},  // dof_to_component
    {FieldArena::Persistent, FieldOwner::Model, FieldPer::Scalar, 1, 1, 4, 0},  // shape_table
    {FieldArena::Persistent, FieldOwner::Model, FieldPer::Scalar, 1, 1, 8, 0},  // excluded_pairs
    {FieldArena::Persistent, FieldOwner::Model, FieldPer::Scalar, 1, 1, 4, 0},  // samp_points
    {FieldArena::Persistent, FieldOwner::Model, FieldPer::Scalar, 1, 1, 4, 0},  // samp_ranges
    {FieldArena::Persistent, FieldOwner::Model, FieldPer::Scalar, 1, 1, 4, 0},  // sdf_headers
    {FieldArena::Persistent, FieldOwner::Model, FieldPer::Scalar, 1, 1, 4, 0},  // sdf_cell_count
    {FieldArena::Persistent, FieldOwner::Model, FieldPer::Scalar, 1, 1, 8, 0},  // sdf_cell_keys
    {FieldArena::Persistent, FieldOwner::Model, FieldPer::Scalar, 1, 1, 4, 0},  // sdf_cell_values
    {FieldArena::Persistent, FieldOwner::Model, FieldPer::Scalar, 1, 3, 12, 0},  // sdf_cell_gradients
    {FieldArena::Persistent, FieldOwner::Model, FieldPer::RowSlot, 4, 1, 16, 0},  // island_row_offsets
    {FieldArena::Persistent, FieldOwner::Model, FieldPer::RowSlot, 2, 1, 8, 0},  // island_color_segments
    {FieldArena::Persistent, FieldOwner::Model, FieldPer::RowSlot, 1, 1, 4, 0},  // row_order
    {FieldArena::Persistent, FieldOwner::Data, FieldPer::Particle, 1, 3, 12, 0},  // particle_prev_pos
    {FieldArena::Persistent, FieldOwner::Data, FieldPer::Particle, 1, 3, 12, 0},  // particle_vel
    {FieldArena::Persistent, FieldOwner::Data, FieldPer::Particle, 1, 1, 4, 0},  // particle_inv_mass
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::Particle, 1, 3, 12, 0},  // particle_v_pre
    {FieldArena::Persistent, FieldOwner::Model, FieldPer::DistCon, 1, 1, 4, 0},  // dist_particle_a
    {FieldArena::Persistent, FieldOwner::Model, FieldPer::DistCon, 1, 1, 4, 0},  // dist_particle_b
    {FieldArena::Persistent, FieldOwner::Model, FieldPer::DistCon, 1, 1, 4, 0},  // dist_rest_length
    {FieldArena::Persistent, FieldOwner::Model, FieldPer::DistCon, 1, 1, 4, 0},  // dist_compliance
    {FieldArena::Persistent, FieldOwner::Data, FieldPer::DistCon, 1, 1, 4, 0},  // dist_lambda
    {FieldArena::Persistent, FieldOwner::Model, FieldPer::BendCon, 4, 1, 16, 0},  // bend_particles
    {FieldArena::Persistent, FieldOwner::Model, FieldPer::BendCon, 4, 3, 48, 0},  // bend_gradients
    {FieldArena::Persistent, FieldOwner::Model, FieldPer::BendCon, 1, 1, 4, 0},  // bend_compliance
    {FieldArena::Persistent, FieldOwner::Data, FieldPer::BendCon, 1, 1, 4, 0},  // bend_lambda
    {FieldArena::Persistent, FieldOwner::Model, FieldPer::VolCon, 4, 1, 16, 0},  // vol_particles
    {FieldArena::Persistent, FieldOwner::Model, FieldPer::VolCon, 1, 1, 4, 0},  // vol_rest_times6
    {FieldArena::Persistent, FieldOwner::Model, FieldPer::VolCon, 1, 1, 4, 0},  // vol_compliance
    {FieldArena::Persistent, FieldOwner::Data, FieldPer::VolCon, 1, 1, 4, 0},  // vol_lambda
    {FieldArena::Persistent, FieldOwner::Model, FieldPer::ShapeMatchSlot, 1, 1, 4, 0},  // sm_cluster_offset
    {FieldArena::Persistent, FieldOwner::Model, FieldPer::ShapeMatchSlot, 1, 1, 4, 0},  // sm_cluster_size
    {FieldArena::Persistent, FieldOwner::Model, FieldPer::ShapeMatchSlot, 1, 1, 4, 0},  // sm_stiffness
    {FieldArena::Persistent, FieldOwner::Model, FieldPer::ShapeMatchSlot, 1, 3, 12, 0},  // sm_rest_centroid
    {FieldArena::Persistent, FieldOwner::Model, FieldPer::ShapeMatchMember, 1, 1, 4, 0},  // sm_particles
    {FieldArena::Persistent, FieldOwner::Model, FieldPer::ShapeMatchMember, 1, 3, 12, 0},  // sm_rest_q
    {FieldArena::Persistent, FieldOwner::Model, FieldPer::ShapeMatchMember, 1, 1, 4, 0},  // sm_mass
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::Particle, 1, 3, 12, 0},  // pbf_predicted_pos
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::Particle, 1, 3, 12, 0},  // pbf_position_delta
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::Particle, 1, 1, 4, 0},  // pbf_density
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::Particle, 1, 1, 4, 0},  // pbf_lambda
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::Particle, 1, 1, 4, 0},  // grid_cell_key
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::Particle, 1, 1, 4, 0},  // grid_particle_idx
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::Scalar, 1, 1, 4, 0},  // grid_cell_start
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::Scalar, 1, 1, 4, 0},  // grid_cell_end
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::Particle, 1, 1, 4, 0},  // grid_neighbor_offset
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::Particle, 1, 1, 4, 0},  // grid_neighbor_count
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::Particle, 32, 1, 128, 0},  // grid_neighbor_idx
    {FieldArena::Persistent, FieldOwner::Data, FieldPer::Env, 1, 1, 8, 0},  // rng_state
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::Env, 1, 1, 4, 0},  // env_status
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::Scalar, 1, 1, 4, 0},  // obs_buffer
    {FieldArena::Persistent, FieldOwner::Data, FieldPer::Env, 1, 1, 4, 0},  // env_terrain_type
    {FieldArena::Persistent, FieldOwner::Data, FieldPer::Env, 1, 1, 4, 0},  // env_terrain_difficulty
    {FieldArena::Persistent, FieldOwner::Data, FieldPer::Link, 1, 1, 4, 0},  // joint_f
    {FieldArena::Persistent, FieldOwner::Model, FieldPer::Link, 1, 1, 4, 0},  // link_geom_kind
    {FieldArena::Persistent, FieldOwner::Model, FieldPer::Link, 4, 1, 16, 0},  // link_geom_params
    {FieldArena::Persistent, FieldOwner::Model, FieldPer::Link, 1, 7, 28, 0},  // link_geom_local
    {FieldArena::Persistent, FieldOwner::Model, FieldPer::Body, 1, 1, 4, 0},  // body_to_link
    {FieldArena::Persistent, FieldOwner::Model, FieldPer::Body, 1, 1, 4, 0},  // body_to_articulation
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::ContactSlot, 4, 1, 16, 0},  // ucontact_a
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::ContactSlot, 4, 1, 16, 0},  // ucontact_b
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::ContactSlot, 4, 1, 16, 0},  // ucontact_gen
    {FieldArena::Persistent, FieldOwner::Model, FieldPer::Scalar, 1, 1, 4, 0},  // heights
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::ContactSlot, 4, 3, 48, 0},  // ucontact_tangent1
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::ContactSlot, 4, 3, 48, 0},  // ucontact_tangent2
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::RowDof, 1, 1, 4, 0},  // chain_jacobian_b
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::RowDof, 1, 1, 4, 0},  // row_minv_jt_b
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::RowSlot, 1, 1, 4, 0},  // row_cj_link_b
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::RowSlot, 1, 3, 12, 0},  // row_cj_point_b
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::RowSlot, 1, 3, 12, 0},  // row_cj_dir_b
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::RowSlot, 3, 1, 12, 0},  // pd_solve_scratch
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::RowSlot, 1, 1, 4, 0},  // row_penetration
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::Link, 1, 1, 4, 0},  // qdot_pseudo
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::Link, 1, 6, 24, 0},  // link_velocity_pseudo
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::ArticulationDof, 1, 1, 4, 0},  // qdot_pseudo_flat
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::Body, 1, 3, 12, 0},  // body_pseudo_linear_velocity
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::Body, 1, 3, 12, 0},  // body_pseudo_angular_velocity
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::Particle, 1, 3, 12, 0},  // particle_pseudo_vel
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::RowSlot, 1, 1, 4, 0},  // row_pseudo_lambda
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::ContactSlot, 4, 1, 16, 0},  // ucontact_a_kind
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::ContactSlot, 4, 1, 16, 0},  // ucontact_b_kind
    {FieldArena::Persistent, FieldOwner::Data, FieldPer::Particle, 1, 3, 12, 0},  // snapshot_particle_pos
    {FieldArena::Persistent, FieldOwner::Data, FieldPer::Particle, 1, 3, 12, 0},  // snapshot_particle_prev_pos
    {FieldArena::Persistent, FieldOwner::Data, FieldPer::Particle, 1, 3, 12, 0},  // snapshot_particle_vel
    {FieldArena::Persistent, FieldOwner::Model, FieldPer::Scalar, 1, 1, 4, 0},  // dist_color_segments
    {FieldArena::Persistent, FieldOwner::Model, FieldPer::Scalar, 1, 1, 4, 0},  // bend_color_segments
    {FieldArena::Persistent, FieldOwner::Model, FieldPer::Scalar, 1, 1, 4, 0},  // vol_color_segments
    {FieldArena::Persistent, FieldOwner::Model, FieldPer::Scalar, 1, 1, 4, 0},  // sm_color_segments
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::Scalar, 1, 1, 1, 0},  // grid_sort_scratch
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::Scalar, 1, 1, 4, 0},  // grid_mass
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::Scalar, 1, 3, 12, 0},  // grid_momentum
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::Scalar, 1, 3, 12, 0},  // grid_velocity
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::Scalar, 1, 3, 12, 0},  // grid_force
    {FieldArena::Persistent, FieldOwner::Data, FieldPer::Particle, 9, 1, 36, 0},  // particle_F
    {FieldArena::Persistent, FieldOwner::Data, FieldPer::Particle, 9, 1, 36, 0},  // particle_C
    {FieldArena::Persistent, FieldOwner::Data, FieldPer::Particle, 1, 1, 4, 0},  // particle_vol0
    {FieldArena::Persistent, FieldOwner::Data, FieldPer::Particle, 1, 1, 4, 0},  // particle_plastic
    {FieldArena::Persistent, FieldOwner::Data, FieldPer::Particle, 1, 1, 4, 0},  // particle_material_id
    {FieldArena::Persistent, FieldOwner::Data, FieldPer::Particle, 9, 1, 36, 0},  // snapshot_particle_F
    {FieldArena::Persistent, FieldOwner::Data, FieldPer::Particle, 9, 1, 36, 0},  // snapshot_particle_C
    {FieldArena::Persistent, FieldOwner::Data, FieldPer::Particle, 1, 1, 4, 0},  // snapshot_particle_plastic
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::Particle, 1, 1, 4, 0},  // mpm_grid_cell_key
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::Particle, 1, 1, 4, 0},  // mpm_grid_part_idx
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::Scalar, 1, 1, 1, 0},  // mpm_sort_scratch
    {FieldArena::Persistent, FieldOwner::Data, FieldPer::Scalar, 1, 1, 4, 0},  // mpm_material_table
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::Scalar, 1, 3, 12, 0},  // grid_body_dp
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::Scalar, 1, 1, 4, 0},  // grid_body_owner
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::Body, 1, 3, 12, 0},  // mpm_body_reaction
    {FieldArena::Scratch, FieldOwner::Data, FieldPer::Body, 1, 3, 12, 0},  // mpm_body_ang_reaction
};

inline constexpr const FieldLayout& LayoutOf(FieldId id) {
    return kFieldLayout[static_cast<int>(id)];
}

} // namespace nuka::nk
