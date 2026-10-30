#ifndef NUKA_NUKA_H
#define NUKA_NUKA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct nuka_device_t* nuka_device_handle;
typedef struct nuka_world_t* nuka_world_handle;
typedef struct nuka_buffer_t* nuka_buffer_handle;
typedef struct nuka_checkpoint_t* nuka_checkpoint_handle;

// Per-env contact-slot count for the per-contact enumeration fields
// (NUKA_FIELD_CONTACT_POINTS / _NORMAL / _FORCE / _LINK). A caller sizes those
// buffers as env_count * NUKA_MAX_CONTACT_SLOTS_PER_ENV. Mirrors the engine's
// internal kMaxFootContactsPerEnv; the two MUST stay equal.
#define NUKA_MAX_CONTACT_SLOTS_PER_ENV 4u

typedef enum nuka_result_t {
    NUKA_RESULT_OK = 0,
    NUKA_RESULT_INVALID_ARG = 1,
    NUKA_RESULT_NULL_HANDLE = 2,
    NUKA_RESULT_CUDA_ERROR = 3,
    NUKA_RESULT_FILE_NOT_FOUND = 4,
    NUKA_RESULT_PARSE_ERROR = 5,
    NUKA_RESULT_OUT_OF_MEMORY = 6,
    NUKA_RESULT_NOT_SUPPORTED = 7,
    NUKA_RESULT_INTERNAL = 100
} nuka_result_t;

typedef struct nuka_version_t {
    uint16_t major;
    uint16_t minor;
    uint16_t patch;
} nuka_version_t;

nuka_version_t nuka_get_version(void);
const char* nuka_result_message(nuka_result_t result);
// The text of the exception behind the last failed call on this thread, "" when there is none.
// Reading it clears it; the pointer stays valid until the next read on the same thread.
const char* nuka_last_error_message(void);

typedef struct nuka_device_desc_t {
    uint32_t gpu_index;
    // Non-null streams are borrowed until device destruction; null creates an owned stream.
    void* cuda_stream;
    uint8_t backend_selection_layer_enabled;
} nuka_device_desc_t;

nuka_result_t nuka_device_create(const nuka_device_desc_t* desc,
                                 nuka_device_handle* out);
void nuka_device_destroy(nuka_device_handle device);

typedef struct nuka_world_desc_t {
    const char* scene_path;
    uint32_t env_count;
    float fixed_dt;
    // Determinism level (p01-W4). 0 = D1 / Strong (the default when the desc is
    // zero-initialized): the current behavior, BIT-EXACT and reproducible (no
    // float atomics, fixed kernel order). 1 = D2 / Weak: the reserved escape
    // hatch for future atomic fast-paths. NOTE: today no hot kernel benefits
    // from atomics (every hotspot is <<<articulation_count, 32>>> with per-env
    // warp reductions -- there is no cross-env ordered reduction an atomic
    // variant would accelerate), so D2 currently selects the SAME kernels as D1
    // and is behaviorally identical; it is wired as the documented mechanism and
    // is NOT held to the D1 bit-exact bar. Any value > 1 is rejected with
    // NUKA_RESULT_INVALID_ARG.
    uint8_t determinism;
    // 0 PD position, 1 torque, 2 velocity, 3 computed torque, 4 OSC, 5 actuator.
    // Every creation entry supports all six modes at any positive environment count.
    uint8_t control_mode;
    // Articulation-local OSC task link; validated against every articulation at creation.
    // Each articulation has independent task inputs, including an optional orientation target.
    uint32_t osc_task_link;
    // Go2-on-stairs Phase 2a: procedural-terrain cook config (the shared
    // nuka::terrain::TerrainParams, minus ground_height which stays the model's).
    // These 5 floats are MODEL-LEVEL (one set for the whole world, NOT per env);
    // the per-env terrain TYPE + DIFFICULTY are written through
    // NUKA_FIELD_ENV_TERRAIN_TYPE / NUKA_FIELD_ENV_TERRAIN_DIFFICULTY. They are
    // applied AFTER cook + BEFORE the nk::World is constructed (the Pipeline reads
    // model.terrain at construct time). A zero-initialized desc leaves ALL FIVE at
    // 0.0 => the cook's flat terrain (SampleTerrainHeight returns ground_height for
    // every env regardless of type) => BYTE-IDENTICAL to a world created without
    // terrain (the D1 default). A caller that wants stairs sets terrain_step_height
    // / terrain_step_width / terrain_platform_width (PyramidStairs/InvertedPyramid)
    // or terrain_grid_width / terrain_grid_height_max (RandomBoxes), then writes a
    // non-zero terrain TYPE per env. See nuka::terrain::TerrainParams for the units.
    float terrain_step_height;      // vertical rise per step ring.
    float terrain_step_width;       // horizontal width of each step ring.
    float terrain_platform_width;   // full edge length of the central top platform.
    float terrain_grid_width;       // cell edge length (RandomBoxes).
    float terrain_grid_height_max;  // max per-cell random rise (RandomBoxes).
    // Multi-articulation CO-RESIDENCE (M10 dog-dog collision foundation, owner
    // bottom line): compose the authored scene with itself `instance_count` times
    // at distinct XY BEFORE cook, so ONE env holds K SEPARATE articulations that
    // co-step + physically collide (the WP1/WP5-8 multi-dog path validated by
    // tests/scenario/multi_dog_costep.cpp). Each replica i (i>=1) is placed at
    // x = i*instance_spacing (a line; the caller overrides BASE_POSE to the real
    // cluster after create) under a distinct name prefix so the cook keys K
    // disjoint kinematic trees -> articulation_count == instance_count. A value of
    // 0 OR 1 (the zero-initialized default) composes NOTHING -> the cook sees the
    // single authored scene -> BYTE-IDENTICAL to every legacy create. instance
    // co-residence is independent of env_count (use env_count == 1 for K dogs in
    // one world); the field arrays (Q / DRIVE_TARGET / ARTICULATION_LINK_POSE) are
    // then sized env_count * (K * per-dog), the K dogs concatenated per env.
    uint32_t instance_count;        // # co-resident articulation replicas (0/1 = single).
    float instance_spacing;         // X gap between replica spawns before cook (m).
    // ONE-GENERAL-SOLVER landing (L1-b): the FUSED runtime is DELETED, so every
    // world now cooks through the GENERAL contact path (LBVH broadphase + cvx GJK/EPA
    // narrowphase + mixed-island solve) unconditionally. This field NO LONGER selects
    // FUSED-vs-general (it is retained in the struct for ABI stability, removed in a
    // later milestone). It now only REQUESTS the baked-heightfield collidable the
    // general feet collide against: 1 == bake a STATIC heightfield collidable from
    // model.terrain (CookHeightfieldGrid samples SampleTerrainHeight over the grid,
    // pushes one static body_init row, and the general narrowphase routes any
    // overlapping foot to the per-cell TRIANGLE_PRISM contact); 0 == bake NOTHING
    // (the scene authors its own ground collidable, or its feet float, like the
    // go2_stand golden). The heightfield geometry is MODEL-LEVEL (one baked terrain
    // type for the whole batch -- general per-env terrain types are a later feature);
    // the type baked is heightfield_terrain_type
    // (0=Flat,1=PyramidStairs,2=InvertedPyramid,3=RandomBoxes). The grid is
    // (nrow x ncol) cells of edge heightfield_cell, centred at the world origin; a 0
    // in any of the three falls back to a default (41 x 41 @ 0.25 m == a ~10 m span)
    // sized to cover a single spawn footprint. Used by the L1 acceptance rollout
    // (trained go2 policy on the general path).
    uint32_t contact_family;          // 1 = bake a heightfield collidable, 0 = none.
    uint32_t heightfield_terrain_type;// baked heightfield type for the general path.
    uint32_t heightfield_nrow;        // general heightfield grid rows (0 => 41).
    uint32_t heightfield_ncol;        // general heightfield grid cols (0 => 41).
    float    heightfield_cell;        // general heightfield cell edge, m (0 => 0.25).
    // Terrain SOURCE selector (additive; a zero-init desc => parametric flat => the
    // byte-identical D1 default). When heightfield_image_path is non-NULL/non-empty
    // the grayscale height map at that path is loaded and placed by radius/elevation/
    // base (a malformed/unsupported image is a LOUD INVALID_ARG, never a silent flat
    // fallback); otherwise the parametric generator selected by heightfield_terrain_type
    // fills the grid from the terrain_* knobs above. The radius/elevation/base apply
    // to the IMAGE path (the parametric path derives its extent from nrow/ncol/cell).
    const char* heightfield_image_path;  // optional; NULL/"" => parametric.
    float    heightfield_radius_x;       // image half-extent X, m.
    float    heightfield_radius_y;       // image half-extent Y, m.
    float    heightfield_elevation_z;    // image value-1 world rise, m.
    float    heightfield_base_z;         // image value-0 world floor, m.
    // World gravity [m/s^2] for all forward dynamics, fixed at world creation.
    // All-zero selects {0,0,-9.81}; set_gravity_z can set zero gravity after creation.
    float gravity_x;
    float gravity_y;
    float gravity_z;
    // Per-env curriculum tile grid (heightfield_terrain_type == 5). The baked field
    // is laid out as curric_levels rows (+Y, difficulty) x curric_types cols (+X, a
    // fixed feature ladder stairs/pit/boxes/flat) of terrain_feature_cell tiles: the
    // col picks the feature, the row scales its amplitude by (row+1)/curric_levels.
    // The spawn/curriculum side maps an env's (type,level) to its tile centre, so obs
    // and physics (both sampled at the robot's world x,y) present that env its own
    // difficulty tile. 0 (the zero-init default) => off (other types unchanged).
    uint32_t curric_levels;          // # difficulty rows (0 => no curriculum grid).
    uint32_t curric_types;           // # feature columns.
    float    terrain_feature_cell;   // curriculum tile edge, m (0 => 6.0).
    // SolverConfig overrides (1:1 with nk::Pipeline::SolverConfig). Each 0 keeps the
    // engine default, so a zero-initialized desc takes today's code path unchanged.
    uint32_t solver_vel_iters;             // 0 => automatic block-descent budget.
    uint32_t solver_pos_iters;             // 0 => 4.
    float    solver_contact_margin;        // 0.0 => cooked default; speculative margin, m.
    uint32_t solver_max_pairs;             // 0 => bodies_per_env * 4 broadphase emit cap.
    float    solver_baumgarte_max_velocity;// 0.0 => cooked model default.
    uint32_t cloth_integrator;              // 0 = BDF2, 1 = backward Euler.
    uint32_t ogc_contact_capacity;          // Per-environment contact slots; 0 uses the cook estimate.
    float    solver_overlap_recovery_velocity; // Speculative overlap push-out bound, m/s; 0 => none.
} nuka_world_desc_t;

nuka_result_t nuka_world_create_from_scene(nuka_device_handle device,
                                           const nuka_world_desc_t* desc,
                                           nuka_world_handle* out);

// A COUPLED world: a rigid/articulated robot cooked from `desc->scene_path` PLUS
// VBD cloth particles, all stepping on the ONE general
// contact pipeline with two-way body<->particle coupling (a robot link is just a
// collidable body, a particle is a sphere; the medium difference is DATA -- per-
// system friction). The robot is cooked through the SAME scene->CookToModel path
// nuka_world_create_from_scene uses; the particles are then cooked onto that Model
// via the engine's particle cook (no scene-file particle schema yet, so the media
// are described programmatically here). The world steps/reads/resets through the
// EXISTING nuka_world_step / nuka_world_get_buffer_view / nuka_world_reset entries;
// read the live particle state via NUKA_FIELD_PARTICLE_POSITION/_VELOCITY.
//
// REACHABILITY SCOPE: this is the create+step+read FLOOR for a coupled world (the
// demo / sim substrate). Full RL on coupled worlds (obs/action/reward layouts over
// the particle state) is a separate RL-track follow-on; this entry does NOT add an
// RL stack -- it makes a coupled world creatable + steppable + readable.
//
// A cloth patch is a flat (cloth_nx x cloth_ny) lattice (>=2 each) at
// cloth_origin_{x,y,z} with cloth_spacing edge; its perimeter is pinned (a taut
// membrane). The cloth must be present (else INVALID_ARG -- use
// nuka_world_create_from_scene for a particle-free world).
typedef struct nuka_coupled_particles_desc_t {
    // --- cloth (vertex-block) lattice; cloth_nx == 0 OR cloth_ny == 0 => no cloth ---
    uint32_t cloth_nx;            // lattice columns (>= 2 to mesh; 0 => no cloth).
    uint32_t cloth_ny;            // lattice rows (>= 2 to mesh).
    float    cloth_spacing;       // lattice edge length, m (> 0 when present).
    float    cloth_origin_x;      // lattice centre X, m.
    float    cloth_origin_y;      // lattice centre Y, m.
    float    cloth_origin_z;      // lattice plane Z, m.
    float    cloth_particle_mass; // per-particle mass, kg (> 0; e.g. 0.01).
    float    cloth_friction;      // body<->cloth mu (finite => the foot grips/drags).
    float    cloth_stretch_stiffness;  // Young's modulus x thickness, N/m (> 0).
    float    cloth_bend_stiffness;     // hinge bending stiffness, N m (>= 0).
    float    cloth_poisson;            // membrane Poisson ratio in [0, 0.5).
    float    cloth_damping;            // Rayleigh damping on the elastic stiffness, s (>= 0).
    // Particle collision radius: a foot touches a particle when their
    // spheres overlap. The body<->particle contact d_min == 2*this (0 => engine
    // default sized from the smaller medium spacing).
    float    contact_radius;
    // Cloth solver selection (0 = VBD default; 1 = mlsmpm, rejected LOUDLY at cook).
    uint32_t soft_sim_method;
    // Cloth perimeter pin. 0 (default) pins the whole perimeter (a taut membrane);
    // 1 skips the pin so the sheet is a free drape (every particle dynamic).
    uint32_t cloth_free;
    // Cloth anisotropic aerodynamic drag, passed straight to VbdCookInput.aero_drag_*
    // (lumped 0.5*rho*Cn normal, 0.5*rho*Ct tangent, per-step velocity-delta clamp).
    // All 0 (default) => no drag op emitted => byte-identical cook; when any is > 0
    // the cloth surface triangles seed the drag op.
    float aero_normal;
    float aero_tangent;
    float aero_max_dv;
    // World SolverConfig overrides (1:1 with nk::Pipeline::SolverConfig), so a coupled
    // scene can author the contact solve. Each 0 keeps the engine default, so a
    // zero-initialized desc takes the SAME code path as today (byte-identical):
    // solver_vel_iters 0=>automatic budget, solver_pos_iters 0=>4, solver_contact_margin 0.0=>0.0,
    // solver_max_pairs 0=>0; baumgarte_max_velocity 0.0 keeps the cooked model default.
    uint32_t solver_vel_iters;
    uint32_t solver_pos_iters;
    float    solver_contact_margin;
    uint32_t solver_max_pairs;
    float    baumgarte_max_velocity;
    uint32_t cloth_integrator;       // 0 = BDF2, 1 = backward Euler.
    float    overlap_recovery_velocity; // Speculative overlap push-out bound, m/s; 0 => none.
} nuka_coupled_particles_desc_t;

// Create a coupled world. `desc` is the same robot/world descriptor
// nuka_world_create_from_scene takes (scene_path, env_count, dt, gravity, control
// mode, optional baked heightfield); `particles` describes the coupled media.
// Steps/reads through the existing world entries. Returns INVALID_ARG if neither
// medium is present, NOT_SUPPORTED if no CUDA backend, OUT_OF_MEMORY / INTERNAL on
// a build failure (e.g. the particle contact budget overflowing the slot block).
nuka_result_t nuka_world_create_coupled_from_scene(
    nuka_device_handle device, const nuka_world_desc_t* desc,
    const nuka_coupled_particles_desc_t* particles, nuka_world_handle* out);

void nuka_world_destroy(nuka_world_handle world);
nuka_result_t nuka_world_step(nuka_world_handle world);
nuka_result_t nuka_world_step_n(nuka_world_handle world, uint32_t step_count);

typedef enum nuka_execution_mode_t {
    NUKA_EXECUTION_EAGER = 0,
    NUKA_EXECUTION_GRAPH = 1
} nuka_execution_mode_t;

enum { NUKA_EXECUTION_INFO_VERSION = 1 };

typedef struct nuka_world_execution_info_t {
    uint32_t struct_size;
    uint32_t schema_version;
    nuka_execution_mode_t mode;
    uint32_t graph_ready;
    uint64_t capture_attempts;
    uint64_t graph_replays;
    nuka_result_t last_result;
    uint32_t failed_op;
    int32_t native_error;
    char message[256];
} nuka_world_execution_info_t;

// Graph selection captures one step without advancing state and fails explicitly if unavailable.
// Reset preserves the graph; changes to the operator sequence invalidate it.
nuka_result_t nuka_world_set_execution_mode(nuka_world_handle world, nuka_execution_mode_t mode);

// Set one template link's mass in all environments, preserving its COM and rotational inertia.
nuka_result_t nuka_world_set_link_mass(nuka_world_handle world, uint32_t link_index, float mass);

// Change uniform gravity Z for the next production step, preserving X/Y.
nuka_result_t nuka_world_set_gravity_z(nuka_world_handle world, float gravity_z);

// Block-descent iterations of the contact solve, 1..65535; the timestep stays fixed.
// A changed value invalidates the execution graph.
nuka_result_t nuka_world_set_velocity_iterations(nuka_world_handle world, uint32_t iterations);
nuka_result_t nuka_world_get_velocity_iterations(nuka_world_handle world, uint32_t* out_iterations);

typedef struct nuka_solver_options_t {
    uint32_t cone;
    uint32_t integrator;
    float impratio;
    uint32_t refsafe;
} nuka_solver_options_t;

nuka_result_t nuka_world_get_solver_options(nuka_world_handle world, nuka_solver_options_t* out);
nuka_result_t nuka_world_get_contact_profile_count(nuka_world_handle world, uint32_t* out);
// The shape name remains valid until the world is rebuilt or destroyed.
nuka_result_t nuka_world_get_contact_profile_name(nuka_world_handle world, uint32_t index, const char** out);

typedef enum nuka_param_type_t {
    NUKA_PARAM_F32 = 0,
    NUKA_PARAM_U32 = 1
} nuka_param_type_t;

// LIVE values change at the next step; REBUILD values change through nuka_world_rebuild_from_scene.
typedef enum nuka_param_tier_t {
    NUKA_PARAM_LIVE = 0,
    NUKA_PARAM_REBUILD = 1
} nuka_param_tier_t;

enum { NUKA_PARAM_INFO_VERSION = 1 };

// One tunable parameter. Its count elements are template links (all environments) or one world
// value; scene_field names the record field it persists to ("joint.damping"), empty for world options.
typedef struct nuka_param_info_t {
    uint32_t struct_size;
    uint32_t schema_version;
    char key[48];
    char unit[16];
    char scene_field[48];
    nuka_param_type_t type;
    nuka_param_tier_t tier;
    uint32_t count;
    double min_value;
    double max_value;
} nuka_param_info_t;

nuka_result_t nuka_world_param_count(nuka_world_handle world, uint32_t* out_count);
// Every entry must carry struct_size and schema_version; capacity must cover the count.
nuka_result_t nuka_world_get_param_infos(nuka_world_handle world, nuka_param_info_t* infos,
                                         uint32_t capacity);
nuka_result_t nuka_world_get_param(nuka_world_handle world, uint32_t param, uint32_t element,
                                   double* out_value);
// A LIVE value applies at the next step and is written to the retained scene. NOT_SUPPORTED for a
// REBUILD parameter and for an element without a scene record, such as a root link's joint.
nuka_result_t nuka_world_set_param(nuka_world_handle world, uint32_t param, uint32_t element,
                                   double value);

// Initialize struct_size and schema_version before querying execution information.
nuka_result_t nuka_world_get_execution_info(nuka_world_handle world, nuka_world_execution_info_t* out);
typedef struct nuka_device_resource_info_t {
    uint64_t free_bytes, total_bytes, thread_stack_limit_bytes;
    uint64_t graph_used_bytes, graph_reserved_bytes;
    uint32_t compute_units, threads_per_unit;
} nuka_device_resource_info_t;

typedef struct nuka_world_resource_info_t {
    uint64_t model_bytes, persistent_bytes, scratch_bytes;
    uint32_t env_count, contacts_per_env, rows_per_env, dofs_per_articulation;
    uint32_t bodies_per_env, particles_per_env;
} nuka_world_resource_info_t;

typedef struct nuka_resource_field_t {
    char name[64];
    uint64_t offset, bytes;
    uint32_t owner, arena;
} nuka_resource_field_t;

typedef struct nuka_execution_window_t {
    double gpu_ms;
    uint64_t graph_replays, nodes, kernels, clears, copies;
} nuka_execution_window_t;

nuka_result_t nuka_device_get_resource_info(nuka_device_handle device, nuka_device_resource_info_t* out);
nuka_result_t nuka_world_get_resource_info(nuka_world_handle world, nuka_world_resource_info_t* out);
uint32_t nuka_world_resource_field_count(nuka_world_handle world);
// The caller allocates resource_field_count entries before reading the immutable layout.
nuka_result_t nuka_world_get_resource_fields(nuka_world_handle world, nuka_resource_field_t* out);
nuka_result_t nuka_world_begin_execution_window(nuka_world_handle world, void** window);
nuka_result_t nuka_world_end_execution_window(nuka_world_handle world, void* window, nuka_execution_window_t* out);
// Wait for submitted physics work and report asynchronous device errors.
nuka_result_t nuka_world_synchronize(nuka_world_handle world);

// Restore all environments' initial physical state and clear solver history.
// Drive/task targets and view storage are preserved; any environment count is supported.
nuka_result_t nuka_world_reset(nuka_world_handle world);
// Reset an environment ID set; duplicates are ignored and count==0 is a no-op.
// Invalid IDs return NUKA_RESULT_INVALID_ARG without changing any environment.
nuka_result_t nuka_world_reset_envs(nuka_world_handle world,
                                    const uint32_t* env_ids,
                                    uint32_t count);

// Checkpoints capture persistent physics, observation output/history, and host configuration.
// Restores validate world ownership and generation and preserve existing view addresses.
nuka_result_t nuka_world_checkpoint_capture(nuka_world_handle world,
                                            nuka_checkpoint_handle* out);
nuka_result_t nuka_world_checkpoint_restore(nuka_world_handle world,
                                            nuka_checkpoint_handle checkpoint);
void nuka_checkpoint_destroy(nuka_checkpoint_handle checkpoint);
// Writes a checkpoint to a file with the layout hash: everything it holds but the render sensors' state.
nuka_result_t nuka_checkpoint_write(nuka_checkpoint_handle checkpoint, const char* path);
// Reads a written checkpoint into a world of the same layout; render sensor state comes from that world.
nuka_result_t nuka_world_checkpoint_read(nuka_world_handle world, const char* path,
                                         nuka_checkpoint_handle* out);
// The persistent physics bytes a checkpoint holds on the host.
nuka_result_t nuka_checkpoint_byte_size(nuka_checkpoint_handle checkpoint, uint64_t* out_bytes);
// 1 when both checkpoints hold the same layout, step count and persistent physics bytes, else 0.
nuka_result_t nuka_checkpoint_equal(nuka_checkpoint_handle a, nuka_checkpoint_handle b, uint32_t* out_equal);
// Hashes persistent physics, observation output/history, and host state in fixed order.
nuka_result_t nuka_world_state_hash(nuka_world_handle world, uint64_t* out_hash);

// Writes the world's retained scene as NKS (meshes go to the sibling .nka file).
nuka_result_t nuka_world_export_scene(nuka_world_handle world, const char* nks_path);
// Writes the retained scene as UsdPhysics (usda, usdc/.usd or usdz by extension), images copied beside it.
// `report` (may be NULL) takes one warning per line, cut to cap with a NUL; *out_len is its full length.
nuka_result_t nuka_world_export_usd(nuka_world_handle world, const char* usd_path, char* report, size_t cap,
                                    size_t* out_len);
// Directory the scene's relative render assets resolve against; empty for built scenes.
// NULL out queries *out_len (length without NUL); cap must hold the NUL.
nuka_result_t nuka_world_get_asset_root(nuka_world_handle world, char* out, size_t cap,
                                        size_t* out_len);

enum { NUKA_REBUILD_INITIAL_STATE = 1 };
// Recooks the world in place from a scene file with its creation options and increments the generation.
// Body and joint state carry over by unique body name unless INITIAL_STATE. Scene-authored sensors
// re-attach from the new scene, which then must give its cameras one size and lidars one fan
// (else INVALID_ARG before anything changes); explicitly attached or runtime-added sensors refuse it.
nuka_result_t nuka_world_rebuild_from_scene(nuka_world_handle world, const char* scene_path,
                                            uint32_t flags);
// The same rebuild from the scene's initial state with `env_count` environments; later rebuilds keep it.
nuka_result_t nuka_world_rebuild_with_envs(nuka_world_handle world, const char* scene_path,
                                           uint32_t env_count);
// The same rebuild, state carried over, stepping every `dt` seconds; later rebuilds keep it.
nuka_result_t nuka_world_rebuild_with_dt(nuka_world_handle world, const char* scene_path, float dt);
nuka_result_t nuka_world_get_generation(nuka_world_handle world, uint64_t* out_generation);
// A lease keeps the current generation's view storage alive across rebuilds until released.
nuka_result_t nuka_world_retain_storage(nuka_world_handle world, uint64_t* out_generation);
nuka_result_t nuka_world_release_storage(nuka_world_handle world, uint64_t generation);
// Recomputes link and body poses from joint state and base poses without advancing time.
nuka_result_t nuka_world_forward_kinematics(nuka_world_handle world);

enum { NUKA_IK_ORIENT = 1 };
// A point of link `link` (per-environment index, in its frame) that should reach world `position`;
// with NUKA_IK_ORIENT the link frame should also take world rotation (w, x, y, z). `weight` scales it.
typedef struct nuka_ik_target_t {
    uint32_t link;
    uint32_t flags;
    float point[3];
    float position[3];
    float rotation[4];
    float weight;
} nuka_ik_target_t;

enum { NUKA_IK_WRITE_POSITIONS = 1, NUKA_IK_WRITE_TARGETS = 2 };
typedef struct nuka_ik_request_t {
    uint32_t env;
    uint32_t root;  // the chains start below this link; UINT32_MAX is each target's articulation root
    const nuka_ik_target_t* targets;
    uint32_t target_count;
    uint32_t iterations;
    float rotation_weight;     // metres per radian of rotation error
    float damping;
    float max_step;            // rad or m per iteration and joint
    float position_tolerance;  // m
    float rotation_tolerance;  // rad
    uint32_t write;            // NUKA_IK_WRITE_* bits
} nuka_ik_request_t;

// Caller arrays: q takes links_per_env coordinates, the errors target_count values; each may be NULL.
typedef struct nuka_ik_result_t {
    float* q;
    float* position_error;  // m
    float* rotation_error;  // rad, 0 for targets without NUKA_IK_ORIENT
    uint32_t iterations;
    uint32_t joints;    // the free joints
    uint32_t at_limit;  // free joints ending at a limit
} nuka_ik_result_t;

// Damped least squares from the environment's joint positions over the moving joints between `root` and the
// targets, mimic followers following; WRITE_POSITIONS also zeroes their velocity and re-poses the world.
nuka_result_t nuka_world_solve_ik(nuka_world_handle world, const nuka_ik_request_t* request,
                                  nuka_ik_result_t* result);

// The CUDA stream the world executes on, as a cudaStream_t.
nuka_result_t nuka_world_get_stream(nuka_world_handle world, void** out_stream);
// Records a caller-created cudaEvent_t on the world's stream after all submitted work.
nuka_result_t nuka_world_record_event(nuka_world_handle world, void* cuda_event);

enum { NUKA_RUNTIME_INFO_VERSION = 1 };

typedef struct nuka_runtime_info_t {
    uint32_t struct_size;
    uint32_t schema_version;
    int32_t cuda_runtime_version;
    int32_t cuda_driver_version;
} nuka_runtime_info_t;

// Initialize struct_size and schema_version before querying. Extensions built against
// this library compare the runtime version with their own CUDA runtime at load.
nuka_result_t nuka_get_runtime_info(nuka_runtime_info_t* out);

typedef enum nuka_state_field_t {
    NUKA_FIELD_RIGID_BODY_TRANSFORM = 0,
    NUKA_FIELD_ARTICULATION_LINK_POSE = 1,
    NUKA_FIELD_JOINT_POSITION = 2,
    NUKA_FIELD_JOINT_VELOCITY = 3,
    NUKA_FIELD_OBSERVATIONS = 4,
    NUKA_FIELD_CONTACT_POINTS = 5,
    // WRITABLE (batched/multi-env path only). The per-env PD position-drive
    // target buffer the batched step reads every Step. The buffer view returned
    // for this field aliases the live device buffer the batched drive params'
    // drive_targets point at, so a caller may
    // write it IN PLACE (zero-copy) and the NEXT nuka_world_step picks up the
    // new targets. Layout is IDENTICAL to NUKA_FIELD_JOINT_POSITION: a
    // float[env_count * base_link_count], env-major, index
    // (env * base_link_count + link). Only links with a non-zero drive stiffness
    // (the actuated joints) act on their target; writes to other slots are
    // silently ignored by the solver. Gains (Kp/Kd/force_limits) stay at the
    // validated rest-pose hold values and are NOT exposed here. Read of this
    // field returns the CURRENT targets (initially the rest-pose hold targets).
    //
    // Per-env slot map (general, robot-agnostic):
    //   slot 0                   = the ROOT link. A drive write here is a NO-OP
    //                              unless a drive joint exists at the root. For a
    //                              FIXED-base scene q[0]/qd[0] are not meaningful
    //                              joint DOFs and the base world pose is the
    //                              constant cooked pose at
    //                              NUKA_FIELD_ARTICULATION_LINK_POSE[env*
    //                              base_link_count]; for a FLOATING-base scene that
    //                              same pose slot carries the LIVE integrated 6-DOF
    //                              base pose and base velocity lives in the engine's
    //                              link_velocity[root] (not exposed through this ABI).
    //   slots 1..base_link_count-1 = the child links in COOKED order. Only slots
    //                              with a non-zero NUKA_FIELD_DRIVE_STIFFNESS act on
    //                              their target (the actuated joints); these are the
    //                              only meaningful q/qd joint entries. A caller maps
    //                              cooked order to its own robot convention.
    // (See examples/ for a concrete per-robot slot mapping, e.g. the Go2 leg-joint
    // map the Python harness applies.)
    NUKA_FIELD_DRIVE_TARGET = 6,
    // READ (batched/multi-env path only). Per-LINK spatial velocity, env-major,
    // with the SAME indexing as NUKA_FIELD_ARTICULATION_LINK_POSE:
    // element_count == env_count * base_link_count, env e's base/root is element
    // e*base_link_count (root == local link 0). Each element is 6 contiguous
    // floats (24 bytes, element_stride_bytes == 6*sizeof(float)) in OMEGA-FIRST
    // spatial order: [wx,wy,wz, vx,vy,vz] (angular 0..2, linear 3..5).
    //
    // FRAME (read carefully -- this is the load-bearing contract for the policy):
    //   * ROOT / base slot (e*base_link_count, the floating-base trunk on
    //     go2_float.usda): the 6-vector is the live base spatial velocity
    //     expressed in the ROOT-LINK BODY frame -- it is ALREADY BODY-LOCAL, NOT
    //     world. (Confirmed from the engine's floating-base integrator: the linear
    //     part v[3..5] is rotated by R(base_rot) to obtain the world translation,
    //     and the angular part v[0..2] right-multiplies the orientation as a
    //     body-frame delta-quaternion. So no world->body rotation is needed; a
    //     policy harness applies only its own fixed axis/scale remap, NOT a
    //     world->body rotate -- doing both would double-rotate.) This root entry
    //     is CURRENT after Step(): the floating-base velocity integrate and the
    //     contact solve both write link_velocity[root] within the same Step, so
    //     unlike ARTICULATION_LINK_POSE it carries NO one-step lag.
    //   * NON-ROOT link slots (1..base_link_count-1): engine-internal Featherstone
    //     per-link velocities expressed in each link's OWN local (Featherstone)
    //     frame -- NOT body, NOT world. They are populated by the ABA pass and are
    //     non-zero, but their exact timing/frame is an implementation detail and is
    //     NOT a defined observation -- do NOT rely on them. They are exposed only
    //     for index symmetry with ARTICULATION_LINK_POSE; only the ROOT slot is
    //     meaningful for a floating-base base-velocity observation. (On a FIXED-
    //     base scene the root carries no free 6-DOF velocity; use go2_float.usda
    //     for a live base.)
    NUKA_FIELD_LINK_VELOCITY = 7,
    // WRITABLE (batched/multi-env path only). The per-env PD drive GAIN buffers
    // the batched step reads every Step (the batched drive params'
    // drive_stiffness / drive_damping / drive_force_limits point straight at them).
    // The view aliases the live device buffer so a caller may write it IN PLACE
    // (zero-copy) and the NEXT nuka_world_step picks up the new gains -- IDENTICAL
    // mechanism and layout to NUKA_FIELD_DRIVE_TARGET: float[env_count *
    // base_link_count], env-major, index (env*base_link_count + link), stride
    // sizeof(float). Per-env slot map matches DRIVE_TARGET: slot 0 = ROOT (its gain
    // is a no-op unless a root drive joint exists), slots 1..base_link_count-1 = the
    // actuated child joints in cooked order. To drive a trained policy at its
    // training PD gains, write its per-joint Kp/Kd on the actuated slots of every
    // env (see examples/ for a concrete per-robot gain set).
    //   STIFFNESS: per-joint proportional gain Kp (tau += Kp*(target - q)).
    //   DAMPING  : per-joint derivative gain   Kd (tau -= Kd*qdot).
    //   FORCE_LIMIT: per-joint symmetric torque clamp |tau| <= limit (Nm).
    // Reads return the CURRENT gains (initially the cooked rest-hold gains).
    NUKA_FIELD_DRIVE_STIFFNESS = 8,
    NUKA_FIELD_DRIVE_DAMPING = 9,
    NUKA_FIELD_DRIVE_FORCE_LIMIT = 10,
    // READ: authoritative roots, env-major E*K transforms [px,py,pz,qw,qx,qy,qz].
    // Current after step/reset; K is the articulation count per environment.
    NUKA_FIELD_BASE_POSE = 11,
    // WRITABLE: per-link torque command, aliased to DRIVE_TARGET; initially zero in torque modes.
    // Torque and actuator modes clamp the command before adding JOINT_FEEDFORWARD.
    NUKA_FIELD_TORQUE_INPUT = 12,
    // WRITABLE: per-link desired velocity; PD/CT use it as the derivative target.
    // Velocity mode applies DRIVE_STIFFNESS * (target - qdot), subject to the effort limit.
    NUKA_FIELD_VELOCITY_TARGET = 13,
    // WRITABLE: per-link no-load speed. Actuator mode uses a signed torque-speed envelope.
    // Non-positive speed disables the speed envelope; a non-positive effort limit is unlimited.
    NUKA_FIELD_ACTUATOR_NOLOAD_SPEED = 14,
    // WRITABLE: E*K world positions [x,y,z], one task per articulation, in environment order.
    // Available for every creation entry and environment count; state reset preserves inputs.
    NUKA_FIELD_TASK_TARGET = 15,
    // READ (batched/multi-env path only). p14a (v0.7): the NET per-LINK contact
    // WRENCH (force + torque) in the WORLD frame, aggregated over every contact
    // incident to the link. Maintained EAGERLY on the normal forward path (the
    // batched Step computes it after the contact solve, so it is always queryable
    // after a Step -- the MuJoCo-style "net contact force on this body" readout).
    //
    // LAYOUT: float[total_link_count * 6], total_link_count == env_count *
    // base_link_count, indexed by GLOBAL link index g == env*base_link_count +
    // local_link (the SAME env-major link indexing as ARTICULATION_LINK_POSE /
    // LINK_VELOCITY). element_count == env_count*base_link_count,
    // element_stride_bytes == 6*sizeof(float). Per element g:
    //   [g*6+0 .. g*6+2] = world contact FORCE F (N): sum over the link's contacts
    //                      of (n*lambda_n + t1*lambda_t1 + t2*lambda_t2)/dt, where
    //                      lambda is the solved per-contact IMPULSE (N*s) and dt is
    //                      the step dt (force = impulse/dt). +Z is up; a foot in a
    //                      stance carries a POSITIVE F_z (the ground pushes up).
    //   [g*6+3 .. g*6+5] = world contact TORQUE tau (N*m) about the LINK FRAME
    //                      ORIGIN (NOT the link COM): sum over the link's contacts
    //                      of (contact_point - link_frame_origin_world) x F_slot.
    //                      The reference point is the link frame origin
    //                      (ARTICULATION_LINK_POSE[g].position); a consumer that
    //                      wants the torque about the COM must shift it by the
    //                      link's COM offset.
    // Links with no incident contact this step read all-zero. dt is the dt of the
    // most-recent Step.
    //
    // SCOPE (v0.7 entry): this currently aggregates the RIGID FOOT-VS-GROUND
    // contacts only -- the sole contact model wired into the per-slot contact array
    // today. When p08 (SDF) / p11 (coupling) contact rows land they MUST be plumbed
    // into the same per-slot array for this to remain the FULL net wrench; until
    // then a link touched only by a not-yet-plumbed contact model reads zero here.
    // (NUKA_FIELD_CONTACT_NORMAL carries the matching caveat.)
    // Returns NUKA_RESULT_NOT_SUPPORTED on the single-env path.
    NUKA_FIELD_LINK_CONTACT_WRENCH = 16,
    // READ (batched/multi-env path only). p14a (v0.7): per-CONTACT-SLOT world unit
    // NORMAL. Together with CONTACT_POINTS / CONTACT_FORCE / CONTACT_LINK this is
    // the per-contact enumeration surface (the mj_contactForce parity: {point,
    // normal, normal-force, friction-force, link} per contact). Aliases the
    // engine's internal per-slot contact_normal buffer (zero-copy).
    //
    // LAYOUT: identical to CONTACT_POINTS -- Vec3 (3 floats) per contact slot,
    // slot_count == env_count * NUKA_MAX_CONTACT_SLOTS_PER_ENV, env-major.
    // element_count == slot_count, element_stride_bytes == 3*sizeof(float).
    // INACTIVE slots are zero. (For the current rigid foot-vs-ground detector the
    // active normal is always world +Z = (0,0,1).) Returns
    // NUKA_RESULT_NOT_SUPPORTED on the single-env path.
    NUKA_FIELD_CONTACT_NORMAL = 17,
    // READ (batched/multi-env path only). p14a (v0.7): per-CONTACT-SLOT contact
    // FORCE in the CONTACT BASIS -- the {Fn, Ft1, Ft2} = {lambda_n, lambda_t1,
    // lambda_t2}/dt components (normal force + the two friction-force magnitudes),
    // i.e. the solved per-slot IMPULSE scaled by 1/dt. Fn is the mj-parity normal
    // force; (Ft1,Ft2) are the friction-force magnitudes along the contact's two
    // world friction tangents.
    //
    // LAYOUT: 3 floats per contact slot, slot_count == env_count *
    // NUKA_MAX_CONTACT_SLOTS_PER_ENV, env-major. element_count == slot_count,
    // element_stride_bytes == 3*sizeof(float). INACTIVE slots are zero. NOTE: the
    // tangent BASIS VECTORS (t1,t2) are NOT exposed, so the per-contact world
    // friction DIRECTION cannot be reconstructed from this field alone -- the net
    // per-link WORLD friction force is available in LINK_CONTACT_WRENCH (which sums
    // n*lambda_n + t1*lambda_t1 + t2*lambda_t2 in world). Returns
    // NUKA_RESULT_NOT_SUPPORTED on the single-env path.
    NUKA_FIELD_CONTACT_FORCE = 18,
    // READ (batched/multi-env path only). p14a (v0.7): per-CONTACT-SLOT owning
    // GLOBAL LINK index (uint32). The CSR-by-link grouping is implicit: slots are
    // env-major at fixed stride NUKA_MAX_CONTACT_SLOTS_PER_ENV, and each slot
    // carries the global link (g == env*base_link_count + local_link) the contact
    // acts on, so a consumer enumerates contacts and groups them by link via this
    // field. Aliases the engine's internal per-slot contact_link buffer (zero-copy).
    //
    // LAYOUT: 1 uint32 per contact slot, slot_count == env_count *
    // NUKA_MAX_CONTACT_SLOTS_PER_ENV, env-major. element_count == slot_count,
    // element_stride_bytes == sizeof(uint32_t). dtype == 1 (UINT32) -- the ONLY
    // field with a non-float32 dtype. INACTIVE slots carry ~0u (0xFFFFFFFF, the
    // invalid-link sentinel). Returns NUKA_RESULT_NOT_SUPPORTED on the single-env
    // path.
    NUKA_FIELD_CONTACT_LINK = 19,
    // WRITE (per-env). Go2-on-stairs Phase 2a: per-env procedural-terrain TYPE
    // code (nuka::terrain::TerrainType: 0 = Flat (default seed), 1 = PyramidStairs,
    // 2 = InvertedPyramid, 3 = RandomBoxes). Aliases the engine's persistent
    // env_terrain_type Data field (zero-copy). (Historically the FUSED foot-contact
    // kernel sampled the shared procedural heightfield h(x,y) per env; L1-b deletes
    // that runtime — the general path's terrain is the cook-time baked heightfield
    // collidable, so this per-env type is now model-level / informational.) The
    // horizontal/
    // vertical TerrainParams (step_height etc.) are set ONCE at world create via
    // nuka_world_desc_t.terrain_* (model-level cook config, NOT per env).
    //
    // LAYOUT: 1 uint32 per env, env-major. element_count == env_count,
    // element_stride_bytes == sizeof(uint32_t). dtype == 1 (UINT32). A write is
    // picked up by the NEXT Step (the curriculum-control surface). Persistent =>
    // round-trips reset.
    NUKA_FIELD_ENV_TERRAIN_TYPE = 20,
    // WRITE (per-env). Go2-on-stairs Phase 2a: per-env procedural-terrain
    // DIFFICULTY scale (f32, default seed 1.0 == unscaled). Aliases the engine's
    // persistent env_terrain_difficulty Data field (zero-copy). The foot-contact
    // kernel multiplies the terrain's VERTICAL feature magnitude (step_height +
    // grid_height_max) by this scale before sampling the heightfield, so a
    // curriculum can raise/lower the SAME step/box footprint per env (legged_gym
    // terrain-level convention). diff == 1.0 leaves the terrain unscaled; 0.0
    // collapses it to a flat plane at ground_height. Ignored by type-0 (Flat) envs
    // (SampleTerrainHeight returns ground_height regardless), so the default seed
    // is byte-identical to the legacy flat path.
    //
    // LAYOUT: 1 float per env, env-major. element_count == env_count,
    // element_stride_bytes == sizeof(float). dtype == 0 (FLOAT32). A write is
    // picked up by the NEXT Step. Persistent => round-trips reset.
    NUKA_FIELD_ENV_TERRAIN_DIFFICULTY = 21,
    // WRITE (per-link). T3 unified-actuator feed-forward joint force: a direct
    // applied torque/force ADDED to the actuator output of EVERY control mode
    // (tau += joint_f), AFTER actuator saturation (MuJoCo qfrc_applied -- a user
    // force, not an actuator output, so the actuator force-range does NOT clamp
    // it). This is the channel for gravity compensation, computed-torque
    // feed-forward, and RL torque residuals. Aliases the engine's persistent
    // joint_f Data field (zero-copy). DEFAULT 0 => no-op (the drive is then
    // byte-identical to a world without feed-forward). LAYOUT and slot map are
    // IDENTICAL to NUKA_FIELD_DRIVE_TARGET: float[env_count * base_link_count],
    // env-major, element_stride_bytes == sizeof(float), dtype == 0 (FLOAT32);
    // slot 0 is the inert root, actuated joints are [1..base_link_count). A write
    // is picked up by the NEXT Step. Persistent => round-trips reset.
    NUKA_FIELD_JOINT_FEEDFORWARD = 22,
    // READ (per-particle). The live particle WORLD positions of a coupled world
    // (cloth/MPM), env-major: float3[env_count * particles_per_env], index
    // (env * particles_per_env + particle), element_stride_bytes == 3*sizeof(float),
    // dtype == 0 (FLOAT32). MPM points occupy [0, n_mpm), vertex blocks the rest. On a world with
    // no particles it RESOLVES as an empty view (element_count == 0), not an error.
    // Read of the coupled medium's state; a write is overwritten by the NEXT Step.
    NUKA_FIELD_PARTICLE_POSITION = 23,
    // READ (per-particle). The live particle velocities, same layout/stride/dtype as
    // NUKA_FIELD_PARTICLE_POSITION; an empty view (element_count == 0) when no particles.
    NUKA_FIELD_PARTICLE_VELOCITY = 24,
    // READ/WRITE (general). The cooked rigid-body SoA world-frame LINEAR / ANGULAR
    // velocity (runtime::rigid::BodyState; nk FieldId BodyLinearVelocity /
    // BodyAngularVelocity). Vec3 per body, env-major: element_count ==
    // env_count*bodies_per_env, index (env*bodies_per_env + body),
    // element_stride_bytes == 3*sizeof(float), dtype == 0 (FLOAT32). Bodies are the
    // cooked rigid rows (articulation link bodies + free rigid bodies). A body-free
    // world resolves an empty view (element_count == 0). Written via
    // nuka_world_upload_field to park a free body (e.g. the water-demo bunny hold).
    NUKA_FIELD_BODY_LINEAR_VELOCITY = 25,
    NUKA_FIELD_BODY_ANGULAR_VELOCITY = 26,
    // READ: lower/upper joint-limit impulse per link, stored as two float32 values.
    NUKA_FIELD_JOINT_LIMIT_IMPULSE = 27,
    // READ: requested effort, saturated actuator effort, and a float32 0/1 clamp flag.
    NUKA_FIELD_ACTUATOR_EFFORT_REQUESTED = 28,
    NUKA_FIELD_ACTUATOR_EFFORT = 29,
    NUKA_FIELD_ACTUATOR_SATURATED = 30,
    // WRITE: E*K world-frame OSC W-first quaternions, one per articulation.
    // A near-zero quaternion selects position-only control.
    NUKA_FIELD_TASK_ROTATION_TARGET = 31,
    // WRITE: E*K rigid local frames on osc_task_link, [px,py,pz,qw,qx,qy,qz].
    // A near-zero quaternion means identity; translation selects the offset point.
    NUKA_FIELD_TASK_LOCAL_POSE = 32,
    // READ (per-contact-slot uint32). Kinds use nuka_contact_side_kind_t.
    NUKA_FIELD_CONTACT_SIDE_A_KIND = 33,
    NUKA_FIELD_CONTACT_SIDE_B_KIND = 34,
    // READ (per-contact-slot uint32). Global body/link/particle/grid index or point-endpoint range index.
    // Static and inactive sides have UINT32_MAX; shape proxies resolve to their owner.
    NUKA_FIELD_CONTACT_SIDE_A_INDEX = 35,
    NUKA_FIELD_CONTACT_SIDE_B_INDEX = 36,
    // WRITE: world COM force [N], float3 per dynamic free-body owner, env-major.
    // Consumed and cleared by the next step; continuous loads must be rewritten each step.
    NUKA_FIELD_BODY_FORCE = 37,
    // WRITE: world torque [N*m] about COM; same layout and lifetime as BODY_FORCE.
    NUKA_FIELD_BODY_TORQUE = 38,
    // READ: environment validity bits, uint32 per environment; synchronize before host inspection.
    NUKA_FIELD_ENV_STATUS = 39,
    // READ: last free-body drift residual, iteration count and nuka_gyro_status_t, env-major.
    NUKA_FIELD_BODY_GYRO_RESIDUAL = 40,
    NUKA_FIELD_BODY_GYRO_ITERATIONS = 41,
    NUKA_FIELD_BODY_GYRO_STATUS = 42,
    // READ: row-major 3x3 per particle, env-major; elastic Fe for plastic materials.
    NUKA_FIELD_PARTICLE_DEFORMATION_GRADIENT = 45,
    // READ: multiplicative Fp; empty when no material requires plastic history.
    NUKA_FIELD_PARTICLE_PLASTIC_DEFORMATION_GRADIENT = 46,
    // READ: cumulative equivalent plastic strain alpha; scalar per particle.
    NUKA_FIELD_PARTICLE_EQUIVALENT_PLASTIC_STRAIN = 47,
    // READ: latest interval counts; peak since reset. Counts except retained are uint64.
    NUKA_FIELD_GRID_CONTACT_ATTEMPTED = 48,
    NUKA_FIELD_GRID_CONTACT_RETAINED = 49,
    NUKA_FIELD_GRID_CONTACT_PEAK = 50,
    NUKA_FIELD_GRID_CONTACT_OVERFLOW = 51,
    // READ: outer-step impulses; body moments use COM/link origin, boundaries world origin.
    NUKA_FIELD_MPM_BODY_IMPULSE = 52,
    NUKA_FIELD_MPM_BODY_ANGULAR_IMPULSE = 53,
    NUKA_FIELD_MPM_BOUNDARY_IMPULSE = 54,
    NUKA_FIELD_MPM_BOUNDARY_ANGULAR_IMPULSE = 55,
    // WRITABLE: per-link feedforward acceleration for computed-torque control, initially zero.
    NUKA_FIELD_ACCELERATION_TARGET = 56,
    // WRITABLE: E*K posture acceleration gains for OSC, initially 10 and 2*sqrt(10).
    NUKA_FIELD_TASK_NULLSPACE_STIFFNESS = 57,
    NUKA_FIELD_TASK_NULLSPACE_DAMPING = 58,
    // READ: uint32[first,count] per endpoint; first addresses POINT_ENDPOINT_TERMS globally.
    NUKA_FIELD_POINT_ENDPOINT_RANGES = 59,
    // READ: uint8 records with the layout of nuka_point_endpoint_term_t, valid for the latest interval.
    NUKA_FIELD_POINT_ENDPOINT_TERMS = 60,
    // Eight uint64 values per environment: float error bits in high 32, ~global_row in low 32.
    // Order: normal/tangent velocity, normal velocity/impulse violation, complementarity, cone, work, dissipation.
    NUKA_FIELD_CONTACT_SOLVE_METRICS = 61,
    // Two uint32 counts per environment: evaluated contact blocks, invalid blocks; latest solve only.
    NUKA_FIELD_CONTACT_SOLVE_COUNTS = 62,
    // WRITE: per-particle inverse mass. Zero selects a driven kinematic particle.
    NUKA_FIELD_PARTICLE_INV_MASS = 63,
    // WRITE: next position of each zero-inverse-mass particle, in world coordinates.
    NUKA_FIELD_PARTICLE_KINEMATIC_TARGET = 64,
    NUKA_FIELD_OGC_CONTACT_COUNT = 65,
    // READ: per-environment particle, joint, and free-body truncations in the latest step.
    NUKA_FIELD_DAT_TRUNCATION_COUNT = 66,
    // READ: per-environment physical kinetic-energy loss from DAT; negative means energy gain.
    NUKA_FIELD_DAT_TRUNCATION_ENERGY = 67,
    // READ: per-environment invalid primitives or inseparable pairs in the latest step.
    NUKA_FIELD_DAT_FAILURE_COUNT = 68,
    // READ: per-environment 0 or 1, whether the latest step exceeded the 20 mm query limit.
    NUKA_FIELD_DAT_QUERY_LIMIT_COUNT = 69,
    // READ: per environment, current contact points and accepted warm starts.
    NUKA_FIELD_CONTACT_WARM_START_COUNTS = 70,
    NUKA_FIELD_DAT_PARTICLE_TRUNCATION_COUNT = 71,
    NUKA_FIELD_DAT_PARTICLE_TRUNCATION_ENERGY = 72,
    NUKA_FIELD_DAT_JOINT_TRUNCATION_COUNT = 73,
    NUKA_FIELD_DAT_JOINT_TRUNCATION_ENERGY = 74,
    NUKA_FIELD_DAT_BODY_TRUNCATION_COUNT = 75,
    NUKA_FIELD_DAT_BODY_TRUNCATION_ENERGY = 76,
    NUKA_FIELD_MESH_OGC_PAIR_COUNT = 77,
    NUKA_FIELD_MESH_SDF_PAIR_COUNT = 78,
    // READ: per environment, velocity sweeps of the latest solve; it stops once SOLVER_RESIDUAL passes the
    // velocity tolerance or the sweep budget is spent.
    NUKA_FIELD_VBD_VELOCITY_SWEEP_COUNT = 79,
    // READ: per environment, vertex colors, dynamic row colors, chain rows, chain islands, rows of
    // the longest island chain, and articulation Schur rows of the latest colored solve.
    NUKA_FIELD_SOLVER_COLOR_COUNTS = 80,
    // READ: actual per-particle DAT fraction; valid after a completed DAT step until reset.
    NUKA_FIELD_DAT_PARTICLE_MOTION_FRACTION = 81,
    // READ: [Newton correction m/s, stationarity force N], float bits high32 / ~particle low32.
    // Valid after a completed solve with contact diagnostics enabled, until reset.
    NUKA_FIELD_VBD_SOLVE_METRICS = 82,
    // Interval records are env-major, with one record per physical substep; demand before stepping.
    // Valid is cleared on reset; nonzero coverage status prevents energy acceptance.
    NUKA_FIELD_ENERGY_LEDGER = 83,
    NUKA_FIELD_ENERGY_LEDGER_STATUS = 84,
    // Signed virtual work of the vertex force residual against its candidate displacement, in J.
    // Same interval extent, demand and validity as ENERGY_LEDGER; does not alter its balance.
    NUKA_FIELD_VBD_FORCE_RESIDUAL_WORK = 85,
    // READ: env/substep/stage/column; five stages and 39 columns, reset invalidates energy VALID.
    NUKA_FIELD_PHYSICS_STAGE_METRICS = 86,
    // READ: env/substep/eight counts; valid when the corresponding energy VALID equals one.
    NUKA_FIELD_CONTACT_AUDIT_COUNTS = 87,
    // READ: seven maxima packed as float bits in high32 and complemented global row ID in low32.
    NUKA_FIELD_CONTACT_AUDIT_METRICS = 88,
    // READ: env-major VBD vertices, actual effective dt in seconds for the last physical substep.
    // Energy VALID controls validity; use the recorded dt to distinguish mixed BE/BDF2 restarts.
    NUKA_FIELD_VBD_EFFECTIVE_DT = 89,
    // READ: shared cooked VBD elements as nuka_vbd_element_t, carried by sixteen uint32 words.
    NUKA_FIELD_VBD_ELEMENTS = 90,
    // READ: env/389 uint64 from the last DAT pass: radius/motion/degenerate/overlap totals, unrecorded, then
    // (reason<<60|owner<<30|owner, count, max overlap f32 bits); owner = kind(body,link,surface,static)<<28|index.
    NUKA_FIELD_DAT_FAILURE_WITNESS = 91,
    // READ: per articulation, four uint64 for the vertex motion, particle pair, body pair and chain
    // travel bounds of the last DAT pass: f32 fraction bits high32, body << 16 | counterpart low32.
    NUKA_FIELD_DAT_ARTICULATION_WITNESS = 92,
    // READ: per articulation, the motion fraction the last DAT pass applied.
    NUKA_FIELD_DAT_ARTICULATION_FRACTION = 93,
    // READ: per body, the predicted vertex speed bound and query radius of the last detection.
    NUKA_FIELD_DAT_BODY_SPEED = 94,
    NUKA_FIELD_DAT_BODY_QUERY_RADIUS = 95,
    // READ: per body, the largest untruncated vertex displacement rate of the last DAT pass.
    NUKA_FIELD_DAT_BODY_MOTION = 96,
    // READ: per link, the untruncated joint speed of the last DAT pass and the detection bound.
    NUKA_FIELD_DAT_JOINT_MOTION = 97,
    NUKA_FIELD_DAT_JOINT_RATE = 98,
    // READ: per environment, uint64 nanoseconds for eight solve phases, then five tile setup subphases:
    // mass response, incidence, coloring, Jacobian loading and initial residual; setup includes its subphases.
    NUKA_FIELD_SOLVER_PHASE_TIME = 99,
    // READ: per VBD vertex, twenty float32 columns from the last solve call, environment-major.
    NUKA_FIELD_VBD_SOLVE_AUDIT = 100,
    // READ: per contact slot, the absolute row offset and normal point count.
    NUKA_FIELD_UCONTACT_ROW_RANGE = 103,
    // WRITE: per link, world force [N] through its COM followed by world torque [N*m], env-major.
    // Dynamics consume it over one outer step, then clear it with BODY_FORCE.
    NUKA_FIELD_LINK_WRENCH = 105,
    // READ: per environment and CONTACT_SOLVE_METRICS entry, 20 float32 terms of that worst row: dual[3],
    // velocity[3], projected normal and tangent[2] impulse, penalty, compliance, response xx yy zz xy xz yz, mu[2], slot bits.
    NUKA_FIELD_CONTACT_SOLVE_WORST_TERMS = 106,
    // READ: per contact slot, the number of valid UCONTACT_POINT entries, environment-major.
    NUKA_FIELD_UCONTACT_COUNT = 107,
    // READ: per contact slot, four world contact points (xyz each), environment-major.
    NUKA_FIELD_UCONTACT_POINT = 108,
    // READ: per environment, four float32 final residuals in m/s: contact normal, tangent, other rows,
    // and the maximum of block primal and coupling residuals.
    NUKA_FIELD_SOLVER_RESIDUAL = 109,
    // READ: per environment and sweep, the four SOLVER_RESIDUAL classes after that sweep; filled while
    // sweep telemetry is on, for at most the sweep budget set when it was turned on.
    NUKA_FIELD_SOLVER_SWEEP_TRACE = 110,
    // Per environment and sampled sweep: Newton iterations, line evaluations and factorizations.
    NUKA_FIELD_SOLVER_WORK_TRACE = 111,
    // READ: per environment and sampled sweep, the largest dense block velocity change in m/s.
    NUKA_FIELD_SOLVER_DENSE_MOTION_TRACE = 112,
    NUKA_FIELD_JOINT_LAW_SOURCE = 113,
    NUKA_FIELD_JOINT_LIMIT_REFERENCE = 114,
    NUKA_FIELD_JOINT_FRICTION_REFERENCE = 115,
    NUKA_FIELD_JOINT_LIMIT_MARGIN = 116,
    NUKA_FIELD_BODY_INVWEIGHT0 = 117,
    NUKA_FIELD_DOF_INVWEIGHT0 = 118,
    NUKA_FIELD_CONTACT_PROFILES = 119,
    // READ/WRITE per link: a spherical joint's rotation of the child in its parent joint frame (wxyz; identity
    // elsewhere) and its three DOF velocities, the relative angular velocity in the child frame (rad/s).
    NUKA_FIELD_JOINT_ROTATION = 120,
    NUKA_FIELD_JOINT_ANGULAR_VELOCITY = 121,
    // READ: the vertex-block coarse hierarchy: each level's first node (levels + 1 entries), then per level and
    // vertex up to three parent nodes (UINT32_MAX unused) and their interpolation weights.
    NUKA_FIELD_VBD_COARSE_LEVEL_NODES = 122,
    NUKA_FIELD_VBD_COARSE_PARENTS = 123,
    NUKA_FIELD_VBD_COARSE_WEIGHTS = 124,
    // Per particle, environment-major: APIC velocity gradient C (row-major, s^-1), rest volume (m^3), material ID.
    NUKA_FIELD_PARTICLE_AFFINE_VELOCITY = 125,
    NUKA_FIELD_PARTICLE_REST_VOLUME = 126,
    NUKA_FIELD_PARTICLE_MATERIAL_ID = 127,
    // READ: four world normals per contact slot, directed toward side A; entries below UCONTACT_COUNT are valid.
    NUKA_FIELD_UCONTACT_NORMAL = 128,
    // READ: four signed depths per contact slot (m); positive means overlap, negative means separation.
    NUKA_FIELD_UCONTACT_DEPTH = 129,
    // READ: four world geometry witnesses per OGC contact slot on side A and side B; undefined for other detectors.
    NUKA_FIELD_UCONTACT_WITNESS_A = 130,
    NUKA_FIELD_UCONTACT_WITNESS_B = 131,
    // READ: per contact slot, nuka_contact_law_t and the detector's friction override (OGC and MPM contacts).
    NUKA_FIELD_UCONTACT_LAW = 132,
    NUKA_FIELD_UCONTACT_FRICTION = 133
} nuka_state_field_t;

typedef enum nuka_contact_law_t {
    NUKA_CONTACT_LAW_COMPLIANT = 0,
    NUKA_CONTACT_LAW_VELOCITY = 1,
    NUKA_CONTACT_LAW_SPECULATIVE = 2
} nuka_contact_law_t;

typedef enum nuka_vbd_element_kind_t {
    NUKA_VBD_TRIANGLE = 0, NUKA_VBD_HINGE = 1, NUKA_VBD_SPRING = 2, NUKA_VBD_ROD_BEND = 3,
    NUKA_VBD_TET = 4
} nuka_vbd_element_kind_t;

typedef enum nuka_vbd_membrane_t {
    NUKA_VBD_MEMBRANE_STVK = 0, NUKA_VBD_MEMBRANE_NEO_HOOKEAN = 1
} nuka_vbd_membrane_t;

typedef struct nuka_vbd_element_t {
    uint32_t kind;
    uint32_t vertex[4];
    float rest[8];
    float damping;
    uint32_t law;  // nuka_vbd_membrane_t of a triangle
    uint32_t reserved;
} nuka_vbd_element_t;

typedef enum nuka_energy_column_t {
    NUKA_ENERGY_BEGIN_KINETIC = 0, NUKA_ENERGY_BEGIN_GRAVITY, NUKA_ENERGY_BEGIN_ELASTIC,
    NUKA_ENERGY_FREE_KINETIC, NUKA_ENERGY_SOLVED_KINETIC,
    NUKA_ENERGY_BEFORE_DAT_KINETIC, NUKA_ENERGY_BEFORE_DAT_GRAVITY, NUKA_ENERGY_BEFORE_DAT_ELASTIC,
    NUKA_ENERGY_END_KINETIC, NUKA_ENERGY_END_GRAVITY, NUKA_ENERGY_END_ELASTIC,
    NUKA_ENERGY_DRIVE_WORK, NUKA_ENERGY_AERO_WORK, NUKA_ENERGY_EXTERNAL_WORK,
    NUKA_ENERGY_KINEMATIC_ELASTIC_WORK, NUKA_ENERGY_KINEMATIC_CONTACT_WORK,
    NUKA_ENERGY_FRICTION_LOSS, NUKA_ENERGY_NORMAL_LOSS, NUKA_ENERGY_LIMIT_LOSS,
    NUKA_ENERGY_PASSIVE_LOSS, NUKA_ENERGY_RAYLEIGH_LOSS,
    NUKA_ENERGY_POSITION_POTENTIAL, NUKA_ENERGY_DAT_KINETIC_LOSS, NUKA_ENERGY_DAT_POTENTIAL,
    NUKA_ENERGY_RESIDUAL, NUKA_ENERGY_THROUGHPUT, NUKA_ENERGY_DT,
    NUKA_ENERGY_ROW_RATE_WORK, NUKA_ENERGY_ROW_PHYSICAL_IMPULSE_WORK,
    NUKA_ENERGY_UNCLASSIFIED_ROW_WORK, NUKA_ENERGY_VALID, NUKA_ENERGY_COLUMN_COUNT
} nuka_energy_column_t;

typedef enum nuka_energy_status_t {
    NUKA_ENERGY_STATUS_NONFINITE = 1u << 0,
    NUKA_ENERGY_STATUS_UNCLASSIFIED_ROWS = 1u << 1,
    NUKA_ENERGY_STATUS_OTHER_PARTICLE_MATERIAL = 1u << 2,
    NUKA_ENERGY_STATUS_GRID_MATERIAL = 1u << 3,
    NUKA_ENERGY_STATUS_EXTERNAL_BODY_LOAD = 1u << 4,
    NUKA_ENERGY_STATUS_FLOATING_POSITION_WORK = 1u << 5,
    NUKA_ENERGY_STATUS_RIGID_ANGULAR_POSITION_WORK = 1u << 6,
    NUKA_ENERGY_STATUS_ARMATURE_ENERGY = 1u << 7,
    NUKA_ENERGY_STATUS_PHYSICS_FAILURE = 1u << 8
} nuka_energy_status_t;

typedef enum nuka_env_status_t {
    NUKA_ENV_STATUS_VALID = 0,
    NUKA_ENV_STATUS_PAIR_OVERFLOW = 1u << 0,
    NUKA_ENV_STATUS_NEIGHBOR_OVERFLOW = 1u << 1,
    NUKA_ENV_STATUS_DOF_OVERFLOW = 1u << 2,
    NUKA_ENV_STATUS_MPM_GRID_ESCAPE = 1u << 3,
    NUKA_ENV_STATUS_MPM_ONE_WAY_BODY = 1u << 4,
    NUKA_ENV_STATUS_GYRO_FAILURE = 1u << 5,
    // Invalid contact ownership or endpoint indices; cleared by world reset.
    NUKA_ENV_STATUS_INVALID_ENDPOINT = 1u << 6,
    // Missing or invalid sampled contact geometry; cleared by world reset.
    NUKA_ENV_STATUS_CONTACT_GEOMETRY_UNAVAILABLE = 1u << 7,
    // Material integration failed; the prior elastic/plastic history is retained.
    NUKA_ENV_STATUS_CONSTITUTIVE_FAILURE = 1u << 8,
    NUKA_ENV_STATUS_GRID_CONTACT_OVERFLOW = 1u << 9,
    NUKA_ENV_STATUS_CONTROL_FAILURE = 1u << 10,
    NUKA_ENV_STATUS_SENSOR_QUEUE_OVERFLOW = 1u << 11,
    // Invalid or initially intersecting DAT geometry; latched until reset.
    NUKA_ENV_STATUS_DAT_FAILURE = 1u << 12,
    // Invalid or nonfinite block response; latched until reset.
    NUKA_ENV_STATUS_SOLVER_FAILURE = 1u << 13
} nuka_env_status_t;

typedef enum nuka_gyro_status_t {
    NUKA_GYRO_OK = 0,
    NUKA_GYRO_NOT_CONVERGED = 1,
    NUKA_GYRO_INVALID_INPUT = 2
} nuka_gyro_status_t;

typedef enum nuka_contact_side_kind_t {
    NUKA_CONTACT_SIDE_RIGID = 0,
    NUKA_CONTACT_SIDE_LINK = 1,
    NUKA_CONTACT_SIDE_PARTICLE = 2,
    NUKA_CONTACT_SIDE_STATIC = 3,
    NUKA_CONTACT_SIDE_GRID = 4,
    NUKA_CONTACT_SIDE_POINT_ENDPOINT = 5
} nuka_contact_side_kind_t;

typedef struct nuka_point_endpoint_range_t {
    uint32_t first;
    uint32_t count;
} nuka_point_endpoint_range_t;

// A column-major velocity map: endpoint velocity = sum_i column_i * point_velocity_i.
// Its transpose distributes the endpoint impulse to the referenced particle or grid node.
typedef struct nuka_point_endpoint_term_t {
    uint32_t kind;
    uint32_t index;
    float column[3][3];
} nuka_point_endpoint_term_t;

// Solver telemetry, one record per physical substep. SOLVE records the solve; SWEEPS adds the
// residual at sweeps 1, 2, 3, 4, 6, 8, 12, ...; ENERGY the ledger totals; VERTICES per-vertex residuals.
typedef enum nuka_telemetry_bits_t {
    NUKA_TELEMETRY_SOLVE = 1u << 0,
    NUKA_TELEMETRY_SWEEPS = 1u << 1,
    NUKA_TELEMETRY_ENERGY = 1u << 2,
    NUKA_TELEMETRY_VERTICES = 1u << 3
} nuka_telemetry_bits_t;

// CAP: the sweep budget ran out above the alert residual. CONVERGED: sweeps stopped at the tolerance.
typedef enum nuka_telemetry_flags_t {
    NUKA_TELEMETRY_CAP = 1u << 0,
    NUKA_TELEMETRY_CONVERGED = 1u << 1,
    NUKA_TELEMETRY_ENERGY_VALID = 1u << 2
} nuka_telemetry_flags_t;

typedef enum nuka_telemetry_element_t {
    NUKA_TELEMETRY_ELEMENT_NONE = 0,
    NUKA_TELEMETRY_ELEMENT_PARTICLE = 1,  // worst_index is a global particle index
    NUKA_TELEMETRY_ELEMENT_ROW = 2        // worst_index is a global row slot
} nuka_telemetry_element_t;

#define NUKA_TELEMETRY_SAMPLES 24

// Residuals are velocities [m/s]: the vertex Newton correction and the contact natural velocity.
// Maxima run over every environment; a row's sides use nuka_contact_side_kind_t kinds, where a
// link side carries its articulation index.
typedef struct nuka_telemetry_record_t {
    uint32_t sequence;          // records written since the mask was set
    uint32_t substep;
    uint32_t sweeps;
    uint32_t sweep_budget;
    uint32_t flags;             // nuka_telemetry_flags_t
    uint32_t contacts;          // evaluated contact blocks
    float vertex_residual;
    float vertex_force;         // largest vertex stationarity force [N]
    float contact_residual;
    float max_penetration;      // [m]
    uint32_t penetration_row;   // UINT32_MAX without penetration
    float acceleration;         // largest multiplier momentum coefficient (FISTA beta), 0 without momentum
    float energy;               // end kinetic + gravity + elastic energy [J]
    float energy_residual;      // ledger balance residual [J]
    uint32_t worst_kind;        // nuka_telemetry_element_t
    uint32_t worst_index;
    float worst_residual;
    uint32_t worst_side_kind[2];
    uint32_t worst_side_index[2];
    uint32_t sample_count;
    uint32_t sample_sweep[NUKA_TELEMETRY_SAMPLES];
    float sample_residual[NUKA_TELEMETRY_SAMPLES];
    uint32_t reserved[2];
} nuka_telemetry_record_t;

typedef struct nuka_buffer_view_t {
    void* device_ptr;
    size_t element_count;
    uint32_t element_stride_bytes;
    // Element scalar type: 0 == float32, 1 == uint32, 2 == uint64, 3 == uint8.
    uint8_t dtype;
} nuka_buffer_view_t;

nuka_result_t nuka_world_get_buffer_view(nuka_world_handle world,
                                         nuka_state_field_t field,
                                         nuka_buffer_view_t* out);

// Host<->field byte transfer for ANY served field (1:1 with nk::Data::UploadField /
// DownloadField). `bytes` is a HOST buffer of `nbytes`; `byte_offset` is relative to
// the field's Arena segment base. This is the GENERAL control verb a demo's
// choreography uses to re-seat NUKA_FIELD_BASE_POSE, zero NUKA_FIELD_LINK_VELOCITY,
// park a body, etc. -- the single byte-level write path (the writable buffer-view
// fields are also writable this way; a read-only view's backing Data field is too).
// INVALID_ARG if [byte_offset, byte_offset+nbytes) exceeds the field's segment (a
// LOUD failure, never a silent clamp) or bytes==NULL with nbytes>0. NOT_SUPPORTED if
// the field is unknown or has no nk Data field (e.g. a control-input alias, or a
// capacity-0 field on this scene). nbytes==0 is a no-op OK.
nuka_result_t nuka_world_upload_field(nuka_world_handle world,
                                      nuka_state_field_t field,
                                      const void* bytes, size_t nbytes,
                                      size_t byte_offset);
nuka_result_t nuka_world_download_field(nuka_world_handle world,
                                        nuka_state_field_t field,
                                        void* bytes, size_t nbytes,
                                        size_t byte_offset);

// Mask 0 removes every telemetry kernel and write; a nonzero mask restarts the 1024-record ring.
// A spent budget is a cap above alert_velocity [m/s], or above the solver tolerance when it is set.
nuka_result_t nuka_world_set_telemetry_mask(nuka_world_handle world, uint32_t mask,
                                            float alert_velocity);
nuka_result_t nuka_world_get_telemetry_mask(nuka_world_handle world, uint32_t* out_mask);
// Copies up to `capacity` records written since the last read, oldest first, synchronizing the
// world stream; *out_dropped counts the records the ring overwrote before they were read.
nuka_result_t nuka_world_read_telemetry(nuka_world_handle world, nuka_telemetry_record_t* out,
                                        uint32_t capacity, uint32_t* out_count,
                                        uint32_t* out_dropped);
// Per particle, environment-major, the largest Newton correction component [m/s] of the last solve,
// 0 outside the vertex blocks. Needs NUKA_TELEMETRY_VERTICES; copies up to `capacity` values of *out_count.
nuka_result_t nuka_world_read_telemetry_vertices(nuka_world_handle world, float* out, size_t capacity,
                                                 size_t* out_count);

// One cooked render surface of the world's particles; its triangles index the global particle array.
typedef struct nuka_particle_surface_t {
    uint32_t triangle_count;
    uint32_t particle_first;
    uint32_t particle_count;
    uint32_t smooth_iters;       // Laplacian passes over the rendered positions
    float smooth_lambda;
    float normal_offset;         // outward inflation along the smooth normal [m]
    uint32_t render_material_id; // a scene render material, or UINT32_MAX
    uint32_t medium;             // the scene medium it draws
} nuka_particle_surface_t;

nuka_result_t nuka_world_get_particle_surface_count(nuka_world_handle world, uint32_t* out_count);
// Describes surface `index` and copies up to `capacity` of its 3 * triangle_count indices.
nuka_result_t nuka_world_get_particle_surface(nuka_world_handle world, uint32_t index,
                                              nuka_particle_surface_t* out, uint32_t* triangles,
                                              size_t capacity);

// Resolve the cooked DOF NAME of articulation-local link `link_index` (0 == the
// root; 1..base_link_count-1 == the actuated joints in COOKED order). Writes the
// NUL-terminated name into `out` and the length (excluding NUL) into *out_len (may
// be NULL). The name is the joint whose CHILD is that link (else the link body's
// name; empty for an unnamed/root link). A NULL `out` is a SIZE QUERY: it returns OK
// with *out_len set to the needed length (allocate len+1, retry). INVALID_ARG if
// link_index is out of range, or if a non-NULL `cap` cannot hold name + NUL (*out_len
// is still set to the needed length). NOT_SUPPORTED if no cooked articulation is
// retained. Resolves through the cooked articulation joint table, so a regex
// name->column map is correct under CookArticulations joint reordering.
nuka_result_t nuka_world_get_dof_name(nuka_world_handle world,
                                      uint32_t link_index, char* out, size_t cap,
                                      size_t* out_len);

typedef enum nuka_kinematic_joint_t {
    NUKA_KINEMATIC_ROOT = 0,
    NUKA_KINEMATIC_FIXED = 1,
    NUKA_KINEMATIC_REVOLUTE = 2,
    NUKA_KINEMATIC_PRISMATIC = 3,
    // Three rotational DOFs; the joint's rotation is NUKA_FIELD_JOINT_ROTATION at the child link.
    NUKA_KINEMATIC_SPHERICAL = 4
} nuka_kinematic_joint_t;

// Immutable encoder kinematics in cooked per-environment link order; roots have parent_index UINT32_MAX.
// local_pose is the zero-coordinate parent-to-child transform (xyz, wxyz); axis is in its child frame.
typedef struct nuka_kinematic_link_t {
    uint32_t parent_index;
    uint32_t articulation_index;
    nuka_kinematic_joint_t joint_type;
    float local_pose[7];
    float axis[3];
} nuka_kinematic_link_t;

// Root local poses are identity: this calibration query exposes no live state or world placement.
nuka_result_t nuka_world_get_kinematic_link(nuka_world_handle world,
                                            uint32_t link_index, nuka_kinematic_link_t* out);

// The body row (scene body record order) and articulation owning a per-environment link;
// a floating root's pose is written through NUKA_FIELD_BASE_POSE at its articulation slot.
typedef struct nuka_link_binding_t {
    uint32_t body_index;
    uint32_t articulation_index;
    uint32_t is_root;
    uint32_t is_floating;
} nuka_link_binding_t;

nuka_result_t nuka_world_get_link_binding(nuka_world_handle world, uint32_t link_index,
                                          nuka_link_binding_t* out);

// One cooked collision shape with its live world pose (xyz, wxyz) in an environment.
// params hold the sphere radius, capsule or cylinder radius and half height, or box half extents.
typedef struct nuka_collider_t {
    uint32_t kind;
    uint32_t is_static;
    float params[4];
    float pose[7];
} nuka_collider_t;

// A null out queries *out_count; otherwise capacity must cover it.
nuka_result_t nuka_world_get_colliders(nuka_world_handle world, uint32_t env_index,
                                       nuka_collider_t* out, uint32_t capacity,
                                       uint32_t* out_count);

// ---------------------------------------------------------------------------
// Offline beauty render of the LIVE world: a one-off HOST RGB image of the world's
// CURRENT state via the self-written offline CUDA path-tracer (NOT the gated
// realtime Vulkan present). The robot link visuals are FK-posed from the live link
// poses, any particle media is surfaced from the live particle field, and a studio
// floor is drawn -- lit/shaded by the SHARED studio render scene (identical to the
// go2_cloth_drape demo). Renders env 0. The render scene is built ONCE (lazily, from
// the world's retained scene + cooked particle-surface topology) and refreshed from
// the live state each call. This is how a Python script reproduces the cloth video.
// ---------------------------------------------------------------------------

// Pinhole camera for nuka_world_render_beauty: positioned at `eye`, looking at
// `look`, with `up` the world up (e.g. 0,0,1) and `fov_deg` the vertical field of
// view (degrees).
typedef struct nuka_beauty_camera_t {
    float eye[3];
    float look[3];
    float up[3];
    float fov_deg;
} nuka_beauty_camera_t;

// Render host pixels, top row first: dtype 0 = uint8 RGB, 1 = float32 RGB in [0,1],
// 2 = nine float32 channels: unquantized display RGB, linear albedo, shading normal.

// Color and denoiser features share the pixel samples; normals need not be unit length.
// `spp` selects the beauty sample count (0 uses the engine default).

// A null out_rgb queries the scalar count: width*height*(dtype == 2 ? 9 : 3).
// Otherwise out_capacity must cover that count; out_pixel_count may be null.

// NOT_SUPPORTED indicates missing offline RT support or renderable geometry.
// INVALID_ARG indicates insufficient capacity, a null camera, invalid dtype or zero dimensions.
nuka_result_t nuka_world_render_beauty(nuka_world_handle world,
                                       const nuka_beauty_camera_t* camera,
                                       uint32_t width, uint32_t height,
                                       uint32_t spp, uint8_t dtype,
                                       void* out_rgb, size_t out_capacity,
                                       size_t* out_pixel_count);

// ---------------------------------------------------------------------------
// Device-resident batched camera sensor: S cameras per env (each on its own mount)
// rendered into a single (env_count, sensors_per_env, height, width, channels)
// device tensor (no host download). The persistent high-quality RT sensor trace
// produces the observation; it is read zero-copy via
// nuka_world_get_sensor_view + torch.from_dlpack.
// With one camera per env the tensor collapses to (env_count, height, width, ch).
// ---------------------------------------------------------------------------

// LINK/BODY/BASE select a live pose row. WORLD fixes camera/lidar poses in world
// coordinates and ignores mount_index; mounted state sensors require a body or link.
typedef enum nuka_sensor_mount_t {
    NUKA_SENSOR_MOUNT_LINK = 0,
    NUKA_SENSOR_MOUNT_BODY = 1,
    NUKA_SENSOR_MOUNT_BASE = 2,
    NUKA_SENSOR_MOUNT_WORLD = 3
} nuka_sensor_mount_t;

// Authored camera placement and optics, resolved against one environment's live state.
// Local offsets use position followed by a wxyz quaternion; view is in world space.
typedef struct nuka_scene_camera_t {
    nuka_sensor_mount_t mount;
    uint32_t mount_index;
    float local_offset[7];
    nuka_beauty_camera_t view;
    float near_clip;
    float far_clip;
    float focus_distance;
    float shadow_radius;
} nuka_scene_camera_t;

// Names match exactly or by an unambiguous slash-delimited suffix.
nuka_result_t nuka_world_get_scene_camera(nuka_world_handle world, const char* name,
                                         uint32_t env_index, nuka_scene_camera_t* out_camera);

// Render environment zero using the named camera's live placement and authored optics.
nuka_result_t nuka_world_render_scene_camera(nuka_world_handle world, const char* name,
                                            uint32_t width, uint32_t height, uint32_t spp,
                                            uint8_t dtype, void* out_rgb, size_t out_capacity,
                                            size_t* out_pixel_count);

// AOV channel the sensor view returns: 0 color(3) 1 depth(1) 2 normal(3) 3
// albedo(3) 4 prim(1, uint32). RANGE(5) is the lidar plane (1 float per (az,el)
// cell of the (E,S,az,el) range tensor); reshape it with nuka_world_get_lidar_dims.
typedef enum nuka_sensor_channel_t {
    NUKA_SENSOR_CHANNEL_COLOR = 0,
    NUKA_SENSOR_CHANNEL_DEPTH = 1,
    NUKA_SENSOR_CHANNEL_NORMAL = 2,
    NUKA_SENSOR_CHANNEL_ALBEDO = 3,
    NUKA_SENSOR_CHANNEL_PRIM = 4,
    NUKA_SENSOR_CHANNEL_RANGE = 5
} nuka_sensor_channel_t;

// Camera AOV selection bits. Bit positions intentionally match
// nuka_sensor_channel_t for COLOR..PRIM; RANGE is a separate lidar tensor. Pass
// 0 to nuka_world_set_sensor_aov_mask to restore the legacy all-AOV profile.
typedef enum nuka_sensor_aov_t {
    NUKA_SENSOR_AOV_COLOR  = 1u << NUKA_SENSOR_CHANNEL_COLOR,
    NUKA_SENSOR_AOV_DEPTH  = 1u << NUKA_SENSOR_CHANNEL_DEPTH,
    NUKA_SENSOR_AOV_NORMAL = 1u << NUKA_SENSOR_CHANNEL_NORMAL,
    NUKA_SENSOR_AOV_ALBEDO = 1u << NUKA_SENSOR_CHANNEL_ALBEDO,
    NUKA_SENSOR_AOV_PRIM   = 1u << NUKA_SENSOR_CHANNEL_PRIM,
    NUKA_SENSOR_AOV_ALL = NUKA_SENSOR_AOV_COLOR | NUKA_SENSOR_AOV_DEPTH |
                          NUKA_SENSOR_AOV_NORMAL | NUKA_SENSOR_AOV_ALBEDO |
                          NUKA_SENSOR_AOV_PRIM
} nuka_sensor_aov_t;

// Build the world's batched sensor scene from its cooked visual instances (the
// per-env visual template, replicated across envs) and attach a camera mounted on
// `mount_frame`[mount_index] offset by `local_offset` (px,py,pz, qw,qx,qy,qz).
// `vfov_deg` is the vertical field of view; width/height size every env image.
// Re-callable: a call at the SAME width/height APPENDS another camera (S cameras
// per env, each its own mount); a call at a DIFFERENT width/height resets to a
// fresh single-camera set (one (E,S,H,W) tensor block needs one image size).
// Requires the world to have visual meshes (cooked from a scene with
// VisualMeshComponents); NOT_SUPPORTED if the engine has no RT sensor backend or
// the scene has no renderable geometry.
nuka_result_t nuka_world_attach_camera_sensor(nuka_world_handle world,
                                              nuka_sensor_mount_t mount_frame,
                                              uint32_t mount_index,
                                              const float local_offset[7],
                                              float vfov_deg,
                                              uint32_t width,
                                              uint32_t height);

// Render every env's camera into the persistent device AOV tensor IN PLACE on the
// world's backend stream (no host round-trip), driven by the world's LIVE link
// poses. Call after step() and after nuka_world_attach_camera_sensor.
nuka_result_t nuka_world_render_sensors(nuka_world_handle world);

// Select which camera AOV planes subsequent renders produce. mask is a bitwise
// OR of nuka_sensor_aov_t; mask==0 restores all planes. DEPTH|PRIM takes the
// primary-hit-only fast path (no hit reconstruction, materials, shadow ray, or
// shading). Omitted channels return NOT_SUPPORTED from get_sensor_view instead of
// exposing stale data. INVALID_ARG on unknown bits; NOT_SUPPORTED without a
// camera attachment.
nuka_result_t nuka_world_set_sensor_aov_mask(nuka_world_handle world,
                                              uint32_t mask);

// Zero-copy device view into the AOV tensor after nuka_world_render_sensors:
// `channel` selects the plane (see nuka_sensor_channel_t). out->device_ptr is the
// (env_count*sensors_per_env*height*width*ch) device base; element_count is that
// float/uint32 count; element_stride_bytes is sizeof(float) (sizeof(uint32) for
// PRIM); dtype is 0 float32 (1 uint32 for PRIM, same code nuka_buffer_view_t uses).
// MIRRORS nuka_world_get_buffer_view so a caller reshapes to (E,S,H,W,ch) (or
// (E,H,W,ch) when S==1). NOT_SUPPORTED before the first render.
nuka_result_t nuka_world_get_sensor_view(nuka_world_handle world,
                                         nuka_sensor_channel_t channel,
                                         nuka_buffer_view_t* out);

// Copies one (env, sensor) tile of `channel` to host memory after a render: H*W*ch values
// for camera planes, az*el for RANGE; all are 4-byte float (uint32 for PRIM). *out_count
// gets the tile's value count; a NULL `out` only reports it, and INVALID_ARG means `capacity`
// (in values) is short or env/sensor is out of range. NOT_SUPPORTED before the first render.
nuka_result_t nuka_world_download_sensor(nuka_world_handle world, nuka_sensor_channel_t channel,
                                         uint32_t env, uint32_t sensor, void* out,
                                         uint64_t capacity, uint64_t* out_count);

// Replaces the world's sensors with the scene's authored Camera/Depth and Lidar/RangeScan
// records, mounted as the cook mounts state sensors. Cameras must share one image size and
// lidars one fan (else INVALID_ARG). Render DR, fidelity, AOV mask and imaging models carry
// from the current attachment. A scene without such records, or without renderable geometry,
// detaches all sensors (counts 0). Once called, every scene rebuild re-runs it. Counts may be NULL.
nuka_result_t nuka_world_attach_scene_sensors(nuka_world_handle world, uint32_t* cameras,
                                              uint32_t* lidars);

// The scene node path of each render instance of the attached sensor scene, one per line
// in the order the PRIM plane's instance field indexes them (empty lines for particle
// surfaces). Sizes like nuka_world_get_asset_root. NOT_SUPPORTED without an attachment.
nuka_result_t nuka_world_get_sensor_instances(nuka_world_handle world, char* out, size_t cap,
                                              size_t* out_len);

// The attached sensor block's logical shape: env_count, sensors_per_env (S cameras
// per env), per-camera height/width, and `channels` (3 for the color/normal/albedo
// planes; depth/prim are 1, derived from a view's element_count). Any out-pointer
// may be NULL. Valid after attach (no render needed). NOT_SUPPORTED if no camera is
// attached. Lets a caller reshape the sensor view to (E,S,H,W,ch) without guessing.
nuka_result_t nuka_world_get_sensor_dims(nuka_world_handle world,
                                         uint32_t* env_count,
                                         uint32_t* sensors_per_env,
                                         uint32_t* height,
                                         uint32_t* width,
                                         uint32_t* channels);

// Device-resident batched lidar/range sensor: S lidars per env, each an (az,el)
// ray fan, rendered into a single (env_count, sensors_per_env, az_count, el_count)
// device RANGE tensor on the SAME RT TLAS the cameras use. A lidar ray and a camera
// ray are both (origin, unit dir) -> ClosestHit -> distance; the lidar writes that
// distance (miss -> max_range, clamped to [min_range, max_range]) instead of an
// image. The range is read zero-copy via nuka_world_get_sensor_view(RANGE).

// Attach a lidar mounted on `mount_frame`[mount_index] offset by `local_offset`
// (px,py,pz, qw,qx,qy,qz). The fan sweeps az_count*el_count rays: az in
// [az_min,az_max] about the local +Z axis, el in [el_min,el_max] toward local +Z
// (az=el=0 => local +X). Ranges clamp to [min_range,max_range]. Re-callable: a lidar
// attach REPLACES the world's lidar set (one (E,S,az,el) tensor needs one fan).
// NOT_SUPPORTED if the engine has no RT sensor backend or the scene has no geometry;
// INVALID_ARG on a zero fan / non-positive max_range / max_range<=min_range.
nuka_result_t nuka_world_attach_lidar_sensor(nuka_world_handle world,
                                             nuka_sensor_mount_t mount_frame,
                                             uint32_t mount_index,
                                             const float local_offset[7],
                                             uint32_t az_count,
                                             uint32_t el_count,
                                             float az_min,
                                             float az_max,
                                             float el_min,
                                             float el_max,
                                             float min_range,
                                             float max_range);

// The attached lidar set's logical shape: env_count, sensors_per_env (S lidars per
// env), and the per-lidar az_count/el_count fan. Any out-pointer may be NULL. Valid
// after a lidar attach (no render needed). NOT_SUPPORTED if no lidar is attached.
// The RANGE view reshapes to (env_count, sensors_per_env, az_count, el_count).
nuka_result_t nuka_world_get_lidar_dims(nuka_world_handle world,
                                        uint32_t* env_count,
                                        uint32_t* sensors_per_env,
                                        uint32_t* az_count,
                                        uint32_t* el_count);

// ---------------------------------------------------------------------------
// Per-env render domain randomization: give each env its OWN appearance (material
// base_color/roughness/metallic, light direction/intensity/color, ambient
// intensity) so the (E,S,H,W,ch) sensor tensor shows appearance variety -- the
// vision-policy / sim2real lever. The trace ALWAYS reads material/light BY ENV;
// DR off (enabled==0, the default) makes every env's table an exact base replica,
// so the cross-env byte-identical invariant is unchanged. Each enabled axis is a
// PURE function of (seed, env, axis) (Philox, no mutable RNG state), so the same
// seed yields the same bytes (RL-reproducible).
// ---------------------------------------------------------------------------

// Each *_jitter is a symmetric band: a randomized scalar is base + u*jitter (u in
// [-1,1]) for the additive axes (color/roughness/metallic/light_dir) or
// base*(1 + u*jitter) for the fractional axes (intensity/color/ambient). A zero
// jitter leaves that axis at its base value (an unset axis is a byte no-op).
typedef struct nuka_render_dr_desc_t {
    float color_jitter;             /* +/- per RGB channel of material base_color */
    float roughness_jitter;         /* +/- added to roughness (clamped) */
    float metallic_jitter;          /* +/- added to metallic (clamped) */
    float light_dir_jitter;         /* +/- per light direction component */
    float light_intensity_jitter;   /* fractional, base*(1 +/- this) */
    float light_color_jitter;       /* fractional per RGB of light color */
    float ambient_intensity_jitter; /* fractional, base*(1 +/- this) */
    uint64_t seed;                  /* Philox key; the recorded deterministic state */
    int enabled;                    /* 0 (default) -> tables are base replicas */
} nuka_render_dr_desc_t;

// Record the per-env render-DR config and (re)fill the per-env appearance tables
// from the base set as a pure function of (seed, env, axis). A NULL desc (or
// enabled==0) restores exact base replicas (cross-env byte-identical). Requires a
// camera attached (the sensor scene owns the tables). Idempotent for a fixed seed;
// safe to re-call per reset. NOT_SUPPORTED if no camera is attached.
nuka_result_t nuka_world_set_render_randomization(
    nuka_world_handle world, const nuka_render_dr_desc_t* desc);

// ---------------------------------------------------------------------------
// Sensor RGB shading: the batched sensor always uses the high-quality path
// (textured materials + MSAA + soft sun shadow + AO/GI + ACES + sRGB) in the
// SAME persistent trace. The profile controls quality/sample counts and the
// atmosphere; it is no longer an opt-in cheap-vs-beauty switch. Deterministic +
// seeded: stochastic samples derive from Philox(seed, env, sensor, pixel, sample).
// depth/normal/albedo/prim always come from the center ray.
// ---------------------------------------------------------------------------

// High-quality sensor profile. A NULL descriptor or an all-zero descriptor uses
// the defaults below. spp / shadow_samples / ao_samples are capped LOUDLY
// (INVALID_ARG over the cap).
typedef struct nuka_sensor_fidelity_desc_t {
    uint32_t spp;                /* jittered sub-pixel samples per pixel (MSAA); 0->1 */
    uint32_t shadow_samples;     /* soft-shadow rays across the sun disc; 0/1 -> hard */
    float sun_angular_radius;    /* sun half-angle (rad) -> penumbra width */
    int ao_enabled;              /* ambient occlusion / sky-dome ambient bounces */
    uint32_t ao_samples;         /* hemisphere rays when ao_enabled */
    float ao_radius;             /* AO ray max distance (world units) */
    int gi_enabled;              /* add the one-bounce diffuse GI term on AO rays */
    int tonemap_enabled;         /* ACES-ish filmic tonemap of averaged color */
    float sky_intensity;         /* scales the sky-dome ambient (0 -> uses default) */
    float fog_density;           /* per-metre height/distance fog extinction (0=off) */
    uint64_t seed;               /* Philox key; the recorded deterministic state */
    int srgb_enabled;            /* encode linear color to the RGB sensor contract */
} nuka_sensor_fidelity_desc_t;

// Record the high-quality sensor profile on the attached sensor. A NULL desc or
// an all-zero desc selects the default high-quality settings. A nonzero descriptor
// may tune samples and lighting, but the sensor always uses the beauty path.
// Requires a camera attached. Idempotent for a fixed seed; safe per reset.
// INVALID_ARG if a sample count is over its cap; NOT_SUPPORTED without a camera.
nuka_result_t nuka_world_set_sensor_fidelity(
    nuka_world_handle world, const nuka_sensor_fidelity_desc_t* desc);

// Exposure is in seconds; charge, readout and offset parameters use electrons.
// ADC bits 0 keeps continuous output; enabled 0 selects ideal RGB.
typedef struct nuka_camera_response_desc_t {
    uint32_t enabled;
    uint32_t shot_noise;
    uint32_t adc_bits;
    float exposure_time;
    float electrons_per_unit_second;
    float full_well_electrons;
    float read_noise_electrons;
    float row_noise_electrons;
    float dark_current;
    float dark_doubling_temperature;
    float temperature;
    float reference_temperature;
    float pixel_gain_stddev;
    float pixel_offset_stddev_electrons;
    float analog_gain;
    float black_level_electrons;
    float dead_pixel_probability;
    float hot_pixel_probability;
    float hot_pixel_current;
    uint64_t seed;
} nuka_camera_response_desc_t;

// Distances are meters; precision is m*sqrt(photon), quadratic_stddev is 1/m.
// return_photons specifies diffuse normal-incidence return at reference_distance.
typedef struct nuka_range_response_desc_t {
    uint32_t enabled;
    float bias;
    float scale_error;
    float distance_stddev;
    float quadratic_stddev;
    float incidence_bias;
    float quantization;
    float return_photons;
    float reference_distance;
    float background_photons;
    float precision;
    uint32_t minimum_return;
    float dropout_probability;
    uint64_t seed;
} nuka_range_response_desc_t;

typedef struct nuka_imaging_stamp_t {
    uint64_t acquisitions;
    double sample_time;
    uint32_t valid;
} nuka_imaging_stamp_t;

// Configure one camera or range channel and clear that sensor's acquisition history.
// A null descriptor restores ideal observations; range channel must be DEPTH or RANGE.
nuka_result_t nuka_world_set_camera_response(nuka_world_handle world, uint32_t sensor_index,
                                              const nuka_camera_response_desc_t* desc);
nuka_result_t nuka_world_set_range_response(nuka_world_handle world, nuka_sensor_channel_t channel,
                                             uint32_t sensor_index, const nuka_range_response_desc_t* desc);

// COLOR or DEPTH selects the shared camera acquisition; RANGE selects a lidar.
// Sample time is the environment's simulation time at explicit render_sensors acquisition.
nuka_result_t nuka_world_get_imaging_stamp(nuka_world_handle world, nuka_sensor_channel_t channel,
                                            uint32_t sensor_index, uint32_t env_index,
                                            nuka_imaging_stamp_t* out);


// ---------------------------------------------------------------------------
// Camera lens model: by default a camera is a centered pinhole (focal from the
// attach vfov+aspect). This applies the schema-carried lens knobs -- radial
// (Brown-Conrady) distortion + a clipped depth range -- to every attached camera.
// The DEFAULT desc (distortion==0, near<=0/far<=0 read as wide-open 0.01/1000) is
// a byte no-op. The principal point + per-axis focal (fx/fy/cx/cy) are renderer-
// side knobs that the scene schema does not carry, so they are not exposed here.
// ---------------------------------------------------------------------------

// The lens-model knobs (the scene CameraIntrinsics set). distortion!=0 enables the
// radial term r2=x*x+y*y, d=1+k1*r2+k2*r2*r2 on each ray's normalized sensor coord
// (k1==k2==0 with distortion on is just identity). near_clip/far_clip clip the
// depth AOV: a hit outside [near,far] reads as a miss. A non-positive near/far
// reads as the wide-open default (0.01 / 1000), so an all-zero desc changes nothing.
typedef struct nuka_camera_intrinsics_desc_t {
    int distortion;   /* 0 (default) -> radial term off (byte no-op) */
    float k1;         /* first radial coefficient */
    float k2;         /* second radial coefficient */
    float near_clip;  /* depth-AOV near clip (<=0 -> default 0.01) */
    float far_clip;   /* depth-AOV far clip (<=0 -> default 1000) */
} nuka_camera_intrinsics_desc_t;

// Record the lens-model knobs on every attached camera and rebuild the sensor
// scene so the next render uses them (the per-env DR + shading fidelity are
// preserved across the rebuild). A NULL desc restores the default pinhole + the
// wide-open clip (a byte no-op for today's scenes). Requires a camera attached.
nuka_result_t nuka_world_set_camera_intrinsics(
    nuka_world_handle world, const nuka_camera_intrinsics_desc_t* desc);

typedef struct nuka_invariant_violation_t {
    uint32_t invariant;
    uint32_t step;
    uint32_t env_id;
    float value;
    float threshold;
} nuka_invariant_violation_t;

nuka_result_t nuka_world_get_last_invariant_violations(nuka_world_handle world,
                                                       uint32_t* out_count,
                                                       void* out_array,
                                                       uint32_t array_capacity);

// ---------------------------------------------------------------------------
// BATCHED arbitrary-(x,y) terrain height sampler over the world's COOKED grid.
// The obs/spawn/render height query reads the SAME full-relief HeightField the
// physics rests feet on (stored on the world at create), via the ONE bilinear
// sampler -- so obs and physics agree by construction (no per-env scale, no python
// mirror to drift; the grid is model-level so any per-env shrink would diverge).
//
//   world          -- the world whose cooked terrain grid is sampled.
//   n              -- element count (number of (x,y) tuples).
//   xs, ys         -- per-element world column (x,y) in meters.
//   out_heights    -- output array (n floats), the absolute surface z per column.
//
// Returns NUKA_RESULT_INVALID_ARG if n > 0 and any of xs/ys/out_heights is NULL,
// or the world handle is null. When no heightfield was cooked, out is filled with
// the model ground floor (0.0). n == 0 is a no-op OK.
nuka_result_t nuka_world_sample_terrain_height_batch(nuka_world_handle world,
                                                     uint32_t n,
                                                     const float* xs,
                                                     const float* ys,
                                                     float* out_heights);

// The cooked terrain grid: world z of value v at (row, col) is base_z + v * scale_z,
// with column c at origin.x + c * cell_x and row r at origin.y + r * cell_y.
typedef struct nuka_terrain_info_t {
    uint32_t nrow;
    uint32_t ncol;
    float cell_x;
    float cell_y;
    float origin[3];
    float base_z;
    float scale_z;
} nuka_terrain_info_t;

// NOT_SUPPORTED without a cooked terrain. A null values reads only the info; otherwise
// capacity must hold nrow * ncol row-major floats.
nuka_result_t nuka_world_get_terrain(nuka_world_handle world, nuka_terrain_info_t* out_info,
                                     float* values, size_t capacity);

#ifdef __cplusplus
}
#endif

#endif /* NUKA_NUKA_H */
