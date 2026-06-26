#ifndef NUKA_NUKA_SCENE_H
#define NUKA_NUKA_SCENE_H
// ---------------------------------------------------------------------------
// Nuka C ABI -- the GENERIC SCENE-AUTHORING surface (M9 T4). A MINIMAL extern
// "C" shim over nuka::scene::SceneIR (the M2b facade) that loads / composes /
// edits / settles / saves a scene WITHOUT ever cooking a steppable world. The
// later world entry (World.create over Scene->CookToModel->nk::World, M9 T5)
// consumes the SAME in-memory SceneRecord this surface authors.
//
// ★ HIGHEST DIRECTIVE -- GENERIC, NOT PER-DEMO. There is exactly ONE entry that
// loads any supported format (mjcf/urdf/usd/nks) and a small uniform editing /
// settle / save surface. NOTHING here special-cases grasp / union / any demo.
// Behavior differences come from the imported SCENE DATA + a per-scene control
// SCRIPT, never from a special API. (owner [[unified-world-no-special-grasp-
// binding]].)
//
// FORMAT DISPATCH (one entry). nuka_scene_load dispatches by file extension:
//   .nks                -> scene::nks::Load  (resolves `imports` internally)
//   .xml | .mjcf        -> import::LoadMjcf
//   .urdf               -> import::LoadUrdf
//   .usd | .usda        -> import::LoadUsd
// reusing the SAME LoadSceneByExtension dispatch world.cpp's create path uses.
//
// NODE ADDRESSING -- by STRING PATH, no node handle. Every editing entry takes
// the node's DERIVED tree path ("h1/right_hand_link", "cup/body"). We chose a
// path-string model over an opaque node handle DELIBERATELY: the SceneIR facade
// re-projects tree_ + ecs_ (creating FRESH entities + SceneNode pointers) after
// any record mutation (a Get*Mut marks the facade dirty -- see scene_ir.hpp),
// so any node/entity handle handed across the C boundary would be invalidated by
// the very next set_local / set_physics_material call. A path string is stable
// across mutations and carries zero lifetime hazard. nuka_scene_find reports
// existence so a caller can validate a path before editing.
//
// QUATERNION LAYOUT -- W FIRST. Every quat[4] is (w, x, y, z), matching
// nuka::math::Quat{w,x,y,z} (math/quat.hpp) and the repo-wide pose7 convention
// (the .nks "root.quat"). pos[3] is (x, y, z) metres.
//
// CUDA GATING. The whole TU is interface-layer C++ with ZERO CUDA tokens; only
// nuka_scene_settle actually touches a device (it cooks the scene to an nk::World
// on the device's OWNED phi v2 backend and runs cook::Settle). On a CUDA-less
// build (no phi v2 backend acquired in nuka_device_create) settle returns
// NUKA_RESULT_NOT_SUPPORTED; every other entry works host-only.
//
// WHY A C ABI (not a C++ class bind): same reason as nuka.h / nuka_recorder
// .h -- the engine is g++-14, the nanobind binding g++-10; only plain C crosses.
//
// ERROR / HANDLE CONVENTIONS mirror nuka.h: NULL out / NULL handle ->
// NUKA_RESULT_NULL_HANDLE, bad arg / unknown extension -> INVALID_ARG, missing
// file -> FILE_NOT_FOUND, no phi v2 device -> NOT_SUPPORTED; out-params are
// zeroed on failure; every body is try/catch -> MapExceptionToResult.
// ---------------------------------------------------------------------------

#include "nuka/nuka.h"  // nuka_result_t, nuka_device_handle, stdint/stddef

#ifdef __cplusplus
extern "C" {
#endif

typedef struct nuka_scene_t* nuka_scene_handle;

// ---------------------------------------------------------------------------
// load / destroy
// ---------------------------------------------------------------------------

// Load a scene from `path`, dispatching by file extension (.nks / .xml / .mjcf
// / .urdf / .usd / .usda). The loaded SceneIR is owned by the returned handle.
// Returns INVALID_ARG on a NULL path or an unrecognized extension,
// FILE_NOT_FOUND if the file is missing, PARSE_ERROR on malformed input.
// Zeroes *out on any failure.
nuka_result_t nuka_scene_load(const char* path, nuka_scene_handle* out);

// Destroy a scene handle (no-op on NULL / unknown handle).
void nuka_scene_destroy(nuka_scene_handle scene);

// ---------------------------------------------------------------------------
// compose
// ---------------------------------------------------------------------------

// Graft `addon` into `base` at the placement transform (pos[3] xyz + quat[4]
// w,x,y,z, in base's frame), via scene::Compose. The merged scene REPLACES
// `base`'s contents in place (base is mutated; addon is unchanged). `attach_at`
// is an OPTIONAL node-path PREFIX for the addon's appended names (NULL / "" ->
// no prefix); it disambiguates duplicate names in the merged scene (the addon
// keeps its OWN root hierarchy -- Compose re-roots each addon root body into
// base's frame at the placement, it does not re-parent under a base node).
// pos / quat NULL -> identity placement. Returns NULL_HANDLE on a bad handle.
nuka_result_t nuka_scene_compose(nuka_scene_handle base, nuka_scene_handle addon,
                                 const float pos[3], const float quat[4],
                                 const char* attach_at);

// ---------------------------------------------------------------------------
// query / edit (address nodes by derived tree path)
// ---------------------------------------------------------------------------

// Report whether a node at the derived tree path `path` exists (SceneGraph::
// NodeOf, which tolerates a leading "Scene" segment). Sets *out_found to 1 / 0.
// Returns NULL_HANDLE on a bad handle, INVALID_ARG on NULL path / out_found.
nuka_result_t nuka_scene_find(nuka_scene_handle scene, const char* path,
                              int* out_found);

// Set the LOCAL transform of the node at `path` (its body or shape record's
// local_transform). pos[3] xyz + quat[4] w,x,y,z; either NULL -> leave that
// component unchanged. Returns NULL_HANDLE on a bad handle, INVALID_ARG on a
// NULL path, FILE_NOT_FOUND if no node matches `path` (reusing the "not found"
// code -- there is no NODE_NOT_FOUND in nuka_result_t).
nuka_result_t nuka_scene_set_local(nuka_scene_handle scene, const char* path,
                                   const float pos[3], const float quat[4]);

// Set physics-material params on every COLLISION-SHAPE node whose derived path
// matches the std::regex `path_regex` (full-match). The record carries a SINGLE
// isotropic friction coefficient (friction_mu), so static_friction and
// dynamic_friction MUST be supplied EQUAL when both are >= 0 (a mismatch ->
// INVALID_ARG); a value < 0 leaves that param untouched. restitution is applied
// to the matched shapes' RESOLVED PhysicsMaterial in the live facade Registry
// for in-memory cook consumers, but is NOT persisted by .nks Save (the .nks /
// record schema has no restitution field -- see the header NOTE / scene.cpp).
// *out_matched (optional) receives the number of shapes touched. Returns
// NULL_HANDLE on a bad handle, INVALID_ARG on a NULL / malformed regex or a
// static!=dynamic friction request.
nuka_result_t nuka_scene_set_physics_material(nuka_scene_handle scene,
                                              const char* path_regex,
                                              float static_friction,
                                              float dynamic_friction,
                                              float restitution,
                                              uint32_t* out_matched);

// ---------------------------------------------------------------------------
// settle (device; CUDA-gated)
// ---------------------------------------------------------------------------

// Cook the scene to an nk::World on `device`'s OWNED phi v2 backend, run
// cook::Settle for `steps` deterministic device steps (dt <= 0 -> 1/240), and
// WRITE THE SETTLED STATE BACK into the scene so a subsequent save persists it:
//   * the settled articulation IC (per-link q + base pose) -> the scene's
//     SceneInitialState, keyed by the cooked articulation root's derived path;
//   * each settled movable FREE-body's world pose -> its body record
//     local_transform (ApplySettleToSceneIR).
// The hold pattern is the scene's authored Settle() spec when present, else a
// generic "hold every articulation link at its seeded q" default. Returns
// NUKA_RESULT_NOT_SUPPORTED when `device` has no phi v2 backend (CUDA-less
// build) or when the cooked world is not Ready / a device op fails; NULL_HANDLE
// on a bad scene/device handle.
nuka_result_t nuka_scene_settle(nuka_scene_handle scene,
                                nuka_device_handle device, uint32_t steps,
                                float dt);

// ---------------------------------------------------------------------------
// save
// ---------------------------------------------------------------------------

// Save the scene to `nks_path` (+ a sibling `<base>.nka`), via scene::nks::Save
// (the .nks own format). Returns NULL_HANDLE on a bad handle, INVALID_ARG on a NULL path,
// INTERNAL / PARSE_ERROR on a write failure.
nuka_result_t nuka_scene_save(nuka_scene_handle scene, const char* nks_path);

// ---------------------------------------------------------------------------
// Built-scene authoring: program a SceneIR (rigid primitives + media records)
// and cook it through the SAME CookSceneToModel a file scene uses. This is the
// GENERAL media authoring substrate -- a tet-soft body / a fluid box / a second
// cloth go through nuka_scene_add_media (a TAGGED record), never by accreting
// flat slots onto nuka_coupled_particles_desc_t. The handle owns one in-memory
// SceneIR (the SAME SceneRecord nuka_scene_load returns); nuka_scene_destroy
// frees it. ADDITIVE: a built scene with zero primitives + zero media that
// imports a file cooks byte-identically to nuka_world_create_from_scene.
// ---------------------------------------------------------------------------

// A rigid collision primitive (one body + one shape). dims by kind: BOX uses
// dims[0..2] as the half-extents (x,y,z); SPHERE uses dims[0] as the radius;
// CAPSULE uses dims[0] radius + dims[1] half-height (local Z axis); PLANE uses
// none (an infinite ground plane is always static). pos[3] is (x,y,z) metres,
// quat[4] is (w,x,y,z) -- an all-zero quat reads as identity.
typedef enum nuka_primitive_kind_t {
    NUKA_PRIMITIVE_PLANE   = 0,
    NUKA_PRIMITIVE_BOX     = 1,
    NUKA_PRIMITIVE_SPHERE  = 2,
    NUKA_PRIMITIVE_CAPSULE = 3
} nuka_primitive_kind_t;

typedef struct nuka_rigid_primitive_desc_t {
    uint32_t kind;        // nuka_primitive_kind_t
    float    dims[3];     // half-extents / radius / (radius,half_height) by kind.
    float    pos[3];      // world position (x,y,z).
    float    quat[4];     // world orientation (w,x,y,z); all-zero => identity.
    uint32_t is_static;   // 1 => static (inv_mass 0); 0 => movable (PLANE forces 1).
    float    mass;        // body mass kg when movable (<= 0 => 1.0). Ignored if static.
    float    friction;    // per-shape Coulomb mu (< 0 => inherit the material default).
} nuka_rigid_primitive_desc_t;

// What a medium IS (solver routing) and HOW it solves (one method per medium).
// The legal (kind x method) set is enforced by the SAME cook validator: cloth =
// XPBD; soft-tet = XPBD or MLS-MPM; fluid = PBF or MLS-MPM. An illegal pair is
// rejected LOUDLY (INVALID_ARG) at nuka_scene_add_media; an MPM + XPBD/PBF mix is
// rejected at cook (nuka_world_create_from_built_scene).
typedef enum nuka_media_kind_t {
    NUKA_MEDIA_CLOTH    = 0,
    NUKA_MEDIA_SOFT_TET = 1,
    NUKA_MEDIA_FLUID    = 2
} nuka_media_kind_t;

typedef enum nuka_media_method_t {
    NUKA_MEDIA_METHOD_XPBD   = 0,
    NUKA_MEDIA_METHOD_PBF    = 1,
    NUKA_MEDIA_METHOD_MLSMPM = 2
} nuka_media_method_t;

// A TAGGED media record: `kind` + `method` select which geometry block and which
// constitutive material block the cook reads (the others are ignored). Each field
// maps 1:1 to a scene::MediaRecord field -- no magic layout, no new physics. A
// zero-initialized desc with only the relevant block filled is the intended use.
typedef struct nuka_media_desc_t {
    uint32_t kind;    // nuka_media_kind_t
    uint32_t method;  // nuka_media_method_t

    // -- geometry (the block matching `kind`) -------------------------------
    // CLOTH: a flat (cloth_nx x cloth_ny) lattice (>= 2 each) at cloth_origin
    // with cloth_spacing edge; perimeter pinned unless cloth_free.
    uint32_t cloth_nx, cloth_ny;
    float    cloth_spacing;
    float    cloth_origin[3];
    uint32_t cloth_free;         // 1 => free drape; 0 => perimeter pinned.
    // SOFT_TET: a sphere tet-lattice at tet_center, tet_radius, tet_cells cells
    // per axis, tet_cell_len cell edge.
    float    tet_center[3];
    float    tet_radius;
    uint32_t tet_cells;
    float    tet_cell_len;
    // FLUID: an AABB box [fluid_min, fluid_max] sampled at fluid_spacing.
    float    fluid_min[3];
    float    fluid_max[3];
    float    fluid_spacing;

    // -- constitutive material (the block matching `method`) -----------------
    // XPBD (cloth uses distance+bend; soft-tet uses distance+volume).
    float    xpbd_particle_mass;
    float    xpbd_friction;
    float    xpbd_distance_alpha;
    float    xpbd_bend_alpha;
    float    xpbd_volume_alpha;
    uint32_t xpbd_iters;
    float    xpbd_aero_drag_normal;
    float    xpbd_aero_drag_tangent;
    float    xpbd_aero_drag_max_dv;
    // PBF (fluid). support_radius = pbf_support_scale * fluid_spacing. The walls
    // fields generate a pinned boundary slab + side ring (box confinement).
    float    pbf_rest_density;
    float    pbf_support_scale;
    uint32_t pbf_iters;
    float    pbf_friction;
    uint32_t pbf_clamp_overdensity;  // 1 => clamp over-density (the default).
    uint32_t pbf_walls_enabled;      // 1 => append the pinned confinement container.
    float    pbf_walls_min[3];
    float    pbf_walls_max[3];
    float    pbf_floor_z;
    uint32_t pbf_boundary_layers;
    // MLS-MPM (soft-tet / fluid). model_kind 0 corotated, 2 neo-hookean, 3 fluid.
    float    mpm_youngs;
    float    mpm_poisson;
    float    mpm_density;
    float    mpm_dp_friction;
    float    mpm_dp_cohesion;
    float    mpm_model_kind;
    float    mpm_bulk_modulus;
    float    mpm_tait_gamma;
    float    mpm_viscosity;
    float    mpm_dx;
    uint32_t mpm_substeps;
    float    mpm_floor_normal[3];
    float    mpm_floor_d;
    float    mpm_floor_friction;

    // -- render skin (surface bake metadata; consumed downstream, not the cook) -
    float    skin_normal_offset;
    uint32_t skin_smooth_iters;
    float    skin_smooth_lambda;
    uint32_t render_material_id;     // ~0u (0xFFFFFFFF) => none.
} nuka_media_desc_t;

// Create a built-scene handle owning one in-memory SceneIR. A non-empty
// `base_scene_path` imports that scene (e.g. a robot .nks/.xml/.urdf/.usd) as the
// base so primitives/media are added ON TOP; NULL or "" starts EMPTY (a robot-free
// soft/fluid scene). Returns NULL on a load failure / out of memory (use
// nuka_scene_load directly when the failure reason is needed). Freed via
// nuka_scene_destroy (the SAME destroy the load handle uses).
nuka_scene_handle nuka_scene_create(const char* base_scene_path);

// Add a rigid collision primitive (one body + one shape) to the built scene via
// SceneIR::AddRigidBody + AddCollisionShape (the records the file cook also reads).
// Returns NULL_HANDLE on a bad handle, INVALID_ARG on a NULL / unknown-kind desc.
nuka_result_t nuka_scene_add_rigid_primitive(nuka_scene_handle scene,
                                             const nuka_rigid_primitive_desc_t* desc);

// Add a media record (cloth / soft-tet / fluid) to the built scene via
// SceneIR::AddMedia. The single record's (kind x method) legality is checked
// immediately by the cook's ValidateMedia -> INVALID_ARG on an illegal pair (the
// cross-medium MPM+XPBD/PBF mix is checked at cook). Returns NULL_HANDLE on a bad
// handle, INVALID_ARG on a NULL / unknown-kind / unknown-method / illegal-pair desc.
nuka_result_t nuka_scene_add_media(nuka_scene_handle scene,
                                   const nuka_media_desc_t* desc);

// Cook the built scene's SceneIR via the SAME cook::CookSceneToModel a file scene
// uses (rigid + media, ONE orchestrator) and build the world through the SAME
// record-assembly path nuka_world_create_from_scene uses. `desc->scene_path` is
// IGNORED (the scene comes from the handle); the other desc fields (env_count, dt,
// control_mode, gravity, optional baked heightfield) apply as usual. An illegal
// media mix throws at cook -> a non-OK result. Returns NULL_HANDLE on a bad scene /
// device handle, INVALID_ARG on a bad desc, NOT_SUPPORTED with no CUDA backend.
nuka_result_t nuka_world_create_from_built_scene(nuka_device_handle device,
                                                 const nuka_world_desc_t* desc,
                                                 nuka_scene_handle scene,
                                                 nuka_world_handle* out);

#ifdef __cplusplus
}
#endif

#endif /* NUKA_NUKA_SCENE_H */
