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
// (union_world.cpp / the .nks "root.quat"). pos[3] is (x, y, z) metres.
//
// CUDA GATING. The whole TU is interface-layer C++ with ZERO CUDA tokens; only
// nuka_scene_settle actually touches a device (it cooks the scene to an nk::World
// on the device's OWNED phi v2 backend and runs cook::Settle). On a CUDA-less
// build (no phi v2 backend acquired in nuka_device_create) settle returns
// NUKA_RESULT_NOT_SUPPORTED; every other entry works host-only.
//
// WHY A C ABI (not a C++ class bind): same reason as nuka_union.h / nuka_recorder
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

#ifdef __cplusplus
}
#endif

#endif /* NUKA_NUKA_SCENE_H */
