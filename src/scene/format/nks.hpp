#pragma once
// ---------------------------------------------------------------------------
// nuka::scene - .nks own scene file format (M2c).
//
// A .nks is the OWN, self-described scene container: a JSON sidecar (<name>.nks)
// carrying the structural scene (materials + the pre-order node tree with full
// LEGACY-FIDELITY record fields per node) plus a sibling binary container
// (<name>.nka) holding the heavy mesh/hull/SDF/texture bytes the JSON references
// via AssetRef strings ("foo.nka#MESH/0").
//
// Save serializes the SceneGraph in PRE-ORDER as a nested `tree` (no imports
// re-emitted), reading each node's field VALUES from the record it maps to so
// cook fidelity is preserved; Load replays the tree in record-id-faithful phases
// (materials, then bodies, then shapes, then joints, then cameras/lights/
// actuators, then the record-only sensors / filter sections), reading the .nka
// mesh chunks back into the record mesh geometry. Same SceneIR => byte-identical
// .nks + .nka (the M2c roundtrip gate asserts this). See nks.cpp for the
// record_name<->derived-path inverse and the id-order argument.
//
// Determinism guarantees (see json.hpp + nka.hpp):
//   - tree is pre-order; sibling order == tree tail-append order; insertion-
//     keyed JSON objects.
//   - floats render via "%.9g" (binary32-lossless).
//   - .nka chunk order == first-touch order during Save; deduped by content hash.
//
// HOST-ONLY: no CUDA, no third-party deps.
// ---------------------------------------------------------------------------

#include "scene/scene_ir.hpp"

#include <string>

namespace nuka::scene::nks {

// Save `scene` to `<nks_path>` (+ a sibling `<base>.nka` next to it). The .nka
// path stored inside the JSON is the sibling file's basename. Throws on I/O or
// structural errors.
void Save(const SceneIR& scene, const std::string& nks_path);

// Load a scene from `<nks_path>` (reads the sibling .nka named inside). The
// reconstructed SceneIR's records (and thus its facade tree/ECS) are identical
// to the one that was Saved. `imports` in the JSON are resolved here (LoadMjcf/
// LoadUsd/LoadUrdf by extension, then Compose with the attach_at prefix). Throws
// json::ParseError / std::runtime_error on malformed input.
SceneIR Load(const std::string& nks_path);

// Override layer: load `base_path`, then apply `overlay_path` (a JSON
// {"overrides": {"<derived/path>": {<partial component objects>}}}). The overlay
// patches the matched nodes' records (transform pos, rigid_body mass, material
// fields, ...) and leaves everything else untouched.
SceneIR Load(const std::string& base_path, const std::string& overlay_path);

} // namespace nuka::scene::nks
