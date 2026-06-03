#pragma once
// ---------------------------------------------------------------------------
// nuka::scene::Compose – format-agnostic SceneIR merge / compose layer
// ---------------------------------------------------------------------------
//
// Both importers (LoadMjcf, LoadUsd) produce the same nuka::scene::SceneIR.
// "USD <-> MJCF coexistence" (owner v0.7 must-do) reduces to a pure SceneIR
// merge: load A and B (any mix of MJCF / USD), then drop B into A at a
// placement transform. The eventual demo is a USD cup composed onto an MJCF
// kitchen counter, and an MJCF robot composed into a USD scene.
//
// Compose is pure and deterministic (D1): it copies `base`, then appends every
// `addon` record in its existing order with all ids offset by base's
// corresponding counts and all intra-addon cross-references remapped by those
// offsets. kInvalid* references are preserved (sentinels are never offset).
// Each addon ROOT body (parent_id == kInvalidBody) is re-rooted into base's
// frame by composing `placement` onto its local transform.
// ---------------------------------------------------------------------------

#include "math/transform.hpp"
#include "scene/scene_ir.hpp"

#include <string>

namespace nuka::scene {

/// Merge `addon` into a copy of `base`, positioning the whole addon at
/// `placement` (in base's frame). All appended record ids are offset by base's
/// corresponding counts; every intra-addon cross-reference (body / material /
/// joint) is remapped accordingly; kInvalid* references stay kInvalid*.
///
/// Placement: each addon root body (parent_id == kInvalidBody) gets
///   local_transform = placement * original_local_transform.
/// Non-root addon bodies keep their local transforms (their world pose follows
/// the re-rooted parent).
///
/// Name prefix: if `addon_name_prefix` is non-empty it is prepended to the
/// names of all appended named records, so a composed scene has no duplicate
/// names. Default empty => names unchanged.
///
/// Pure: `base` and `addon` are not modified; the merged scene is returned.
SceneIR Compose(const SceneIR& base, const SceneIR& addon,
                const math::Transform& placement,
                const std::string& addon_name_prefix = "");

}  // namespace nuka::scene
