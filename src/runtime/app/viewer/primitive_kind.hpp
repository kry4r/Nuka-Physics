#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::app::viewer -- the primitives the editor spawns as free bodies.
//
// Shared by the spawn mutator (editor_edits) and the picker UI state (imgui_layer)
// so neither has to pull in the other's heavy includes. Plain enum, zero deps.
// ---------------------------------------------------------------------------

#include <cstdint>

namespace nuka::runtime::app::viewer {

// A spawnable primitive body kind. Box/Sphere/Capsule spawn as DYNAMIC movable
// bodies; Plane spawns as a static ground-like (massless) body.
enum class PrimitiveKind : uint8_t { Box, Sphere, Capsule, Plane };

}  // namespace nuka::runtime::app::viewer
