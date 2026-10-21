#pragma once

#include "math/vec3.hpp"
#include "rt/two_level_render.hpp"

#include <array>
#include <memory>
#include <vector>

namespace nuka::render {

struct RenderWorld;
class RtBackendI;

struct IrradianceProbe {
    // Cosine-convolved radiance divided by pi, in real SH order through degree two.
    std::array<math::Vec3, 9> diffuse{};
    std::vector<float> distance;
};

struct IrradianceVolume {
    math::Vec3 origin{}, spacing{1.0f, 1.0f, 1.0f};
    std::array<uint32_t, 3> dimensions{3u, 3u, 3u};
    uint32_t face_size = 8u;
    float max_distance = 1000.0f;
    std::vector<IrradianceProbe> probes;

    void Validate() const;
};

struct IrradianceBakeOptions {
    math::Vec3 origin{}, spacing{1.0f, 1.0f, 1.0f};
    std::array<uint32_t, 3> dimensions{3u, 3u, 3u};
    uint32_t face_size = 8u;
    float max_distance = 1000.0f;
    rt::BeautyOptions lighting;
    rt::Light fallback_light;
};

// Bake static geometry and lights through the existing ray backend; moving objects receive the field.
std::shared_ptr<const IrradianceVolume> BakeIrradianceVolume(
    RtBackendI& backend, const RenderWorld& world, const IrradianceBakeOptions& options);

}  // namespace nuka::render
