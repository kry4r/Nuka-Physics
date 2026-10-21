#include "render/irradiance_volume.hpp"

#include "render/render_world.hpp"
#include "render/rt_adapter.hpp"
#include "render/rt_backend.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace nuka::render {
namespace {
constexpr float kPi = 3.14159265358979323846f;
const std::array<math::Vec3, 6> kDirections{{{1, 0, 0}, {-1, 0, 0}, {0, 1, 0},
                                           {0, -1, 0}, {0, 0, 1}, {0, 0, -1}}};

uint64_t ValidateBounds(math::Vec3 origin, math::Vec3 spacing,
                        const std::array<uint32_t, 3>& dimensions, uint32_t face_size, float distance) {
    uint64_t count = 1u;
    const float origins[] = {origin.x, origin.y, origin.z};
    const float spacings[] = {spacing.x, spacing.y, spacing.z};
    for (uint32_t axis = 0u; axis < 3u; ++axis) {
        if (!std::isfinite(origins[axis]) || !std::isfinite(spacings[axis]) || spacings[axis] <= 0.0f ||
            dimensions[axis] < 2u || dimensions[axis] > 64u ||
            !std::isfinite(origins[axis] + spacings[axis] * float(dimensions[axis])))
            throw std::invalid_argument("invalid lighting volume bounds");
        count *= dimensions[axis];
    }
    if (count > 4096u || face_size < 4u || face_size > 32u ||
        !std::isfinite(distance * distance) || distance <= 0.0f ||
        count * face_size * face_size * 6u > 16u * 1024u * 1024u)
        throw std::invalid_argument("lighting volume exceeds the bake budget");
    return count;
}

std::array<float, 9> Basis(const math::Vec3& n) {
    return {0.2820948f, 0.4886025f * n.y, 0.4886025f * n.z, 0.4886025f * n.x,
        1.0925484f * n.x * n.y, 1.0925484f * n.y * n.z, 0.3153916f * (3.0f * n.z * n.z - 1.0f),
        1.0925484f * n.x * n.z, 0.5462742f * (n.x * n.x - n.y * n.y)};
}

struct SceneOwner {
    RtBackendI& backend;
    RtSceneHandle* handle;
    ~SceneOwner() { if (handle) backend.FreeScene(handle); }
};
}  // namespace

void IrradianceVolume::Validate() const {
    const uint64_t count = ValidateBounds(origin, spacing, dimensions, face_size, max_distance);
    if (probes.size() != count) throw std::invalid_argument("lighting volume probe count mismatch");
    for (const auto& probe : probes) {
        if (probe.distance.size() != size_t{6u} * face_size * face_size)
            throw std::invalid_argument("lighting volume distance count mismatch");
        for (const auto& color : probe.diffuse)
            if (!std::isfinite(color.x) || !std::isfinite(color.y) || !std::isfinite(color.z))
                throw std::invalid_argument("nonfinite lighting volume coefficient");
        for (float distance : probe.distance)
            if (!std::isfinite(distance) || distance < 0.0f || distance > max_distance)
                throw std::invalid_argument("invalid lighting volume distance");
    }
}

std::shared_ptr<const IrradianceVolume> BakeIrradianceVolume(
    RtBackendI& backend, const RenderWorld& world, const IrradianceBakeOptions& options) {
    const uint64_t count = ValidateBounds(options.origin, options.spacing, options.dimensions,
                                          options.face_size, options.max_distance);

    auto scene = RenderWorldToTwoLevelScene(world);
    std::vector<rt::Instance> instances;
    for (size_t i = 0u; i < world.instances.size(); ++i)
        if (world.instances[i].pose_source.kind == PoseSource::Kind::Static && !world.instances[i].deforming)
            instances.push_back(scene.instances[i]);
    scene.instances = std::move(instances);
    if (scene.instances.empty()) throw std::invalid_argument("local lighting needs static scene geometry");

    std::vector<rt::Light> lights;
    for (const auto& source : world.lights) {
        if (source.pose_source.kind != PoseSource::Kind::Static) continue;
        rt::Light light;
        light.directional = source.type == scene::LightComponent::Type::Directional;
        light.direction = source.world_xform.TransformDirection({0, 0, -1}).Normalized();
        light.position = source.world_xform.position;
        light.color = source.color;
        light.intensity = source.intensity;
        lights.push_back(light);
    }
    if (world.lights.empty()) lights.push_back(options.fallback_light);
    if (lights.empty()) { rt::Light dark; dark.intensity = 0.0f; lights.push_back(dark); }

    auto volume = std::make_shared<IrradianceVolume>();
    volume->origin = options.origin;
    volume->spacing = options.spacing;
    volume->dimensions = options.dimensions;
    volume->face_size = options.face_size;
    volume->max_distance = options.max_distance;
    volume->probes.resize(static_cast<size_t>(count));
    const uint32_t pixels = options.face_size * options.face_size;
    for (auto& probe : volume->probes) probe.distance.resize(size_t{6u} * pixels);

    rt::BeautyOptions lighting = options.lighting;
    lighting.fog_density = 0.0f;
    lighting.ao_radius = options.max_distance;
    lighting.download.albedo = lighting.download.normal = lighting.download.uv = lighting.download.prim = false;
    SceneOwner owner{backend, backend.BuildScene(scene)};
    if (!owner.handle) throw std::runtime_error("cannot build local lighting scene");
    for (size_t pass = 0u; pass < lights.size(); ++pass) {
        scene.light = lights[pass];
        if (pass == 1u) {
            scene.environment.intensity = 0.0f;
            lighting.sky_intensity = 0.0f;
            lighting.sky_top = lighting.sky_bottom = lighting.sky_ground = math::Vec3::Zero();
            lighting.sun_disc_radiance = math::Vec3::Zero();
            for (auto& material : scene.materials) {
                material.emission = math::Vec3::Zero();
                material.emissive_tex = -1;
            }
            backend.UpdateScene(owner.handle, scene);
        }
        for (uint32_t p = 0u; p < count; ++p) {
            const uint32_t x = p % options.dimensions[0];
            const uint32_t y = (p / options.dimensions[0]) % options.dimensions[1];
            const uint32_t z = p / (options.dimensions[0] * options.dimensions[1]);
            const math::Vec3 position = options.origin + math::Vec3{
                float(x) * options.spacing.x, float(y) * options.spacing.y, float(z) * options.spacing.z};
            auto& probe = volume->probes[p];
            std::array<math::Vec3, 9> coefficients{};
            double total_weight = 0.0;
            for (uint32_t face = 0u; face < 6u; ++face) {
                const math::Vec3 up = face < 4u ? math::Vec3{0, 0, 1} : math::Vec3{0, 1, 0};
                auto camera = rt::BuildPinholeBasis(position, kDirections[face], up,
                    0.5f * kPi, options.face_size, options.face_size);
                camera.near_clip = 1.0e-4f;
                camera.far_clip = options.max_distance;
                const auto frame = backend.TraceBeautyToHost(owner.handle, scene, camera, lighting);
                for (uint32_t py = 0u; py < options.face_size; ++py)
                    for (uint32_t px = 0u; px < options.face_size; ++px) {
                        const float u = 2.0f * (float(px) + 0.5f) / float(options.face_size) - 1.0f;
                        const float v = 2.0f * (float(py) + 0.5f) / float(options.face_size) - 1.0f;
                        const float weight = 1.0f / std::pow(1.0f + u * u + v * v, 1.5f);
                        total_weight += weight;
                        const auto basis = Basis(camera.GenerateRay(px, py).dir);
                        const auto color = frame.ColorAt(px, py);
                        if (!std::isfinite(color.x + color.y + color.z))
                            throw std::runtime_error("nonfinite local lighting sample");
                        for (uint32_t k = 0u; k < 9u; ++k)
                            coefficients[k] = coefficients[k] + color * (weight * basis[k]);
                        if (pass == 0u) {
                            const float depth = frame.DepthAt(px, py);
                            if (!(depth >= 0.0f)) throw std::runtime_error("invalid local lighting depth");
                            probe.distance[size_t{face} * pixels + py * options.face_size + px] =
                                std::min(depth, options.max_distance);
                        }
                    }
            }
            for (uint32_t k = 0u; k < 9u; ++k) {
                const float convolution = k == 0u ? 1.0f : k < 4u ? 2.0f / 3.0f : 0.25f;
                probe.diffuse[k] = probe.diffuse[k] + coefficients[k] *
                    (4.0f * kPi * convolution / static_cast<float>(total_weight));
            }
        }
    }
    volume->Validate();
    return volume;
}
}  // namespace nuka::render
