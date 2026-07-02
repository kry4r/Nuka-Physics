// ---------------------------------------------------------------------------
// nuka::render -- the shared studio render scene (live robot+particle look).
// ---------------------------------------------------------------------------
// Builds the RenderWorld + material palette + ground floor + RasterOptions ONCE,
// refreshes the link poses + the deforming particle surface each frame, and traces
// the offline CUDA path-tracer to a host RGBA8 report. The demo and the C-ABI
// render-beauty verb share this one path so their output is identical.
// ---------------------------------------------------------------------------

#include "render/studio_beauty.hpp"

#include "math/quat.hpp"
#include "render/rt_adapter.hpp"
#include "render/rt_backend.hpp"
#include "render/rt_framebuffer_to_report.hpp"
#include "scene/ecs/registry.hpp"

#include <cstdint>
#include <string>

namespace nuka::render {

namespace {

namespace soft = nuka::runtime::soft;
using nuka::math::Transform;
using nuka::math::Vec3;

constexpr float kStudioPi = 3.14159265358979323846f;

scene::RenderMaterial Mk(float r, float g, float b, float metallic, float rough,
                         float sheen = 0.0f) {
    scene::RenderMaterial m;
    m.base_color[0] = r; m.base_color[1] = g; m.base_color[2] = b; m.base_color[3] = 1.0f;
    m.metallic = metallic; m.roughness = rough; m.sheen = sheen;
    return m;
}

MeshGeometry MakeFloorGeo(float half, float z) {
    MeshGeometry g;
    const float xs[2] = {-half, half};
    const float ys[2] = {-half, half};
    for (int yi = 0; yi < 2; ++yi)
        for (int xi = 0; xi < 2; ++xi)
            g.positions.insert(g.positions.end(), {xs[xi], ys[yi], z});
    g.indices.insert(g.indices.end(), {0u, 1u, 3u, 0u, 3u, 2u});
    return g;
}

// Bake every particle in [first, first+count) into one mesh of radius-r octahedra
// (6 verts / 8 tris each; the beauty tracer's smooth normals round them off).
MeshGeometry BakeParticleSpheres(const std::vector<Vec3>& pos, uint32_t first,
                                 uint32_t count, float r) {
    MeshGeometry g;
    const uint32_t total = static_cast<uint32_t>(pos.size());
    const uint32_t lo = first < total ? first : total;
    const uint32_t n = (count == 0u) ? (total - lo)
                                     : (count < total - lo ? count : total - lo);
    static const float kV[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0},
                                   {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
    static const uint32_t kF[8][3] = {{0, 2, 4}, {2, 1, 4}, {1, 3, 4}, {3, 0, 4},
                                      {2, 0, 5}, {1, 2, 5}, {3, 1, 5}, {0, 3, 5}};
    g.positions.reserve(static_cast<size_t>(n) * 18u);
    g.indices.reserve(static_cast<size_t>(n) * 24u);
    for (uint32_t i = 0; i < n; ++i) {
        const Vec3& p = pos[lo + i];
        const uint32_t base = i * 6u;
        for (int v = 0; v < 6; ++v) {
            g.positions.insert(g.positions.end(),
                               {p.x + r * kV[v][0], p.y + r * kV[v][1],
                                p.z + r * kV[v][2]});
        }
        for (const auto& f : kF)
            g.indices.insert(g.indices.end(), {base + f[0], base + f[1], base + f[2]});
    }
    return g;
}

// Append one scene Registry render material to the studio palette; kNoId when the
// id is unset/unknown (the caller keeps its studio default then).
uint32_t InternSceneMaterial(StudioScene& s, const scene::Registry& registry,
                             uint32_t scene_material_id) {
    if (scene_material_id == kNoId) return kNoId;
    const scene::RenderMaterial* m = registry.GetRenderMaterial(scene_material_id);
    if (m == nullptr) return kNoId;
    s.world.materials.push_back(*m);
    return static_cast<uint32_t>(s.world.materials.size() - 1u);
}

}  // namespace

StudioScene BuildStudioScene(const scene::Registry& registry,
                             const scene::SceneMap& map,
                             const soft::SurfaceTopology& surface_topology,
                             uint32_t width, uint32_t height) {
    std::vector<soft::SurfaceTopology> topologies;
    if (!surface_topology.triangles.empty()) topologies.push_back(surface_topology);
    return BuildStudioScene(registry, map, topologies, width, height);
}

StudioScene BuildStudioScene(const scene::Registry& registry,
                             const scene::SceneMap& map,
                             const std::vector<soft::SurfaceTopology>& surface_topologies,
                             uint32_t width, uint32_t height) {
    StudioScene s;
    s.world = BuildRenderWorld(registry, map);

    // The per-instance link + geom-local rebind (the proven shape->link rebind):
    // a non-static instance follows LinkPose[row] composed with its visual offset.
    const std::size_t ninst = s.world.instances.size();
    s.link_of_instance.assign(ninst, 0u);
    s.visual_local.assign(ninst, Transform::Identity());
    for (std::size_t i = 0; i < ninst; ++i) {
        const PoseSource& ps = s.world.instances[i].pose_source;
        if (ps.kind == PoseSource::Kind::Static || ps.row == kNoId) continue;
        s.link_of_instance[i] = ps.row;
        s.visual_local[i] = s.world.instances[i].cached_visual_local;
    }
    s.link_instance_count = static_cast<uint32_t>(ninst);

    // Record each instance's AUTHORED material id, then APPEND the premium studio
    // palette (the hero look) after the authored table. The role override below
    // points every instance at the palette; UseAuthoredSceneMaterials rebinds.
    s.authored_material_of_instance.assign(ninst, kNoId);
    for (std::size_t i = 0; i < ninst; ++i) {
        const uint32_t m = s.world.instances[i].render_material_id;
        if (m < s.world.materials.size()) s.authored_material_of_instance[i] = m;
    }
    const uint32_t base = static_cast<uint32_t>(s.world.materials.size());
    const uint32_t kMatShell = base + 0u, kMatMetal = base + 1u, kMatFoot = base + 2u;
    const uint32_t kMatAccent = base + 3u, kMatCloth = base + 4u, kMatFloor = base + 5u;
    s.world.materials.push_back(Mk(0.072f, 0.076f, 0.085f, 0.05f, 0.55f));  // shell charcoal
    s.world.materials.push_back(Mk(0.090f, 0.096f, 0.110f, 0.85f, 0.34f));  // dark gunmetal
    s.world.materials.push_back(Mk(0.045f, 0.045f, 0.050f, 0.04f, 0.72f));  // dark rubber foot
    s.world.materials.push_back(Mk(0.16f, 0.180f, 0.225f, 0.65f, 0.30f));   // cool steel accent
    // Deep crimson matte fabric: a saturated dielectric at fabric roughness with a
    // faint grazing sheen for velvet quality, not a glossy glint.
    s.world.materials.push_back(Mk(0.400f, 0.035f, 0.060f, 0.00f, 0.58f, 0.6f));  // crimson silk
    s.world.materials.push_back(Mk(0.190f, 0.196f, 0.210f, 0.02f, 0.82f));  // studio floor
    s.world.default_material_id = kMatShell;

    // Each cooked medium with a triangulated surface becomes one deforming instance;
    // the crimson media material reads as fabric on a cloth and a jelly on a soft body.
    s.surfaces.clear();
    for (const soft::SurfaceTopology& topo : surface_topologies) {
        if (topo.triangles.empty()) continue;
        StudioScene::DeformingSurface ds;
        ds.topology = topo;
        ds.material_id = kMatCloth;
        s.surfaces.push_back(std::move(ds));
    }

    auto link_role_material = [&](uint32_t link, const Transform& vlocal) -> uint32_t {
        if (link == 0u) return kMatShell;
        const uint32_t k = ((link - 1u) % 3u);
        if (k == 0u) return kMatMetal;
        if (k == 1u) return kMatShell;
        return (vlocal.position.z < -0.15f) ? kMatFoot : kMatMetal;
    };
    int accent_inst = -1;
    for (std::size_t i = 0; i < ninst; ++i) {
        s.world.instances[i].render_material_id =
            link_role_material(s.link_of_instance[i], s.visual_local[i]);
        const PoseSource& ps = s.world.instances[i].pose_source;
        if (s.link_of_instance[i] == 0u && ps.kind != PoseSource::Kind::Static)
            accent_inst = static_cast<int>(i);
    }
    if (accent_inst >= 0)
        s.world.instances[static_cast<std::size_t>(accent_inst)].render_material_id = kMatAccent;

    // The studio floor (a large static disc the scene sits on).
    const uint32_t floor_mesh =
        s.world.meshes.InternPrimitive("floor", [&] { return MakeFloorGeo(8.0f, 0.0f); });
    RenderInstance fi;
    fi.mesh_id = floor_mesh; fi.render_material_id = kMatFloor;
    fi.world_xform = Transform::Identity();
    fi.pose_source.kind = PoseSource::Kind::Static;
    s.world.instances.push_back(fi);

    // Cinematic RasterOptions (no implicit ground: the explicit floor mesh carries it).
    RasterOptions& o = s.options;
    o.width = width; o.height = height;
    o.draw_ground = false;
    o.hero_framing = false;
    o.use_camera_override = true;
    o.camera_up = {0.0f, 0.0f, 1.0f};
    o.camera_fov_degrees = 40.0f;
    o.background = {14, 17, 23, 255};
    o.ground_color[0] = 0.180f; o.ground_color[1] = 0.190f; o.ground_color[2] = 0.210f;
    o.contact_shadow_strength = 0.0f;
    o.use_sun_light = true;
    o.sun_direction[0] = 0.42f; o.sun_direction[1] = -0.34f; o.sun_direction[2] = 0.50f;
    o.sun_color[0] = 2.80f; o.sun_color[1] = 2.70f; o.sun_color[2] = 2.55f;
    o.sun_ambient_sky[0] = 0.16f; o.sun_ambient_sky[1] = 0.19f; o.sun_ambient_sky[2] = 0.26f;
    o.sun_ambient_ground[0] = 0.10f; o.sun_ambient_ground[1] = 0.10f; o.sun_ambient_ground[2] = 0.11f;
    o.shadow_strength = 0.9f;
    o.shadow_map_size = 2048u;
    o.shadow_bias = 0.0020f;
    o.sky_gradient = true;
    o.sky_top[0] = 0.22f; o.sky_top[1] = 0.32f; o.sky_top[2] = 0.52f;
    o.sky_bottom[0] = 0.58f; o.sky_bottom[1] = 0.66f; o.sky_bottom[2] = 0.74f;
    return s;
}

void SetStudioSurfaceMaterial(StudioScene& scene, const scene::Registry& registry,
                              std::size_t surface_index, uint32_t scene_material_id) {
    if (surface_index >= scene.surfaces.size()) return;
    const uint32_t slot = InternSceneMaterial(scene, registry, scene_material_id);
    if (slot != kNoId) scene.surfaces[surface_index].material_id = slot;
}

void UseAuthoredSceneMaterials(StudioScene& scene) {
    const std::size_t n = scene.authored_material_of_instance.size();
    for (std::size_t i = 0; i < n && i < scene.world.instances.size(); ++i) {
        const uint32_t m = scene.authored_material_of_instance[i];
        if (m != kNoId) scene.world.instances[i].render_material_id = m;
    }
}

void AddStudioParticleSkin(StudioScene& scene, const scene::Registry& registry,
                           uint32_t scene_material_id, float radius,
                           uint32_t first, uint32_t count) {
    StudioScene::ParticleSkin sk;
    sk.radius = radius > 0.0f ? radius : 0.005f;
    sk.first = first;
    sk.count = count;
    uint32_t slot = InternSceneMaterial(scene, registry, scene_material_id);
    if (slot == kNoId) {
        // Neutral matte granite grey (the default granular grain look).
        scene.world.materials.push_back(Mk(0.42f, 0.385f, 0.34f, 0.0f, 0.85f));
        slot = static_cast<uint32_t>(scene.world.materials.size() - 1u);
    }
    sk.material_id = slot;
    scene.particle_skins.push_back(sk);
}

void PublishStudioScene(StudioScene& scene,
                        const std::vector<Transform>& link_pose,
                        const std::vector<Vec3>& particle_pos,
                        const std::vector<Transform>& body_pose) {
    for (uint32_t i = 0; i < scene.link_instance_count; ++i) {
        const PoseSource& ps = scene.world.instances[i].pose_source;
        if (ps.kind == PoseSource::Kind::Static) {
            continue;  // pinned at its bind pose (a static scene body's visual).
        }
        if (ps.kind == PoseSource::Kind::Body) {   // a free rigid body's visual.
            if (ps.row < body_pose.size())
                scene.world.instances[i].world_xform =
                    body_pose[ps.row] * scene.visual_local[i];
            continue;
        }
        const uint32_t lk = scene.link_of_instance[i];
        if (lk < link_pose.size())
            scene.world.instances[i].world_xform = link_pose[lk] * scene.visual_local[i];
    }

    // Rebuild every deforming surface from the live particles over its own triangle
    // topology; each mesh id is stable (interned once) so the table never grows.
    for (std::size_t si = 0; si < scene.surfaces.size(); ++si) {
        StudioScene::DeformingSurface& surf = scene.surfaces[si];
        if (surf.topology.triangles.empty()) continue;
        MeshGeometry mesh;
        soft::BuildSurfaceMesh(particle_pos, surf.topology, mesh);
        if (surf.mesh_id == kNoId) {
            surf.mesh_id = scene.world.meshes.InternPrimitive(
                "particle_surface" + std::to_string(si), [&] { return mesh; });
            RenderInstance ci;
            ci.mesh_id = surf.mesh_id;
            ci.render_material_id = surf.material_id;
            ci.world_xform = Transform::Identity();
            ci.pose_source.kind = PoseSource::Kind::Static;
            scene.world.instances.push_back(ci);
            surf.instance = scene.world.instances.size() - 1u;
        } else {
            scene.world.meshes.ReplaceGeometry(surf.mesh_id, std::move(mesh));
        }
    }

    // Re-bake every instanced-sphere particle skin from the live positions.
    for (std::size_t si = 0; si < scene.particle_skins.size(); ++si) {
        StudioScene::ParticleSkin& sk = scene.particle_skins[si];
        MeshGeometry mesh =
            BakeParticleSpheres(particle_pos, sk.first, sk.count, sk.radius);
        if (mesh.positions.empty()) continue;
        if (sk.mesh_id == kNoId) {
            sk.mesh_id = scene.world.meshes.InternPrimitive(
                "particle_skin" + std::to_string(si), [&] { return mesh; });
            RenderInstance ci;
            ci.mesh_id = sk.mesh_id;
            ci.render_material_id = sk.material_id;
            ci.world_xform = Transform::Identity();
            ci.pose_source.kind = PoseSource::Kind::Static;
            scene.world.instances.push_back(ci);
            sk.instance = scene.world.instances.size() - 1u;
        } else {
            scene.world.meshes.ReplaceGeometry(sk.mesh_id, std::move(mesh));
        }
    }
}

// ---------------------------------------------------------------------------
// StudioRtRenderer -- the offline CUDA path-tracer over a StudioScene RenderWorld.
// ---------------------------------------------------------------------------
struct StudioRtRenderer::Impl {
    std::unique_ptr<RtBackendI> backend;
    RtSceneHandle* handle = nullptr;
    rt::TwoLevelScene scene;
    bool beauty = true;
    uint32_t samples = 16u;

    Impl() { backend = CreateCudaRtBackend(); }
    ~Impl() { if (backend && handle) backend->FreeScene(handle); }

    void ApplyLighting(const RasterOptions& opts) {
        rt::Light& l = scene.light;
        l.directional = true;
        Vec3 to_sun{opts.sun_direction[0], opts.sun_direction[1], opts.sun_direction[2]};
        if (to_sun.Length() > 1e-6f) to_sun = to_sun.Normalized();
        l.direction = -to_sun;  // opts points TOWARD the sun; rt travels AWAY.
        l.color = {opts.sun_color[0], opts.sun_color[1], opts.sun_color[2]};
        l.intensity = 1.0f;
        scene.ambient.color = {
            0.5f * (opts.sun_ambient_sky[0] + opts.sun_ambient_ground[0]),
            0.5f * (opts.sun_ambient_sky[1] + opts.sun_ambient_ground[1]),
            0.5f * (opts.sun_ambient_sky[2] + opts.sun_ambient_ground[2])};
    }
    rt::PinholeCamera CameraFromOptions(const RasterOptions& opts) {
        const float vfov = opts.camera_fov_degrees * (kStudioPi / 180.0f);
        return rt::BuildPinhole(opts.camera_eye, opts.camera_target, opts.camera_up,
                                vfov, opts.width, opts.height);
    }
    rt::BeautyOptions BeautyFromOptions(const RasterOptions& opts) {
        rt::BeautyOptions b;
        b.samples = samples;
        b.shadow_rays = (opts.shadow_strength > 0.0f) ? 6u : 1u;
        b.sun_angular_radius = 0.045f;
        b.gi_bounces = 1u;
        b.ao_samples = 4u;
        b.ao_radius = 0.7f;
        b.seed = 0x9e3779b9u;
        b.smooth_normals = true;  // fabric wrinkles + shell shade smoothly, not faceted.
        b.sky_top = {opts.sky_top[0], opts.sky_top[1], opts.sky_top[2]};
        b.sky_bottom = {opts.sky_bottom[0], opts.sky_bottom[1], opts.sky_bottom[2]};
        b.sky_ground = {opts.ground_color[0], opts.ground_color[1], opts.ground_color[2]};
        b.fog_color = {opts.fog_color[0], opts.fog_color[1], opts.fog_color[2]};
        b.fog_density = opts.fog_density;
        b.sky_intensity = 0.30f;
        b.download = rt::AovDownloadMask{};
        b.download.depth = false; b.download.normal = false;
        b.download.albedo = false; b.download.uv = false;
        return b;
    }
};

StudioRtRenderer::StudioRtRenderer() : impl_(std::make_unique<Impl>()) {}
StudioRtRenderer::~StudioRtRenderer() = default;

bool StudioRtRenderer::ok() const { return impl_ && impl_->backend != nullptr; }

void StudioRtRenderer::SetBeauty(bool on, uint32_t samples) {
    impl_->beauty = on;
    impl_->samples = samples;
}

VulkanOffscreenReport StudioRtRenderer::Render(const RenderWorld& world,
                                               const RasterOptions& options) {
    Impl& im = *impl_;
    // The surface deforms every frame, so the BLAS is rebuilt over the CURRENT
    // meshes (offline by design): free the prior handle, rebuild, trace.
    if (im.backend && im.handle) im.backend->FreeScene(im.handle);
    im.scene = RenderWorldToTwoLevelScene(world);
    im.handle = im.backend->BuildScene(im.scene);
    im.ApplyLighting(options);
    const rt::PinholeCamera cam = im.CameraFromOptions(options);
    rt::Framebuffer fb;
    if (im.beauty)
        fb = im.backend->TraceBeautyToHost(im.handle, im.scene, cam, im.BeautyFromOptions(options));
    else
        fb = im.backend->TraceToHost(im.handle, im.scene, cam);
    // An HDR environment IS the backdrop: keep the traced radiance on miss pixels
    // (no environment => the flat background fill, byte-identical to today).
    return FramebufferToReport(fb, options.background,
                               im.beauty && world.environment.Enabled());
}

}  // namespace nuka::render
