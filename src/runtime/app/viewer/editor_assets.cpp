#include "runtime/app/viewer/editor_assets.hpp"

#include "render/texture_image.hpp"
#include "runtime/app/viewer/editor_profile.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <system_error>

namespace nuka::runtime::app::viewer {
namespace {

namespace fs = std::filesystem;

constexpr int kScanDepth = 8;
constexpr int kSupersample = 2;
constexpr float kFill = 0.84f;  // the share of the picture the asset spans
constexpr float kYaw = 0.6109f;    // 35 degrees
constexpr float kPitch = 0.4363f;  // 25 degrees
constexpr int kRoundSegments = 16;
constexpr int kRoundRings = 8;
constexpr float kAmbient = 0.28f;
const math::Vec3 kNoMaterialColor{0.70f, 0.72f, 0.75f};
const math::Vec3 kSkyTop{0.105f, 0.115f, 0.145f};
const math::Vec3 kSkyBottom{0.045f, 0.050f, 0.065f};

constexpr float kPi = 3.14159265f;
constexpr int kBallSupersample = 3;
constexpr float kBallRadius = 0.39f;  // of the picture's side
constexpr float kBallYaw = 0.55f;     // the ball turns its seam away and tips its top to the eye
constexpr float kBallTilt = 0.30f;

// The ball's lights in view space (x right, y up, z to the eye), toward the light, with their
// irradiance over pi: a warm key up left, a cool fill and a rim from behind.
struct BallLight {
    math::Vec3 dir;
    math::Vec3 color;
};
const BallLight kBallLights[] = {
    {math::Vec3{-0.55f, 0.62f, 0.56f}.Normalized(), math::Vec3{1.0f, 0.95f, 0.88f} * 1.9f},
    {math::Vec3{0.80f, 0.05f, 0.60f}.Normalized(), math::Vec3{0.55f, 0.65f, 0.85f} * 0.40f},
    {math::Vec3{0.45f, 0.55f, -0.70f}.Normalized(), math::Vec3{1.0f, 1.0f, 1.0f} * 1.2f},
};
const math::Vec3 kStudioFloor{0.030f, 0.030f, 0.035f};
const math::Vec3 kStudioSky{0.42f, 0.46f, 0.55f};
const math::Vec3 kSoftbox{3.0f, 2.85f, 2.65f};  // the key's panel, as reflections see it

struct Tri {
    math::Vec3 p[3];
    math::Vec3 color;
};

void AddTriangles(const std::vector<math::Vec3>& points, const std::vector<uint32_t>& indices,
                  const math::Transform& world, const math::Vec3& color, std::vector<Tri>* out) {
    for (size_t i = 0; i + 2u < indices.size(); i += 3u)
        out->push_back(Tri{{world.TransformPoint(points[indices[i]]), world.TransformPoint(points[indices[i + 1u]]),
                            world.TransformPoint(points[indices[i + 2u]])},
                           color});
}

// A sphere, or a capsule along local Z with its hemispheres `half_height` apart.
void AddRound(float radius, float half_height, const math::Transform& world, const math::Vec3& color, std::vector<Tri>* out) {
    std::vector<math::Vec3> points;
    for (int ring = 0; ring <= kRoundRings + 1; ++ring) {
        // The equator ring repeats, once per hemisphere, so a capsule gets its straight band.
        const int r = ring <= kRoundRings / 2 ? ring : ring - 1;
        const float theta = 3.14159265f * static_cast<float>(r) / static_cast<float>(kRoundRings);
        const float z = radius * std::cos(theta) + (ring <= kRoundRings / 2 ? half_height : -half_height);
        for (int s = 0; s < kRoundSegments; ++s) {
            const float phi = 6.2831853f * static_cast<float>(s) / static_cast<float>(kRoundSegments);
            points.push_back(math::Vec3{radius * std::sin(theta) * std::cos(phi), radius * std::sin(theta) * std::sin(phi), z});
        }
    }
    std::vector<uint32_t> indices;
    for (int ring = 0; ring <= kRoundRings; ++ring) {
        for (int s = 0; s < kRoundSegments; ++s) {
            const auto a = static_cast<uint32_t>(ring * kRoundSegments + s);
            const auto b = static_cast<uint32_t>(ring * kRoundSegments + (s + 1) % kRoundSegments);
            const auto c = a + static_cast<uint32_t>(kRoundSegments);
            const auto d = b + static_cast<uint32_t>(kRoundSegments);
            indices.insert(indices.end(), {a, c, b, b, c, d});
        }
    }
    AddTriangles(points, indices, world, color, out);
}

// A flat-capped cylinder along local Z: rings at +-half_height and a fan per cap.
void AddCylinder(float radius, float half_height, const math::Transform& world, const math::Vec3& color,
                 std::vector<Tri>* out) {
    constexpr auto kSeg = static_cast<uint32_t>(kRoundSegments);
    std::vector<math::Vec3> points;
    for (const float z : {-half_height, half_height})
        for (uint32_t s = 0; s < kSeg; ++s) {
            const float phi = 6.2831853f * static_cast<float>(s) / static_cast<float>(kSeg);
            points.push_back(math::Vec3{radius * std::cos(phi), radius * std::sin(phi), z});
        }
    points.push_back(math::Vec3{0.0f, 0.0f, -half_height});
    points.push_back(math::Vec3{0.0f, 0.0f, half_height});
    std::vector<uint32_t> indices;
    for (uint32_t i = 0; i < kSeg; ++i) {
        const uint32_t j = (i + 1u) % kSeg;
        indices.insert(indices.end(), {i, j, kSeg + j, i, kSeg + j, kSeg + i});
        indices.insert(indices.end(), {2u * kSeg, j, i, 2u * kSeg + 1u, kSeg + i, kSeg + j});
    }
    AddTriangles(points, indices, world, color, out);
}

void AddShape(const scene::CollisionShapeRecord& sh, const math::Transform& world, const math::Vec3& color,
              std::vector<Tri>* out) {
    switch (sh.type) {
        case scene::ShapeType::Box: {
            std::vector<math::Vec3> corners;
            for (int i = 0; i < 8; ++i)
                corners.push_back(math::Vec3{(i & 1) != 0 ? sh.half_extents.x : -sh.half_extents.x,
                                             (i & 2) != 0 ? sh.half_extents.y : -sh.half_extents.y,
                                             (i & 4) != 0 ? sh.half_extents.z : -sh.half_extents.z});
            AddTriangles(corners, {0, 2, 1, 1, 2, 3, 4, 5, 6, 5, 7, 6, 0, 1, 4, 1, 5, 4,
                                   2, 6, 3, 3, 6, 7, 0, 4, 2, 2, 4, 6, 1, 3, 5, 3, 7, 5},
                         world, color, out);
            break;
        }
        case scene::ShapeType::Sphere: AddRound(sh.radius, 0.0f, world, color, out); break;
        case scene::ShapeType::Capsule: AddRound(sh.radius, sh.half_height, world, color, out); break;
        case scene::ShapeType::Cylinder: AddCylinder(sh.radius, sh.half_height, world, color, out); break;
        case scene::ShapeType::ConvexHull:
        case scene::ShapeType::TriMesh: {
            std::vector<math::Vec3> points;
            for (size_t i = 0; i + 2u < sh.mesh_vertices.size(); i += 3u)
                points.push_back(math::Vec3{sh.mesh_vertices[i], sh.mesh_vertices[i + 1u], sh.mesh_vertices[i + 2u]});
            AddTriangles(points, sh.mesh_indices, world, color, out);
            break;
        }
        case scene::ShapeType::Plane:
        case scene::ShapeType::HeightField: break;
    }
}

bool Visual(const scene::CollisionShapeRecord& sh) { return sh.contype == 0u && sh.conaffinity == 0u; }

// What the scene shows: a body's visual shapes when it has any, else its colliders.
std::vector<Tri> SceneTriangles(const scene::SceneIR& s) {
    std::vector<uint8_t> has_visual(s.RigidBodyCount() + 1u, 0u);
    const auto slot = [&](scene::BodyId b) { return b == scene::kInvalidBody ? s.RigidBodyCount() : static_cast<size_t>(b); };
    for (const scene::CollisionShapeRecord& sh : s.Shapes())
        if (Visual(sh)) has_visual[slot(sh.body_id)] = 1u;
    std::vector<Tri> tris;
    for (const scene::CollisionShapeRecord& sh : s.Shapes()) {
        if (Visual(sh) != (has_visual[slot(sh.body_id)] != 0u)) continue;
        const math::Transform body = sh.body_id != scene::kInvalidBody ? BodyWorld(s, sh.body_id) : math::Transform::Identity();
        const math::Vec3 color = sh.material_id < s.MaterialCount() ? s.GetMaterial(sh.material_id).base_color : kNoMaterialColor;
        AddShape(sh, body * sh.local_transform, color, &tris);
    }
    return tris;
}

unsigned char Byte(float linear) {
    const float srgb = std::pow(std::clamp(linear, 0.0f, 1.0f), 1.0f / 2.2f);
    return static_cast<unsigned char>(std::lround(srgb * 255.0f));
}

float Edge(float ax, float ay, float bx, float by, float px, float py) { return (bx - ax) * (py - ay) - (by - ay) * (px - ax); }

uint64_t Fnv1a(const std::string& bytes) {
    uint64_t hash = 14695981039346656037ull;
    for (const char c : bytes) {
        hash ^= static_cast<unsigned char>(c);
        hash *= 1099511628211ull;
    }
    return hash;
}

// A sibling first, so the UI never loads a half-written picture.
void WritePpm(const fs::path& out, int width, int height, const std::string& pixels) {
    fs::path staged = out;
    staged += ".tmp";
    {
        std::ofstream file(staged, std::ios::binary | std::ios::trunc);
        file << "P6\n" << width << " " << height << "\n255\n" << pixels;
    }
    fs::rename(staged, out);
}

math::Vec3 Mul(const math::Vec3& a, const math::Vec3& b) { return math::Vec3{a.x * b.x, a.y * b.y, a.z * b.z}; }
math::Vec3 Lerp(const math::Vec3& a, const math::Vec3& b, float t) { return a + (b - a) * t; }

// The texel channels at (u, v), bilinear and wrapping, grey spread over rgb; sRGB reads back linear.
std::array<float, 4> Sample(const rt::Texture& t, float u, float v) {
    const auto w = static_cast<int>(t.width), h = static_cast<int>(t.height);
    const float x = (u - std::floor(u)) * static_cast<float>(w) - 0.5f;
    const float y = (v - std::floor(v)) * static_cast<float>(h) - 0.5f;
    const int x0 = static_cast<int>(std::floor(x)), y0 = static_cast<int>(std::floor(y));
    const float fx = x - static_cast<float>(x0), fy = y - static_cast<float>(y0);
    std::array<float, 4> out{};
    for (int k = 0; k < 4; ++k) {
        const int tx = ((x0 + (k & 1)) % w + w) % w, ty = ((y0 + (k >> 1)) % h + h) % h;
        const float weight = ((k & 1) != 0 ? fx : 1.0f - fx) * ((k >> 1) != 0 ? fy : 1.0f - fy);
        const float* p = &t.texels[(static_cast<size_t>(ty) * t.width + static_cast<size_t>(tx)) * t.channels];
        const bool rgb = t.channels >= 3u;
        out[0] += weight * p[0];
        out[1] += weight * (rgb ? p[1] : p[0]);
        out[2] += weight * (rgb ? p[2] : p[0]);
        out[3] += weight * (t.channels == 4u ? p[3] : t.channels == 2u ? p[1] : 1.0f);
    }
    if (t.srgb != 0u)
        for (int c = 0; c < 3; ++c)
            out[c] = out[c] <= 0.04045f ? out[c] / 12.92f : std::pow((out[c] + 0.055f) / 1.055f, 2.4f);
    return out;
}

// What a direction sees of the studio: a sky over a dark floor, and the softbox up left.
math::Vec3 Studio(const math::Vec3& d) {
    const float sky = std::clamp(d.y * 0.6f + 0.45f, 0.0f, 1.0f);
    const float box = std::clamp((d.Dot(kBallLights[0].dir) - 0.86f) / 0.08f, 0.0f, 1.0f);
    return Lerp(kStudioFloor, kStudioSky, sky * sky) + kSoftbox * box;
}

// The studio light a surface facing `n` gathers.
math::Vec3 Irradiance(const math::Vec3& n) {
    return Lerp(kStudioFloor, kStudioSky, std::clamp(n.y * 0.5f + 0.5f, 0.0f, 1.0f)) * 0.8f;
}

math::Vec3 Aces(const math::Vec3& c) {
    const auto f = [](float x) {
        return std::clamp(x * (2.51f * x + 0.03f) / (x * (2.43f * x + 0.59f) + 0.14f), 0.0f, 1.0f);
    };
    return math::Vec3{f(c.x), f(c.y), f(c.z)};
}

}  // namespace

std::vector<AssetFile> ScanAssets(const std::vector<std::string>& roots) {
    std::vector<AssetFile> out;
    for (const std::string& root : roots) {
        std::error_code ec;
        const fs::path base = Utf8Path(root);
        if (!fs::is_directory(base, ec)) continue;
        const auto end = fs::recursive_directory_iterator();
        for (auto it = fs::recursive_directory_iterator(base, fs::directory_options::skip_permission_denied, ec); it != end;
             it.increment(ec)) {
            const std::string name = PathText(it->path().filename());
            if (it->is_directory(ec)) {
                if (name.front() == '.' || name == "__pycache__" || it.depth() + 1 >= kScanDepth) it.disable_recursion_pending();
                continue;
            }
            AssetFile asset = AssetAt(PathText(it->path()), root);
            if (asset.kind != AssetKind::Unknown) out.push_back(std::move(asset));
        }
    }
    std::sort(out.begin(), out.end(),
              [](const AssetFile& a, const AssetFile& b) { return a.root != b.root ? a.root < b.root : a.path < b.path; });
    return out;
}

AssetFile AssetAt(const std::string& path, const std::string& root) {
    std::error_code ec;
    const fs::path file = Utf8Path(path);
    return AssetFile{NormalPath(path), PathText(file.filename()), root, AssetKindOf(path),
                     static_cast<uint64_t>(fs::file_size(file, ec)),
                     static_cast<int64_t>(fs::last_write_time(file, ec).time_since_epoch().count())};
}

// An orthographic view from above and to the side, flat shaded under one light, supersampled.
bool RenderThumbnail(const scene::SceneIR& s, const fs::path& out) {
    const std::vector<Tri> tris = SceneTriangles(s);
    if (tris.empty()) return false;
    const math::Vec3 eye{std::cos(kPitch) * std::cos(kYaw), std::cos(kPitch) * std::sin(kYaw), std::sin(kPitch)};
    const math::Vec3 forward = -eye;
    const math::Vec3 right = forward.Cross(math::Vec3{0.0f, 0.0f, 1.0f}).Normalized();
    const math::Vec3 up = right.Cross(forward);
    const math::Vec3 light = (eye * 0.7f + up * 0.6f - right * 0.4f).Normalized();

    const float fmax = std::numeric_limits<float>::max();
    float x0 = fmax, x1 = -fmax, y0 = fmax, y1 = -fmax;
    for (const Tri& t : tris) {
        for (const math::Vec3& p : t.p) {
            x0 = std::min(x0, p.Dot(right));
            x1 = std::max(x1, p.Dot(right));
            y0 = std::min(y0, p.Dot(up));
            y1 = std::max(y1, p.Dot(up));
        }
    }
    const int w = kThumbWidth * kSupersample, h = kThumbHeight * kSupersample;
    const float scale = kFill * std::min(static_cast<float>(w) / std::max(x1 - x0, 1e-6f),
                                         static_cast<float>(h) / std::max(y1 - y0, 1e-6f));
    const float cx = 0.5f * (x0 + x1), cy = 0.5f * (y0 + y1);

    std::vector<math::Vec3> color(static_cast<size_t>(w) * static_cast<size_t>(h));
    std::vector<float> depth(color.size(), fmax);
    for (int y = 0; y < h; ++y) {
        const float k = static_cast<float>(y) / static_cast<float>(h - 1);
        for (int x = 0; x < w; ++x) color[static_cast<size_t>(y * w + x)] = kSkyTop * (1.0f - k) + kSkyBottom * k;
    }
    for (const Tri& t : tris) {
        float sx[3], sy[3], sz[3];
        for (int i = 0; i < 3; ++i) {
            sx[i] = 0.5f * static_cast<float>(w) + (t.p[i].Dot(right) - cx) * scale;
            sy[i] = 0.5f * static_cast<float>(h) - (t.p[i].Dot(up) - cy) * scale;
            sz[i] = t.p[i].Dot(forward);
        }
        const float area = Edge(sx[0], sy[0], sx[1], sy[1], sx[2], sy[2]);
        if (std::fabs(area) < 1e-9f) continue;
        const math::Vec3 n = (t.p[1] - t.p[0]).Cross(t.p[2] - t.p[0]).Normalized();
        const math::Vec3 shade = t.color * (kAmbient + (1.0f - kAmbient) * std::fabs(n.Dot(light)));
        const int bx0 = std::max(0, static_cast<int>(std::floor(std::min({sx[0], sx[1], sx[2]}))));
        const int bx1 = std::min(w - 1, static_cast<int>(std::ceil(std::max({sx[0], sx[1], sx[2]}))));
        const int by0 = std::max(0, static_cast<int>(std::floor(std::min({sy[0], sy[1], sy[2]}))));
        const int by1 = std::min(h - 1, static_cast<int>(std::ceil(std::max({sy[0], sy[1], sy[2]}))));
        for (int y = by0; y <= by1; ++y) {
            for (int x = bx0; x <= bx1; ++x) {
                const float px = static_cast<float>(x) + 0.5f, py = static_cast<float>(y) + 0.5f;
                const float w0 = Edge(sx[1], sy[1], sx[2], sy[2], px, py) / area;
                const float w1 = Edge(sx[2], sy[2], sx[0], sy[0], px, py) / area;
                const float w2 = 1.0f - w0 - w1;
                if (w0 < 0.0f || w1 < 0.0f || w2 < 0.0f) continue;
                const float z = w0 * sz[0] + w1 * sz[1] + w2 * sz[2];
                const size_t at = static_cast<size_t>(y * w + x);
                if (z >= depth[at]) continue;
                depth[at] = z;
                color[at] = shade;
            }
        }
    }

    std::string pixels(static_cast<size_t>(kThumbWidth * kThumbHeight * 3), '\0');
    for (int y = 0; y < kThumbHeight; ++y) {
        for (int x = 0; x < kThumbWidth; ++x) {
            math::Vec3 sum{};
            for (int dy = 0; dy < kSupersample; ++dy)
                for (int dx = 0; dx < kSupersample; ++dx)
                    sum += color[static_cast<size_t>((y * kSupersample + dy) * w + x * kSupersample + dx)];
            sum /= static_cast<float>(kSupersample * kSupersample);
            const size_t at = static_cast<size_t>((y * kThumbWidth + x) * 3);
            pixels[at] = static_cast<char>(Byte(sum.x));
            pixels[at + 1u] = static_cast<char>(Byte(sum.y));
            pixels[at + 2u] = static_cast<char>(Byte(sum.z));
        }
    }
    WritePpm(out, kThumbWidth, kThumbHeight, pixels);
    return true;
}

// Up to 4 x 4 texels average into a pixel; the image keeps its aspect on the thumbnail backdrop.
bool RenderImage(const rt::Texture& image, int width, int height, const fs::path& out) {
    if (image.Empty()) return false;
    const float fit = std::min(static_cast<float>(width) / static_cast<float>(image.width),
                               static_cast<float>(height) / static_cast<float>(image.height));
    const int w = std::max(1, static_cast<int>(std::lround(static_cast<float>(image.width) * fit)));
    const int h = std::max(1, static_cast<int>(std::lround(static_cast<float>(image.height) * fit)));
    const int x0 = (width - w) / 2, y0 = (height - h) / 2;
    const float back[3] = {Byte(kSkyBottom.x) / 255.0f, Byte(kSkyBottom.y) / 255.0f, Byte(kSkyBottom.z) / 255.0f};
    const uint32_t ch = image.channels;
    const float sx = static_cast<float>(image.width) / static_cast<float>(w);
    const float sy = static_cast<float>(image.height) / static_cast<float>(h);
    const int nx = std::clamp(static_cast<int>(std::ceil(sx)), 1, 4);
    const int ny = std::clamp(static_cast<int>(std::ceil(sy)), 1, 4);
    // Sample `k` of `n` across pixel `p`, on an axis of `size` texels `step` to a pixel.
    const auto texel = [](int p, int k, int n, float step, uint32_t size) {
        const float at = (static_cast<float>(p) + (static_cast<float>(k) + 0.5f) / static_cast<float>(n)) * step;
        return std::min(size - 1u, static_cast<uint32_t>(at));
    };
    std::string pixels(static_cast<size_t>(width * height * 3), '\0');
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float c[3] = {back[0], back[1], back[2]};
            if (x >= x0 && x < x0 + w && y >= y0 && y < y0 + h) {
                float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                for (int j = 0; j < ny; ++j) {
                    for (int i = 0; i < nx; ++i) {
                        const uint32_t tx = texel(x - x0, i, nx, sx, image.width);
                        const uint32_t ty = texel(y - y0, j, ny, sy, image.height);
                        const float* t = &image.texels[(static_cast<size_t>(ty) * image.width + tx) * ch];
                        const bool rgb = ch >= 3u;
                        sum[0] += t[0];
                        sum[1] += rgb ? t[1] : t[0];
                        sum[2] += rgb ? t[2] : t[0];
                        sum[3] += ch == 4u ? t[3] : ch == 2u ? t[1] : 1.0f;
                    }
                }
                const float n = static_cast<float>(nx * ny), a = sum[3] / n;
                for (int k = 0; k < 3; ++k) c[k] = sum[k] / n * a + back[k] * (1.0f - a);
            }
            const size_t at = static_cast<size_t>((y * width + x) * 3);
            for (size_t k = 0; k < 3u; ++k)
                pixels[at + k] = static_cast<char>(std::lround(std::clamp(c[k], 0.0f, 1.0f) * 255.0f));
        }
    }
    WritePpm(out, width, height, pixels);
    return true;
}

// Ray cast per subpixel: the ball's look at the hit, metallic-roughness shading under three lights
// and the studio, on the thumbnails' backdrop with a contact shadow. Maps use UVs of the ball's
// latitude and longitude, or a triplanar projection over a ball a metre across.
bool RenderMaterialBall(const scene::RenderMaterial& m, int size, const fs::path& out) {
    const auto load = [](const std::string& path, bool srgb) {
        return path.empty() ? rt::Texture{} : render::LoadTexture(path, srgb);
    };
    const rt::Texture albedo_map = load(m.albedo_map, true), emissive_map = load(m.emissive_map, true);
    const rt::Texture rough_map = load(m.roughness_map, false), metal_map = load(m.metallic_map, false);
    const rt::Texture normal_map = load(m.normal_map, false);
    const int n = size * kBallSupersample;
    const float side = static_cast<float>(n);
    const float radius = kBallRadius * side, cx = 0.5f * side, cy = 0.46f * side;
    const float cyaw = std::cos(kBallYaw), syaw = std::sin(kBallYaw);
    const float ctilt = std::cos(kBallTilt), stilt = std::sin(kBallTilt);
    const auto to_ball = [&](const math::Vec3& d) {
        const math::Vec3 t{d.x, d.y * ctilt + d.z * stilt, -d.y * stilt + d.z * ctilt};
        return math::Vec3{t.x * cyaw + t.z * syaw, t.y, -t.x * syaw + t.z * cyaw};
    };
    const auto to_view = [&](const math::Vec3& b) {
        const math::Vec3 t{b.x * cyaw - b.z * syaw, b.y, b.x * syaw + b.z * cyaw};
        return math::Vec3{t.x, t.y * ctilt - t.z * stilt, t.y * stilt + t.z * ctilt};
    };
    const auto backdrop = [&](float x, float y) {
        const float sx = (x - cx) / (radius * 0.95f), sy = (y - cy - radius * 1.02f) / (radius * 0.16f);
        return Lerp(kSkyTop, kSkyBottom, std::clamp(y / side, 0.0f, 1.0f)) *
               (1.0f - 0.6f * std::exp(-(sx * sx + sy * sy)));
    };
    const math::Vec3 base_color{m.base_color[0], m.base_color[1], m.base_color[2]};
    const math::Vec3 emissive{m.emissive[0], m.emissive[1], m.emissive[2]};
    const math::Vec3 view{0.0f, 0.0f, 1.0f};

    std::vector<math::Vec3> color(static_cast<size_t>(n) * static_cast<size_t>(n));
    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
            const float px = static_cast<float>(x) + 0.5f, py = static_cast<float>(y) + 0.5f;
            const float dx = (px - cx) / radius, dy = (cy - py) / radius;
            const math::Vec3 back = backdrop(px, py);
            math::Vec3& pixel = color[static_cast<size_t>(y * n + x)];
            if (dx * dx + dy * dy >= 1.0f) {
                pixel = back;
                continue;
            }
            const math::Vec3 o = to_ball(math::Vec3{dx, dy, std::sqrt(1.0f - dx * dx - dy * dy)});
            const float phi = std::atan2(o.x, o.z), theta = std::acos(std::clamp(o.y, -1.0f, 1.0f));
            const auto at = [&](const rt::Texture& t) {
                if (!m.triplanar)
                    return Sample(t, (0.5f + phi / (2.0f * kPi)) * 2.0f * m.uv_scale, theta / kPi * m.uv_scale);
                const float wx = std::pow(std::fabs(o.x), 4.0f), wy = std::pow(std::fabs(o.y), 4.0f);
                const float wz = std::pow(std::fabs(o.z), 4.0f), sum = wx + wy + wz;
                const float k = 0.5f * m.uv_scale;
                const std::array<float, 4> a = Sample(t, o.z * k, -o.y * k), b = Sample(t, o.x * k, o.z * k);
                const std::array<float, 4> c = Sample(t, o.x * k, -o.y * k);
                std::array<float, 4> mix{};
                for (size_t i = 0; i < 4u; ++i) mix[i] = (a[i] * wx + b[i] * wy + c[i] * wz) / sum;
                return mix;
            };
            math::Vec3 albedo = base_color, glow = emissive;
            float alpha = m.base_color[3] * m.opacity, rough = m.roughness, metal = m.metallic;
            if (!albedo_map.Empty()) {
                const std::array<float, 4> t = at(albedo_map);
                albedo = Mul(albedo, math::Vec3{t[0], t[1], t[2]});
                alpha *= t[3];
            }
            if (!emissive_map.Empty()) {
                const std::array<float, 4> t = at(emissive_map);
                glow = Mul(glow, math::Vec3{t[0], t[1], t[2]});
            }
            if (!rough_map.Empty()) rough *= at(rough_map)[std::min(m.roughness_channel, 2u)];
            if (!metal_map.Empty()) metal *= at(metal_map)[std::min(m.metallic_channel, 2u)];
            rough = std::clamp(rough, 0.04f, 1.0f);
            metal = std::clamp(metal, 0.0f, 1.0f);
            math::Vec3 normal = o;
            if (!normal_map.Empty() && !m.triplanar) {
                // Tangent along rising u, bitangent toward the top, as glTF's maps expect.
                const std::array<float, 4> t = at(normal_map);
                const math::Vec3 tangent{std::cos(phi), 0.0f, -std::sin(phi)};
                const math::Vec3 bitangent{-std::cos(theta) * std::sin(phi), std::sin(theta),
                                           -std::cos(theta) * std::cos(phi)};
                normal = (tangent * (t[0] * 2.0f - 1.0f) + bitangent * (t[1] * 2.0f - 1.0f) + o * (t[2] * 2.0f - 1.0f))
                             .Normalized();
            }
            const math::Vec3 nrm = to_view(normal);
            const float nv = std::max(nrm.Dot(view), 1e-4f);
            const math::Vec3 f0 = Lerp(math::Vec3{0.04f, 0.04f, 0.04f}, albedo, metal);
            const float a2 = rough * rough * rough * rough, k = (rough + 1.0f) * (rough + 1.0f) / 8.0f;
            math::Vec3 lit = Irradiance(nrm), specular{};
            for (const BallLight& l : kBallLights) {
                const float nl = nrm.Dot(l.dir);
                if (nl <= 0.0f) continue;
                const math::Vec3 h = (l.dir + view).Normalized();
                const float nh = std::max(nrm.Dot(h), 0.0f), vh = std::max(view.Dot(h), 0.0f);
                const float d = nh * nh * (a2 - 1.0f) + 1.0f;
                const float g = nl / (nl * (1.0f - k) + k) * nv / (nv * (1.0f - k) + k);
                const math::Vec3 f = f0 + (math::Vec3{1.0f, 1.0f, 1.0f} - f0) * std::pow(1.0f - vh, 5.0f);
                specular += Mul(f, l.color) * (a2 / (d * d) * g / (4.0f * nv));
                lit += l.color * nl;
            }
            // The studio's reflection, blurred toward its irradiance by roughness, weighted by the
            // split-sum fit of Karis's mobile approximation.
            const math::Vec3 r = nrm * (2.0f * nv) - view;
            const math::Vec3 env = Lerp(Studio(r), Irradiance(r), std::clamp(rough * 1.5f, 0.0f, 1.0f));
            const float c0 = 1.0f - rough, c1 = -0.0275f * rough + 0.0425f;
            const float a004 = std::min(c0 * c0, std::exp2(-9.28f * nv)) * c0 + c1;
            const float fa = -1.04f * a004 - 0.572f * rough + 1.04f, fb = 1.04f * a004 + 0.022f * rough - 0.04f;
            specular += Mul(env, f0 * fa + math::Vec3{fb, fb, fb});
            math::Vec3 diffuse = Mul(albedo, lit) * (1.0f - metal);
            if (m.transmission > 0.0f) {
                const float bend = radius * 0.5f * (m.ior - 1.0f);
                const math::Vec3 through = Mul(backdrop(px - nrm.x * bend, py + nrm.y * bend), albedo);
                diffuse = Lerp(diffuse, through, std::clamp(m.transmission, 0.0f, 1.0f));
            }
            pixel = Lerp(back, Aces(diffuse + specular + glow), std::clamp(alpha, 0.0f, 1.0f));
        }
    }

    std::string pixels(static_cast<size_t>(size * size * 3), '\0');
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            math::Vec3 sum{};
            for (int dy = 0; dy < kBallSupersample; ++dy)
                for (int dx = 0; dx < kBallSupersample; ++dx)
                    sum += color[static_cast<size_t>((y * kBallSupersample + dy) * n + x * kBallSupersample + dx)];
            sum /= static_cast<float>(kBallSupersample * kBallSupersample);
            const size_t at = static_cast<size_t>((y * size + x) * 3);
            pixels[at] = static_cast<char>(Byte(sum.x));
            pixels[at + 1u] = static_cast<char>(Byte(sum.y));
            pixels[at + 2u] = static_cast<char>(Byte(sum.z));
        }
    }
    WritePpm(out, size, size, pixels);
    return true;
}

ThumbnailCache::ThumbnailCache(fs::path dir, int image_width, int image_height)
    : dir_(std::move(dir)), image_width_(image_width), image_height_(image_height) {
    fs::create_directories(dir_);
    worker_ = std::thread([this] { Run(); });
}

ThumbnailCache::~ThumbnailCache() {
    {
        std::lock_guard<std::mutex> lock(mu_);
        stop_ = true;
    }
    cv_.notify_all();
    worker_.join();
}

std::string ThumbnailCache::Thumbnail(const AssetFile& asset) {
    return Picture(asset.path + "\n" + std::to_string(asset.bytes) + "\n" + std::to_string(asset.stamp),
                   [this, asset](const fs::path& out) {
                       if (asset.kind == AssetKind::Image)
                           return RenderImage(render::LoadTexture(asset.path, false), image_width_, image_height_, out);
                       const ImportAnalysis a = AnalyzeImport(asset.path, ImportOptions{"legacy"});
                       return a.scene && RenderThumbnail(*a.scene, out);
                   });
}

std::string ThumbnailCache::Ball(const scene::RenderMaterial& m, const std::string& key, int size) {
    return Picture("ball\n" + std::to_string(size) + "\n" + key,
                   [m, size](const fs::path& out) { return RenderMaterialBall(m, size, out); });
}

std::string ThumbnailCache::Picture(const std::string& key, Draw draw) {
    char name[24];
    std::snprintf(name, sizeof(name), "%016llx", static_cast<unsigned long long>(Fnv1a(key)));
    std::lock_guard<std::mutex> lock(mu_);
    if (const auto it = known_.find(name); it != known_.end()) return it->second;
    const fs::path file = dir_ / (std::string(name) + ".ppm");
    std::string& known = known_[name];
    if (fs::exists(file)) known = PathText(file);
    else if (!fs::exists(dir_ / (std::string(name) + ".none"))) {
        queue_.emplace_back(std::move(draw), file);
        cv_.notify_one();
    }
    return known;
}

// A picture that cannot be drawn leaves a .none marker, so it is not tried again.
void ThumbnailCache::Run() {
    for (;;) {
        std::pair<Draw, fs::path> job;
        {
            std::unique_lock<std::mutex> lock(mu_);
            cv_.wait(lock, [this] { return stop_ || !queue_.empty(); });
            if (stop_) return;
            job = std::move(queue_.front());
            queue_.pop_front();
        }
        const bool drawn = job.first(job.second);
        if (!drawn) std::ofstream(fs::path(job.second).replace_extension(".none"), std::ios::binary);
        {
            std::lock_guard<std::mutex> lock(mu_);
            known_[PathText(job.second.stem())] = drawn ? PathText(job.second) : std::string();
        }
        ++version_;
    }
}

}  // namespace nuka::runtime::app::viewer
