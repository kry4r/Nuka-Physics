#include "runtime/app/viewer/editor_assets.hpp"

#include "runtime/app/viewer/editor_profile.hpp"

#include <algorithm>
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
    // A sibling first, so the UI never loads a half-written picture.
    fs::path staged = out;
    staged += ".tmp";
    {
        std::ofstream file(staged, std::ios::binary | std::ios::trunc);
        file << "P6\n" << kThumbWidth << " " << kThumbHeight << "\n255\n" << pixels;
    }
    fs::rename(staged, out);
    return true;
}

ThumbnailCache::ThumbnailCache(fs::path dir) : dir_(std::move(dir)) {
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
    char name[24];
    std::snprintf(name, sizeof(name), "%016llx",
                  static_cast<unsigned long long>(
                      Fnv1a(asset.path + "\n" + std::to_string(asset.bytes) + "\n" + std::to_string(asset.stamp))));
    std::lock_guard<std::mutex> lock(mu_);
    if (const auto it = known_.find(name); it != known_.end()) return it->second;
    const fs::path file = dir_ / (std::string(name) + ".ppm");
    std::string& known = known_[name];
    if (fs::exists(file)) known = PathText(file);
    else if (!fs::exists(dir_ / (std::string(name) + ".none"))) {
        queue_.emplace_back(asset, file);
        cv_.notify_one();
    }
    return known;
}

// A picture that cannot be drawn leaves a .none marker, so it is not tried again.
void ThumbnailCache::Run() {
    for (;;) {
        std::pair<AssetFile, fs::path> job;
        {
            std::unique_lock<std::mutex> lock(mu_);
            cv_.wait(lock, [this] { return stop_ || !queue_.empty(); });
            if (stop_) return;
            job = std::move(queue_.front());
            queue_.pop_front();
        }
        const ImportAnalysis a = AnalyzeImport(job.first.path, "legacy");
        const bool drawn = a.scene && RenderThumbnail(*a.scene, job.second);
        if (!drawn) std::ofstream(fs::path(job.second).replace_extension(".none"), std::ios::binary);
        {
            std::lock_guard<std::mutex> lock(mu_);
            known_[PathText(job.second.stem())] = drawn ? PathText(job.second) : std::string();
        }
        ++version_;
    }
}

}  // namespace nuka::runtime::app::viewer
