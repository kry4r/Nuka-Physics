#include "import/cooker/mesh_surface_cooker.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>

#include "sha256.h"

namespace nuka::import::cooker {
namespace {

constexpr uint32_t kSurfaceCacheVersion = 3u;

template <typename T>
void Append(std::string& bytes, const T& value) {
    bytes.append(reinterpret_cast<const char*>(&value), sizeof(T));
}

void AppendString(std::string& bytes, const std::string& value) {
    Append(bytes, static_cast<uint32_t>(value.size()));
    bytes.append(value);
}

void AppendPoint(std::string& bytes, CoverPoint point) {
    Append(bytes, point.x);
    Append(bytes, point.y);
    Append(bytes, point.z);
}

struct Reader {
    const char* cursor;
    const char* end;

    template <typename T>
    bool Read(T& value) {
        if (static_cast<size_t>(end - cursor) < sizeof(T)) return false;
        std::memcpy(&value, cursor, sizeof(T));
        cursor += sizeof(T);
        return true;
    }

    bool String(std::string& value, uint32_t maximum) {
        uint32_t size = 0;
        if (!Read(size) || size > maximum || static_cast<size_t>(end - cursor) < size) return false;
        value.assign(cursor, size);
        cursor += size;
        return true;
    }

    bool Point(CoverPoint& value) {
        return Read(value.x) && Read(value.y) && Read(value.z) &&
            std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
    }
};

std::string CacheKey(const float* vertices, uint32_t vertex_count,
    const uint32_t* indices, uint32_t triangle_count, bool require_convex,
    const MeshSurfaceCookOptions& options, const std::string& backend) {
    std::string settings = "nuka.exact-surface.convex-cover";
    Append(settings, kSurfaceCacheVersion);
    Append(settings, vertex_count);
    Append(settings, triangle_count);
    Append(settings, uint32_t(require_convex));
    Append(settings, uint32_t(options.decompose));
    Append(settings, options.cover.relative_error);
    Append(settings, options.cover.max_parts);
    Append(settings, options.cover.max_planes);
    Append(settings, options.cover.max_cells);
    Append(settings, options.cover.max_operations);
    AppendString(settings, backend);
    sha256::Hasher hash;
    hash.Update(settings.data(), settings.size());
    hash.Update(vertices, static_cast<size_t>(vertex_count) * 3u * sizeof(float));
    hash.Update(indices, static_cast<size_t>(triangle_count) * 3u * sizeof(uint32_t));
    return sha256::ToHex(hash.Final());
}

std::string Encode(const CookedMeshSurface& surface) {
    std::string bytes;
    Append(bytes, kSurfaceCacheVersion);
    AppendString(bytes, surface.cache_key);
    Append(bytes, surface.info.vertex_count);
    Append(bytes, surface.info.triangle_count);
    Append(bytes, surface.info.node_count);
    Append(bytes, surface.info.flags);
    Append(bytes, static_cast<uint32_t>(surface.cover.status));
    AppendString(bytes, surface.cover.backend);
    AppendString(bytes, surface.cover.reason);
    Append(bytes, surface.cover.operations);
    Append(bytes, surface.cover.distance_cells);
    Append(bytes, surface.cover.query_points);
    Append(bytes, uint32_t(surface.cover_hierarchy));
    for (const auto& node : surface.nodes) {
        Append(bytes, node.lower.x); Append(bytes, node.lower.y); Append(bytes, node.lower.z);
        Append(bytes, node.escape);
        Append(bytes, node.upper.x); Append(bytes, node.upper.y); Append(bytes, node.upper.z);
        Append(bytes, node.triangle);
    }
    Append(bytes, static_cast<uint32_t>(surface.cover.parts.size()));
    for (const auto& part : surface.cover.parts) {
        Append(bytes, static_cast<uint32_t>(part.planes.size()));
        for (const auto& plane : part.planes) { AppendPoint(bytes, plane.normal); Append(bytes, plane.offset); }
        Append(bytes, static_cast<uint32_t>(part.vertices.size()));
        for (const auto& vertex : part.vertices) AppendPoint(bytes, vertex);
    }
    sha256::Hasher hash;
    hash.Update(bytes.data(), bytes.size());
    const auto checksum = hash.Final();
    bytes.append(reinterpret_cast<const char*>(checksum.data()), checksum.size());
    return bytes;
}

bool Decode(const std::string& bytes, const float* vertices, uint32_t vertex_count,
    const uint32_t* indices, uint32_t triangle_count, bool require_convex,
    const MeshSurfaceCookOptions& options, CookedMeshSurface& result) {
    constexpr size_t checksum_bytes = 32u;
    if (bytes.size() < checksum_bytes) return false;
    sha256::Hasher hash;
    hash.Update(bytes.data(), bytes.size() - checksum_bytes);
    const auto checksum = hash.Final();
    if (std::memcmp(checksum.data(), bytes.data() + bytes.size() - checksum_bytes, checksum_bytes)) return false;
    Reader reader{bytes.data(), bytes.data() + bytes.size() - checksum_bytes};
    uint32_t version = 0, status = 0, hierarchy = 0;
    std::string key;
    auto& info = result.info;
    if (!reader.Read(version) || version != kSurfaceCacheVersion || !reader.String(key, 64u) ||
        key != result.cache_key || !reader.Read(info.vertex_count) || info.vertex_count != vertex_count ||
        !reader.Read(info.triangle_count) || info.triangle_count != triangle_count ||
        !reader.Read(info.node_count) || uint64_t(info.node_count) != uint64_t(triangle_count) * 2u - 1u ||
        !reader.Read(info.flags) || (info.flags & ~3u) != 0u ||
        (require_convex && info.flags != (collision::kMeshSurfaceClosed | collision::kMeshSurfaceConvex)) ||
        !reader.Read(status) || status > static_cast<uint32_t>(ConvexCoverStatus::BackendFailure) ||
        !reader.String(result.cover.backend, 16u) || !reader.String(result.cover.reason, 4096u) ||
        !reader.Read(result.cover.operations) || !reader.Read(result.cover.distance_cells) ||
        !reader.Read(result.cover.query_points) || !reader.Read(hierarchy) || hierarchy > 1u) return false;
    result.cover.status = static_cast<ConvexCoverStatus>(status);
    result.cover_hierarchy = hierarchy != 0u;
    result.nodes.resize(info.node_count);
    for (auto& node : result.nodes) {
        if (!reader.Read(node.lower.x) || !reader.Read(node.lower.y) || !reader.Read(node.lower.z) ||
            !reader.Read(node.escape) || !reader.Read(node.upper.x) || !reader.Read(node.upper.y) ||
            !reader.Read(node.upper.z) || !reader.Read(node.triangle)) return false;
    }
    uint32_t count = 0;
    if (!reader.Read(count) || count > options.cover.max_parts) return false;
    if ((status == static_cast<uint32_t>(ConvexCoverStatus::Complete)) != (count > 0u)) return false;
    if (result.cover_hierarchy && count < 2u) return false;
    result.cover.parts.resize(count);
    for (auto& part : result.cover.parts) {
        uint32_t planes = 0, points = 0;
        if (!reader.Read(planes) || planes < 4u || planes > options.cover.max_planes) return false;
        part.planes.resize(planes);
        for (auto& plane : part.planes) {
            if (!reader.Point(plane.normal) || !reader.Read(plane.offset) || !std::isfinite(plane.offset))
                return false;
            const auto n = plane.normal;
            if (std::fabs(n.x * n.x + n.y * n.y + n.z * n.z - 1.0) > 1.0e-8) return false;
        }
        if (!reader.Read(points) || points < 4u || uint64_t(points) > uint64_t(options.cover.max_planes) * 2u)
            return false;
        part.vertices.resize(points);
        for (auto& point : part.vertices) if (!reader.Point(point)) return false;
    }
    const collision::MeshSurfaceView source{vertices, indices, result.nodes.data(),
        {vertex_count, triangle_count, info.node_count}};
    return reader.cursor == reader.end && MeshSurfaceTreeValid(source, info);
}

bool Load(const std::filesystem::path& path, const float* vertices, uint32_t vertex_count,
    const uint32_t* indices, uint32_t triangle_count, bool require_convex,
    const MeshSurfaceCookOptions& options, CookedMeshSurface& surface) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) return false;
    const auto length = stream.tellg();
    const uint64_t maximum = 8192ull + (uint64_t(triangle_count) * 2u - 1u) * 32u +
        uint64_t(options.cover.max_parts) * (16ull + uint64_t(options.cover.max_planes) * 80u);
    if (length < 0 || uint64_t(length) > maximum ||
        uint64_t(length) > std::numeric_limits<size_t>::max()) return false;
    std::string bytes(static_cast<size_t>(length), '\0');
    stream.seekg(0);
    if (!stream.read(bytes.data(), static_cast<std::streamsize>(bytes.size()))) return false;
    return Decode(bytes, vertices, vertex_count, indices, triangle_count, require_convex, options, surface);
}

void Store(const std::filesystem::path& path, const CookedMeshSurface& surface) {
    std::error_code error;
    std::filesystem::create_directories(path.parent_path(), error);
    if (error) return;
    static std::atomic<uint64_t> serial{0};
    const auto stamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    const std::filesystem::path staging(path.string() + "." + std::to_string(stamp) + "." +
                                       std::to_string(serial.fetch_add(1u)) + ".tmp");
    const std::string bytes = Encode(surface);
    {
        std::ofstream stream(staging, std::ios::binary | std::ios::trunc);
        if (!stream) return;
        stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        stream.close();
        if (!stream) { std::filesystem::remove(staging, error); return; }
    }
    std::filesystem::rename(staging, path, error);
    if (error) {
        std::filesystem::remove(path, error);
        error.clear();
        std::filesystem::rename(staging, path, error);
        if (error) std::filesystem::remove(staging, error);
    }
}

}  // namespace

CookedMeshSurface CookMeshSurfaceCached(const float* vertices, uint32_t vertex_count,
    const uint32_t* indices, uint32_t triangle_count, bool require_convex,
    const MeshSurfaceCookOptions& options) {
    ValidateMeshSurfaceInput(vertices, vertex_count, indices, triangle_count);
    if (options.decompose && (!std::isfinite(options.cover.relative_error) ||
        !(options.cover.relative_error > 0.0) || options.cover.max_parts == 0u ||
        options.cover.max_planes < 6u || options.cover.max_cells == 0u ||
        options.cover.max_operations == 0u))
        throw std::invalid_argument("Invalid automatic mesh cook budget");
    const std::string backend = options.decompose && !require_convex
        ? MeshQueryBackendName(options.allow_device) : "none";
    const std::string key = CacheKey(vertices, vertex_count, indices, triangle_count,
                                     require_convex, options, backend);
    const std::filesystem::path path = std::filesystem::path(options.cache_directory) / (key + ".nukasurf");
    CookedMeshSurface surface;
    surface.cache_key = key;
    if (!options.cache_directory.empty() &&
        Load(path, vertices, vertex_count, indices, triangle_count, require_convex, options, surface)) {
        surface.cache_hit = true;
        return surface;
    }
    surface = CookMeshSurface(vertices, vertex_count, indices, triangle_count, require_convex);
    surface.cache_key = key;
    if (options.decompose && !require_convex && (surface.info.flags & collision::kMeshSurfaceClosed)) {
        const collision::MeshSurfaceView source{vertices, indices, surface.nodes.data(),
            {vertex_count, triangle_count, surface.info.node_count}};
        try {
            auto queries = CreateMeshQueryBackend(source, surface.info, options.allow_device);
            surface.cover = BuildConvexCover(source, surface.info, options.cover, *queries);
            GroupMeshSurfaceByCover(surface, vertices, indices);
        } catch (const std::exception& failure) {
            surface.cover.parts.clear();
            surface.cover.status = ConvexCoverStatus::BackendFailure;
            surface.cover.backend = backend;
            surface.cover.reason = failure.what();
        }
    }
    if (!options.cache_directory.empty() && surface.cover.status != ConvexCoverStatus::BackendFailure)
        Store(path, surface);
    return surface;
}

}  // namespace nuka::import::cooker
