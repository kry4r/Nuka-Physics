// ---------------------------------------------------------------------------
// nuka::import::cooker – V-HACD convex decomposition implementation
// ---------------------------------------------------------------------------
// COOK-TIME / HOST-CPU ONLY (master plan §5.6). This is the single translation
// unit that compiles the V-HACD implementation (ENABLE_VHACD_IMPLEMENTATION).
// ---------------------------------------------------------------------------

#include "import/cooker/convex_decomposition.hpp"

#include "sha256.h"

// Compile the V-HACD implementation into exactly this TU.
#define ENABLE_VHACD_IMPLEMENTATION 1
#include "VHACD.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <numeric>

namespace nuka::import::cooker {

namespace {

// --- V-HACD configuration ---------------------------------------------------
// Deterministic config: async OFF (single-threaded), no task runner, no
// callbacks, fixed params. V-HACD's algorithm is deterministic given fixed
// params on a single thread; the multi-threaded path can reorder work.
VHACD::IVHACD::Parameters MakeVhacdParameters(const ConvexDecompositionParams& params) {
    VHACD::IVHACD::Parameters p;
    p.m_callback   = nullptr;
    p.m_logger     = nullptr;
    p.m_taskRunner = nullptr;
    p.m_asyncACD   = false;                       // <- determinism: single-threaded
    p.m_maxConvexHulls = params.max_pieces;
    p.m_resolution     = params.voxel_resolution;
    p.m_minimumVolumePercentErrorAllowed =
        static_cast<double>(params.concavity_threshold);
    p.m_maxNumVerticesPerCH = params.max_vertices_per_piece;
    p.m_shrinkWrap          = params.project_hull_vertices;
    p.m_fillMode            = VHACD::FillMode::FLOOD_FILL;
    p.m_maxRecursionDepth   = 10;
    p.m_minEdgeLength       = 2;
    p.m_findBestPlane       = false;
    return p;
}

// --- Canonicalization -------------------------------------------------------
// Sort a single hull's vertices into a stable order and remap its triangle
// indices, so the output byte layout does not depend on V-HACD's internal
// vertex emission order.
void CanonicalizePiece(ConvexPiece& piece) {
    const uint32_t vcount = static_cast<uint32_t>(piece.vertices.size() / 3);
    if (vcount == 0) {
        return;
    }

    // Build a permutation that sorts vertices lexicographically by (x,y,z).
    std::vector<uint32_t> order(vcount);
    std::iota(order.begin(), order.end(), 0u);
    const std::vector<float>& v = piece.vertices;
    std::sort(order.begin(), order.end(), [&v](uint32_t a, uint32_t b) {
        const float* va = &v[a * 3];
        const float* vb = &v[b * 3];
        if (va[0] != vb[0]) return va[0] < vb[0];
        if (va[1] != vb[1]) return va[1] < vb[1];
        return va[2] < vb[2];
    });

    // old index -> new (sorted) index
    std::vector<uint32_t> remap(vcount);
    for (uint32_t new_i = 0; new_i < vcount; ++new_i) {
        remap[order[new_i]] = new_i;
    }

    std::vector<float> sorted;
    sorted.resize(piece.vertices.size());
    for (uint32_t new_i = 0; new_i < vcount; ++new_i) {
        const uint32_t old_i = order[new_i];
        sorted[new_i * 3 + 0] = piece.vertices[old_i * 3 + 0];
        sorted[new_i * 3 + 1] = piece.vertices[old_i * 3 + 1];
        sorted[new_i * 3 + 2] = piece.vertices[old_i * 3 + 2];
    }
    piece.vertices = std::move(sorted);

    for (uint32_t& idx : piece.indices) {
        idx = remap[idx];
    }
}

// Stable ordering key for pieces: volume, then aabb_min, then aabb_max. This
// removes any run-to-run reordering of the hull list.
bool PieceLess(const ConvexPiece& a, const ConvexPiece& b) {
    if (a.volume != b.volume) return a.volume < b.volume;
    for (int i = 0; i < 3; ++i) {
        if (a.aabb_min[i] != b.aabb_min[i]) return a.aabb_min[i] < b.aabb_min[i];
    }
    for (int i = 0; i < 3; ++i) {
        if (a.aabb_max[i] != b.aabb_max[i]) return a.aabb_max[i] < b.aabb_max[i];
    }
    // Final tie-break on vertex count then bytes, so identical-shaped pieces
    // still have a total order.
    if (a.vertices.size() != b.vertices.size()) {
        return a.vertices.size() < b.vertices.size();
    }
    return a.vertices < b.vertices;
}

// --- Serialization (for the on-disk cache) ----------------------------------
template <typename T>
void AppendPod(std::string& out, const T& value) {
    out.append(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T>
bool ReadPod(const char*& cur, const char* end, T& value) {
    if (static_cast<size_t>(end - cur) < sizeof(T)) {
        return false;
    }
    std::memcpy(&value, cur, sizeof(T));
    cur += sizeof(T);
    return true;
}

std::string SerializeResult(const ConvexDecompositionResult& result) {
    std::string out;
    const uint32_t piece_count = static_cast<uint32_t>(result.pieces.size());
    AppendPod(out, piece_count);
    for (const auto& piece : result.pieces) {
        const uint32_t vbytes = static_cast<uint32_t>(piece.vertices.size());
        const uint32_t ibytes = static_cast<uint32_t>(piece.indices.size());
        AppendPod(out, vbytes);
        AppendPod(out, ibytes);
        AppendPod(out, piece.volume);
        for (int i = 0; i < 3; ++i) AppendPod(out, piece.aabb_min[i]);
        for (int i = 0; i < 3; ++i) AppendPod(out, piece.aabb_max[i]);
        if (vbytes) out.append(reinterpret_cast<const char*>(piece.vertices.data()),
                               vbytes * sizeof(float));
        if (ibytes) out.append(reinterpret_cast<const char*>(piece.indices.data()),
                               ibytes * sizeof(uint32_t));
    }
    return out;
}

bool DeserializeResult(const std::string& data, ConvexDecompositionResult& result) {
    const char* cur = data.data();
    const char* end = data.data() + data.size();
    uint32_t piece_count = 0;
    if (!ReadPod(cur, end, piece_count)) return false;
    result.pieces.clear();
    result.pieces.reserve(piece_count);
    for (uint32_t p = 0; p < piece_count; ++p) {
        ConvexPiece piece;
        uint32_t vcount = 0, icount = 0;
        if (!ReadPod(cur, end, vcount)) return false;
        if (!ReadPod(cur, end, icount)) return false;
        if (!ReadPod(cur, end, piece.volume)) return false;
        for (int i = 0; i < 3; ++i) if (!ReadPod(cur, end, piece.aabb_min[i])) return false;
        for (int i = 0; i < 3; ++i) if (!ReadPod(cur, end, piece.aabb_max[i])) return false;
        if (static_cast<size_t>(end - cur) < vcount * sizeof(float)) return false;
        piece.vertices.resize(vcount);
        if (vcount) std::memcpy(piece.vertices.data(), cur, vcount * sizeof(float));
        cur += vcount * sizeof(float);
        if (static_cast<size_t>(end - cur) < icount * sizeof(uint32_t)) return false;
        piece.indices.resize(icount);
        if (icount) std::memcpy(piece.indices.data(), cur, icount * sizeof(uint32_t));
        cur += icount * sizeof(uint32_t);
        result.pieces.push_back(std::move(piece));
    }
    result.succeeded = true;
    return cur == end;
}

// --- In-process memo --------------------------------------------------------
std::mutex& MemoMutex() {
    static std::mutex m;
    return m;
}

std::map<std::string, ConvexDecompositionResult>& Memo() {
    static std::map<std::string, ConvexDecompositionResult> memo;
    return memo;
}

} // namespace

ConvexDecompositionResult DecomposeMesh(const float* vertices, uint32_t vertex_count,
                                        const uint32_t* indices, uint32_t triangle_count,
                                        const ConvexDecompositionParams& params) {
    ConvexDecompositionResult result;

    if (vertices == nullptr || indices == nullptr || vertex_count == 0 ||
        triangle_count == 0) {
        result.succeeded = false;
        result.error_message = "DecomposeMesh: empty or null mesh input";
        return result;
    }

    VHACD::IVHACD* vhacd = VHACD::CreateVHACD();   // synchronous (blocking) impl
    if (vhacd == nullptr) {
        result.succeeded = false;
        result.error_message = "DecomposeMesh: failed to create V-HACD instance";
        return result;
    }

    const VHACD::IVHACD::Parameters p = MakeVhacdParameters(params);
    const bool started =
        vhacd->Compute(vertices, vertex_count, indices, triangle_count, p);
    if (!started) {
        vhacd->Release();
        result.succeeded = false;
        result.error_message = "DecomposeMesh: V-HACD Compute() failed to start";
        return result;
    }

    const uint32_t hull_count = vhacd->GetNConvexHulls();
    result.pieces.reserve(hull_count);
    for (uint32_t h = 0; h < hull_count; ++h) {
        VHACD::IVHACD::ConvexHull hull;
        if (!vhacd->GetConvexHull(h, hull)) {
            continue;
        }
        ConvexPiece piece;
        piece.vertices.reserve(hull.m_points.size() * 3);
        for (const VHACD::Vertex& vtx : hull.m_points) {
            piece.vertices.push_back(static_cast<float>(vtx.mX));
            piece.vertices.push_back(static_cast<float>(vtx.mY));
            piece.vertices.push_back(static_cast<float>(vtx.mZ));
        }
        piece.indices.reserve(hull.m_triangles.size() * 3);
        for (const VHACD::Triangle& tri : hull.m_triangles) {
            piece.indices.push_back(tri.mI0);
            piece.indices.push_back(tri.mI1);
            piece.indices.push_back(tri.mI2);
        }
        piece.volume = static_cast<float>(hull.m_volume);
        piece.aabb_min[0] = static_cast<float>(hull.mBmin.GetX());
        piece.aabb_min[1] = static_cast<float>(hull.mBmin.GetY());
        piece.aabb_min[2] = static_cast<float>(hull.mBmin.GetZ());
        piece.aabb_max[0] = static_cast<float>(hull.mBmax.GetX());
        piece.aabb_max[1] = static_cast<float>(hull.mBmax.GetY());
        piece.aabb_max[2] = static_cast<float>(hull.mBmax.GetZ());

        CanonicalizePiece(piece);
        result.pieces.push_back(std::move(piece));
    }

    vhacd->Clean();
    vhacd->Release();

    // Canonical piece order: removes any run-to-run reordering of the hull list.
    std::sort(result.pieces.begin(), result.pieces.end(), PieceLess);

    result.succeeded = !result.pieces.empty();
    if (!result.succeeded) {
        result.error_message = "DecomposeMesh: V-HACD produced no convex hulls";
    }
    return result;
}

std::string ComputeCacheKey(const float* vertices, uint32_t vertex_count,
                            const uint32_t* indices, uint32_t triangle_count,
                            const ConvexDecompositionParams& params) {
    nuka::sha256::Hasher hasher;
    // Domain tag so the key space can evolve without colliding with future
    // serializations.
    static const char kTag[] = "nuka.vhacd.v1";
    hasher.Update(kTag, sizeof(kTag));
    hasher.Update(&vertex_count, sizeof(vertex_count));
    hasher.Update(&triangle_count, sizeof(triangle_count));
    if (vertices && vertex_count) {
        hasher.Update(vertices, static_cast<size_t>(vertex_count) * 3u * sizeof(float));
    }
    if (indices && triangle_count) {
        hasher.Update(indices, static_cast<size_t>(triangle_count) * 3u * sizeof(uint32_t));
    }
    // Serialize params field-by-field (avoid hashing struct padding).
    hasher.Update(&params.max_pieces, sizeof(params.max_pieces));
    hasher.Update(&params.voxel_resolution, sizeof(params.voxel_resolution));
    hasher.Update(&params.concavity_threshold, sizeof(params.concavity_threshold));
    hasher.Update(&params.max_vertices_per_piece, sizeof(params.max_vertices_per_piece));
    const uint8_t project = params.project_hull_vertices ? 1u : 0u;
    hasher.Update(&project, sizeof(project));
    return nuka::sha256::ToHex(hasher.Final());
}

ConvexDecompositionResult DecomposeMeshCached(const float* vertices, uint32_t vertex_count,
                                              const uint32_t* indices, uint32_t triangle_count,
                                              const ConvexDecompositionParams& params,
                                              const std::string& cache_dir,
                                              bool* was_hit) {
    if (was_hit) *was_hit = false;
    const std::string key =
        ComputeCacheKey(vertices, vertex_count, indices, triangle_count, params);

    // 1) In-process memo.
    {
        std::lock_guard<std::mutex> lock(MemoMutex());
        const auto it = Memo().find(key);
        if (it != Memo().end()) {
            if (was_hit) *was_hit = true;
            return it->second;
        }
    }

    // 2) On-disk cache.
    std::filesystem::path cache_file;
    if (!cache_dir.empty()) {
        cache_file = std::filesystem::path(cache_dir) / (key + ".nukacvx");
        std::error_code ec;
        if (std::filesystem::exists(cache_file, ec)) {
            std::ifstream in(cache_file, std::ios::binary);
            if (in) {
                std::string data((std::istreambuf_iterator<char>(in)),
                                 std::istreambuf_iterator<char>());
                ConvexDecompositionResult cached;
                if (DeserializeResult(data, cached)) {
                    {
                        std::lock_guard<std::mutex> lock(MemoMutex());
                        Memo()[key] = cached;
                    }
                    if (was_hit) *was_hit = true;
                    return cached;
                }
            }
        }
    }

    // 3) Cold path: run the (pure) decomposition and store.
    ConvexDecompositionResult result =
        DecomposeMesh(vertices, vertex_count, indices, triangle_count, params);

    if (result.succeeded) {
        {
            std::lock_guard<std::mutex> lock(MemoMutex());
            Memo()[key] = result;
        }
        if (!cache_dir.empty()) {
            std::error_code ec;
            std::filesystem::create_directories(cache_dir, ec);
            std::ofstream out(cache_file, std::ios::binary | std::ios::trunc);
            if (out) {
                const std::string blob = SerializeResult(result);
                out.write(blob.data(), static_cast<std::streamsize>(blob.size()));
            }
        }
    }
    return result;
}

void ClearDecompositionMemo() {
    std::lock_guard<std::mutex> lock(MemoMutex());
    Memo().clear();
}

} // namespace nuka::import::cooker
