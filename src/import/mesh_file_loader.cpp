// ---------------------------------------------------------------------------
// nuka::import - Self-contained mesh-file loader implementation (STL / OBJ)
// ---------------------------------------------------------------------------

#include "import/mesh_file_loader.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace nuka::import {

namespace {

// Read an entire file into a byte buffer. Throws if it cannot be opened.
std::vector<char> ReadAllBytes(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        throw std::runtime_error("mesh-loader: cannot open file: " + path);
    }
    const std::streamoff size = in.tellg();
    if (size < 0) {
        throw std::runtime_error("mesh-loader: cannot size file: " + path);
    }
    std::vector<char> buf(static_cast<std::size_t>(size));
    in.seekg(0);
    if (size > 0) {
        in.read(buf.data(), size);
        if (!in) {
            throw std::runtime_error("mesh-loader: short read on file: " + path);
        }
    }
    return buf;
}

// Read a little-endian uint32 from a byte buffer at the given offset (via
// memcpy to avoid alignment / strict-aliasing issues). x86_64 is LE.
std::uint32_t ReadU32LE(const std::vector<char>& buf, std::size_t off) {
    std::uint32_t v = 0;
    std::memcpy(&v, buf.data() + off, sizeof(v));
    return v;
}

// Read a float32 from a byte buffer at the given offset.
float ReadF32(const std::vector<char>& buf, std::size_t off) {
    float v = 0.f;
    std::memcpy(&v, buf.data() + off, sizeof(v));
    return v;
}

// ---- Binary STL ----------------------------------------------------------
// Layout: 80-byte header + uint32 count + count * 50 bytes
//   (12B normal + 3 * 12B vertices + 2B attribute byte count).
MeshGeometry ParseBinaryStl(const std::vector<char>& buf, std::uint32_t count) {
    MeshGeometry g;
    g.vertices.reserve(static_cast<std::size_t>(count) * 9);
    g.indices.reserve(static_cast<std::size_t>(count) * 3);

    std::uint32_t index = 0;
    std::size_t off = 84;  // skip 80B header + 4B count
    for (std::uint32_t t = 0; t < count; ++t) {
        off += 12;  // skip the per-facet normal
        for (int v = 0; v < 3; ++v) {
            g.vertices.push_back(ReadF32(buf, off + 0));
            g.vertices.push_back(ReadF32(buf, off + 4));
            g.vertices.push_back(ReadF32(buf, off + 8));
            off += 12;
            g.indices.push_back(index++);
        }
        off += 2;  // attribute byte count
    }
    return g;
}

// ---- ASCII STL -----------------------------------------------------------
// Strict parser: tokens flow `vertex x y z` (3 per facet). Throws on a
// malformed structure or zero facets so a truncated *binary* STL (which falls
// through to this path) does not silently yield empty geometry.
MeshGeometry ParseAsciiStl(const std::vector<char>& buf, const std::string& path) {
    MeshGeometry g;
    std::istringstream in(std::string(buf.data(), buf.size()));

    std::string token;
    std::uint32_t index = 0;
    std::uint32_t verts_in_facet = 0;
    bool in_facet = false;
    std::uint32_t facet_count = 0;

    while (in >> token) {
        if (token == "facet") {
            in_facet = true;
            verts_in_facet = 0;
        } else if (token == "endfacet") {
            if (verts_in_facet != 3) {
                throw std::runtime_error(
                    "mesh-loader: ASCII STL facet without 3 vertices: " + path);
            }
            in_facet = false;
            ++facet_count;
        } else if (token == "vertex") {
            float x = 0.f, y = 0.f, z = 0.f;
            if (!(in >> x >> y >> z)) {
                throw std::runtime_error(
                    "mesh-loader: ASCII STL malformed vertex: " + path);
            }
            g.vertices.push_back(x);
            g.vertices.push_back(y);
            g.vertices.push_back(z);
            g.indices.push_back(index++);
            ++verts_in_facet;
        }
        // All other tokens (solid, facet's "normal", outer, loop, endloop,
        // endsolid, numbers, names) are skipped.
    }

    if (in_facet) {
        throw std::runtime_error(
            "mesh-loader: ASCII STL truncated inside a facet: " + path);
    }
    if (facet_count == 0 || g.indices.empty()) {
        throw std::runtime_error(
            "mesh-loader: STL is empty or not valid ASCII/binary: " + path);
    }
    return g;
}

// ---- OBJ -----------------------------------------------------------------
// Parse one OBJ index component. Positive indices are 1-based; negative
// indices are relative to the declaration count active when the face appears.
int ParseObjIndex(const std::string& num, std::size_t count,
                  const std::string& path, const char* kind) {
    if (num.empty()) return -1;
    long raw = 0;
    try {
        std::size_t consumed = 0;
        raw = std::stol(num, &consumed);
        if (consumed != num.size()) {
            throw std::runtime_error("mesh-loader: OBJ non-integer " +
                                     std::string(kind) + " index '" + num + "': " + path);
        }
    } catch (const std::invalid_argument&) {
        throw std::runtime_error("mesh-loader: OBJ non-integer " +
                                 std::string(kind) + " index '" + num + "': " + path);
    } catch (const std::out_of_range&) {
        throw std::runtime_error("mesh-loader: OBJ " + std::string(kind) +
                                 " index out of range '" + num + "': " + path);
    }
    if (raw == 0) {
        throw std::runtime_error("mesh-loader: OBJ " + std::string(kind) +
                                 " index 0 is invalid: " + path);
    }
    const long zero_based = raw > 0 ? raw - 1 : static_cast<long>(count) + raw;
    if (zero_based < 0 || zero_based >= static_cast<long>(count)) {
        throw std::runtime_error("mesh-loader: OBJ " + std::string(kind) +
                                 " index out of bounds: " + path);
    }
    return static_cast<int>(zero_based);
}

struct ObjFaceVertex {
    int vertex = -1;
    int texcoord = -1;
    int normal = -1;
};

ObjFaceVertex ParseObjFaceVertex(const std::string& token,
                                 std::size_t vertex_count,
                                 std::size_t texcoord_count,
                                 std::size_t normal_count,
                                 const std::string& path) {
    ObjFaceVertex out;
    const std::size_t first = token.find('/');
    out.vertex = ParseObjIndex(
        first == std::string::npos ? token : token.substr(0, first), vertex_count, path, "vertex");
    if (first == std::string::npos) return out;

    const std::size_t second = token.find('/', first + 1);
    const std::string uv = token.substr(first + 1,
                                       second == std::string::npos ? std::string::npos
                                                                    : second - first - 1);
    out.texcoord = ParseObjIndex(uv, texcoord_count, path, "texture");
    if (second != std::string::npos) {
        out.normal = ParseObjIndex(token.substr(second + 1), normal_count, path, "normal");
    }
    return out;
}


} // namespace

MeshGeometry LoadStl(const std::string& path) {
    const std::vector<char> buf = ReadAllBytes(path);

    // Binary detection: size == 84 + 50 * count (count = LE uint32 @ offset 80).
    // The "solid" prefix is intentionally NOT used as the discriminator.
    if (buf.size() >= 84) {
        const std::uint32_t count = ReadU32LE(buf, 80);
        const std::uint64_t expected =
            84ull + 50ull * static_cast<std::uint64_t>(count);
        if (expected == static_cast<std::uint64_t>(buf.size())) {
            return ParseBinaryStl(buf, count);
        }
    }
    // Not a well-formed binary STL (including truncated binaries) -> ASCII.
    // The strict ASCII parser throws if the content is not valid ASCII STL.
    return ParseAsciiStl(buf, path);
}

MeshGeometry LoadObj(const std::string& path) {
    const std::vector<char> buf = ReadAllBytes(path);
    std::istringstream in(std::string(buf.data(), buf.size()));

    std::vector<float> positions;
    std::vector<float> texcoords;
    std::vector<float> normals;
    std::vector<std::vector<ObjFaceVertex>> faces;
    std::string line;
    bool complete_face_uvs = true;
    bool saw_face = false;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::istringstream ls(line);
        std::string tag;
        if (!(ls >> tag)) continue;
        if (tag == "v") {
            float x = 0.f, y = 0.f, z = 0.f;
            if (!(ls >> x >> y >> z)) {
                throw std::runtime_error("mesh-loader: OBJ malformed vertex: " + path);
            }
            positions.push_back(x); positions.push_back(y); positions.push_back(z);
        } else if (tag == "vt") {
            float u = 0.f, v = 0.f;
            if (!(ls >> u >> v)) {
                throw std::runtime_error("mesh-loader: OBJ malformed texture coordinate: " + path);
            }
            texcoords.push_back(u); texcoords.push_back(v);
        } else if (tag == "vn") {
            float nx = 0.f, ny = 0.f, nz = 0.f;
            if (!(ls >> nx >> ny >> nz)) {
                throw std::runtime_error("mesh-loader: OBJ malformed normal: " + path);
            }
            normals.push_back(nx); normals.push_back(ny); normals.push_back(nz);
        } else if (tag == "f") {
            std::vector<ObjFaceVertex> face;
            std::string token;
            while (ls >> token) {
                ObjFaceVertex corner = ParseObjFaceVertex(
                    token, positions.size() / 3u, texcoords.size() / 2u,
                    normals.size() / 3u, path);
                if (corner.vertex < 0) {
                    throw std::runtime_error("mesh-loader: OBJ face missing vertex index: " + path);
                }
                face.push_back(corner);
            }
            if (face.size() < 3) {
                throw std::runtime_error("mesh-loader: OBJ face with < 3 vertices: " + path);
            }
            saw_face = true;
            for (const ObjFaceVertex& corner : face) {
                complete_face_uvs = complete_face_uvs && corner.texcoord >= 0;
            }
            faces.push_back(std::move(face));
        }
    }

    if (positions.empty()) {
        throw std::runtime_error("mesh-loader: OBJ has no vertices: " + path);
    }

    MeshGeometry g;
    if (!saw_face || !complete_face_uvs) {
        // Preserve the legacy no-UV representation exactly: positions and
        // declaration-order normals remain shared by the original indices.
        g.vertices = std::move(positions);
        g.normals = std::move(normals);
        for (const auto& face : faces) {
            for (std::size_t i = 1; i + 1 < face.size(); ++i) {
                g.indices.push_back(static_cast<std::uint32_t>(face[0].vertex));
                g.indices.push_back(static_cast<std::uint32_t>(face[i].vertex));
                g.indices.push_back(static_cast<std::uint32_t>(face[i + 1].vertex));
            }
        }
    } else {
        // OBJ stores position, UV, and normal indices independently. Expand
        // face corners so UV seams cannot corrupt the shared position stream.
        bool complete_normals = !normals.empty();
        auto append_corner = [&](const ObjFaceVertex& corner) -> std::uint32_t {
            const std::size_t v = static_cast<std::size_t>(corner.vertex);
            g.vertices.push_back(positions[v * 3u]);
            g.vertices.push_back(positions[v * 3u + 1u]);
            g.vertices.push_back(positions[v * 3u + 2u]);
            if (corner.texcoord >= 0) {
                const std::size_t uv = static_cast<std::size_t>(corner.texcoord);
                g.uvs.push_back(texcoords[uv * 2u]);
                g.uvs.push_back(texcoords[uv * 2u + 1u]);
            } else {
                g.uvs.push_back(0.0f);
                g.uvs.push_back(0.0f);
            }
            if (corner.normal >= 0) {
                const std::size_t n = static_cast<std::size_t>(corner.normal);
                g.normals.push_back(normals[n * 3u]);
                g.normals.push_back(normals[n * 3u + 1u]);
                g.normals.push_back(normals[n * 3u + 2u]);
            } else {
                complete_normals = false;
            }
            return static_cast<std::uint32_t>(g.vertices.size() / 3u - 1u);
        };
        for (const auto& face : faces) {
            for (std::size_t i = 1; i + 1 < face.size(); ++i) {
                g.indices.push_back(append_corner(face[0]));
                g.indices.push_back(append_corner(face[i]));
                g.indices.push_back(append_corner(face[i + 1]));
            }
        }
        if (!complete_normals) g.normals.clear();
    }

    if (g.normals.size() != g.vertices.size()) g.normals.clear();
    return g;
}

MeshGeometry LoadMeshFile(const std::string& path) {
    std::string ext = std::filesystem::path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (ext == ".stl") {
        return LoadStl(path);
    }
    if (ext == ".obj") {
        return LoadObj(path);
    }
    throw std::runtime_error("mesh-loader: unsupported mesh extension '" + ext +
                             "' for file: " + path);
}

} // namespace nuka::import
