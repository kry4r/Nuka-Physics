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
// Parse the leading integer of an OBJ face token ("12", "12/3", "12//4",
// "12/3/4", or negative). Returns the 0-based index resolved against the
// current vertex count. Throws on a malformed / out-of-range reference.
std::uint32_t ParseObjFaceIndex(const std::string& tok,
                                std::size_t vertex_count,
                                const std::string& path) {
    // Take the substring up to the first '/'.
    const std::size_t slash = tok.find('/');
    const std::string num = (slash == std::string::npos) ? tok : tok.substr(0, slash);
    if (num.empty()) {
        throw std::runtime_error("mesh-loader: OBJ face missing vertex index: " + path);
    }
    long raw = 0;
    try {
        std::size_t consumed = 0;
        raw = std::stol(num, &consumed);
        if (consumed != num.size()) {
            throw std::runtime_error("mesh-loader: OBJ non-integer face index '" +
                                     num + "': " + path);
        }
    } catch (const std::invalid_argument&) {
        throw std::runtime_error("mesh-loader: OBJ non-integer face index '" +
                                 num + "': " + path);
    } catch (const std::out_of_range&) {
        throw std::runtime_error("mesh-loader: OBJ face index out of range '" +
                                 num + "': " + path);
    }

    long zero_based = 0;
    if (raw > 0) {
        zero_based = raw - 1;                       // 1-based -> 0-based
    } else if (raw < 0) {
        zero_based = static_cast<long>(vertex_count) + raw;  // relative
    } else {
        throw std::runtime_error("mesh-loader: OBJ face index 0 is invalid: " + path);
    }

    if (zero_based < 0 || zero_based >= static_cast<long>(vertex_count)) {
        throw std::runtime_error("mesh-loader: OBJ face index out of bounds: " + path);
    }
    return static_cast<std::uint32_t>(zero_based);
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

    MeshGeometry g;
    std::string line;
    while (std::getline(in, line)) {
        // Strip a trailing CR (CRLF files).
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        std::istringstream ls(line);
        std::string tag;
        if (!(ls >> tag)) {
            continue;  // blank line
        }
        if (tag == "v") {
            float x = 0.f, y = 0.f, z = 0.f;
            if (!(ls >> x >> y >> z)) {
                throw std::runtime_error("mesh-loader: OBJ malformed vertex: " + path);
            }
            g.vertices.push_back(x);
            g.vertices.push_back(y);
            g.vertices.push_back(z);
        } else if (tag == "vn") {
            // Authored per-vertex normal (x,y,z triple, declaration order). Most
            // exported OBJs declare one vn per v with matching indices (a/t/n with
            // n==v), so file-order vn aligns 1:1 with v; faces are still indexed by
            // the vertex slot. A vn-less OBJ leaves g.normals empty (zero-fill).
            float nx = 0.f, ny = 0.f, nz = 0.f;
            if (!(ls >> nx >> ny >> nz)) {
                throw std::runtime_error("mesh-loader: OBJ malformed normal: " + path);
            }
            g.normals.push_back(nx);
            g.normals.push_back(ny);
            g.normals.push_back(nz);
        } else if (tag == "f") {
            // Collect all face vertex tokens, then fan-triangulate.
            std::vector<std::uint32_t> face;
            const std::size_t vc = g.vertices.size() / 3;
            std::string ftok;
            while (ls >> ftok) {
                face.push_back(ParseObjFaceIndex(ftok, vc, path));
            }
            if (face.size() < 3) {
                throw std::runtime_error("mesh-loader: OBJ face with < 3 vertices: " +
                                         path);
            }
            for (std::size_t i = 1; i + 1 < face.size(); ++i) {
                g.indices.push_back(face[0]);
                g.indices.push_back(face[i]);
                g.indices.push_back(face[i + 1]);
            }
        }
        // vt / mtllib / usemtl / g / o / s / # and others: ignored (vn handled).
    }

    if (g.vertices.empty()) {
        throw std::runtime_error("mesh-loader: OBJ has no vertices: " + path);
    }
    // Authored normals carry through only when they align 1:1 with vertices (one
    // vn per v in declaration order, the common export). Any mismatch -> drop
    // them so EncodeMesh zero-fills and the render side synthesizes.
    if (g.normals.size() != g.vertices.size()) {
        g.normals.clear();
    }
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
