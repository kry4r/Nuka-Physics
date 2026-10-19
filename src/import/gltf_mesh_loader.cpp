// ---------------------------------------------------------------------------
// nuka::import - glTF 2.0 mesh loading through the vendored cgltf reader
// ---------------------------------------------------------------------------

#define CGLTF_IMPLEMENTATION
#include "cgltf.h"

#include "import/mesh_file_loader.hpp"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>

namespace nuka::import {

namespace {

const char* ResultText(cgltf_result r) {
    switch (r) {
        case cgltf_result_data_too_short: return "data too short";
        case cgltf_result_unknown_format: return "unknown format";
        case cgltf_result_invalid_json: return "invalid JSON";
        case cgltf_result_invalid_gltf: return "invalid glTF";
        case cgltf_result_invalid_options: return "invalid options";
        case cgltf_result_file_not_found: return "file not found";
        case cgltf_result_io_error: return "I/O error";
        case cgltf_result_out_of_memory: return "out of memory";
        case cgltf_result_legacy_gltf: return "glTF 1.0 is not supported";
        default: return "unreadable file";
    }
}

void Check(cgltf_result r, const std::string& path) {
    if (r != cgltf_result_success)
        throw std::runtime_error("mesh-loader: glTF " + std::string(ResultText(r)) + ": " + path);
}

// glTF is +Y up and the scene +Z up: (x, y, z) -> (x, -z, y).
void PushZUp(std::vector<float>& out, float x, float y, float z) {
    out.push_back(x);
    out.push_back(-z);
    out.push_back(y);
}

// Where a texture's image lives: an absolute file, or "file#offset+length" for bytes inside the .glb
// or a .bin. Data URIs and maps on a second UV set give "".
std::string ImageRef(const cgltf_texture_view& view, const cgltf_data& data, const std::string& path) {
    if (view.texture == nullptr || view.texture->image == nullptr || view.texcoord != 0) return {};
    const cgltf_image& image = *view.texture->image;
    const std::filesystem::path file = std::filesystem::absolute(path).lexically_normal();
    const auto beside = [&](const char* uri) -> std::string {
        if (uri == nullptr || std::strncmp(uri, "data:", 5) == 0) return {};
        std::string name(uri);
        cgltf_decode_uri(name.data());
        name.resize(std::strlen(name.c_str()));
        return (file.parent_path() / name).lexically_normal().string();
    };
    const cgltf_buffer_view* bytes = image.buffer_view;
    if (bytes == nullptr) return beside(image.uri);
    std::string holder = file.string();
    cgltf_size offset = bytes->offset;
    if (bytes->buffer->uri != nullptr) {
        holder = beside(bytes->buffer->uri);
    } else {
        const char* glb = static_cast<const char*>(data.file_data);
        offset += static_cast<cgltf_size>(static_cast<const char*>(data.bin) - glb);
    }
    if (holder.empty()) return {};
    return holder + "#" + std::to_string(offset) + "+" + std::to_string(bytes->size);
}

// The part's material: metallic-roughness factors and maps (specular-glossiness keeps its diffuse
// and turns glossiness into roughness), the normal map and the emissive colour.
void ReadMaterial(const cgltf_material& mat, const cgltf_data& data, const std::string& path, MeshPart& part) {
    part.material = mat.name != nullptr ? mat.name : "material_" + std::to_string(cgltf_material_index(&data, &mat));
    part.has_color = true;
    if (mat.has_pbr_specular_glossiness) {
        const cgltf_pbr_specular_glossiness& sg = mat.pbr_specular_glossiness;
        for (int k = 0; k < 4; ++k) part.color[k] = sg.diffuse_factor[k];
        part.metallic = 0.0f;
        part.roughness = 1.0f - sg.glossiness_factor;
        part.albedo_map = ImageRef(sg.diffuse_texture, data, path);
    } else {
        const cgltf_pbr_metallic_roughness& pbr = mat.pbr_metallic_roughness;
        for (int k = 0; k < 4; ++k) part.color[k] = pbr.base_color_factor[k];
        part.metallic = pbr.metallic_factor;
        part.roughness = pbr.roughness_factor;
        part.albedo_map = ImageRef(pbr.base_color_texture, data, path);
        part.roughness_map = part.metallic_map = ImageRef(pbr.metallic_roughness_texture, data, path);
        part.roughness_channel = 1;
        part.metallic_channel = 2;
    }
    part.normal_map = ImageRef(mat.normal_texture, data, path);
    const float strength = mat.has_emissive_strength ? mat.emissive_strength.emissive_strength : 1.0f;
    for (int k = 0; k < 3; ++k) part.emissive[k] = mat.emissive_factor[k] * strength;
    part.emissive_map = ImageRef(mat.emissive_texture, data, path);
}

// Appends one primitive under the column-major world matrix `m` as a new part.
void AppendPrimitive(const cgltf_primitive& prim, const float m[16], MeshGeometry& g, bool& all_normals,
                     bool& any_uv, const cgltf_data& data, const std::string& path) {
    if (prim.has_draco_mesh_compression)
        throw std::runtime_error("mesh-loader: glTF Draco-compressed meshes are not supported: " + path);
    if (prim.type != cgltf_primitive_type_triangles && prim.type != cgltf_primitive_type_triangle_strip &&
        prim.type != cgltf_primitive_type_triangle_fan)
        return;
    const cgltf_accessor* position = nullptr;
    const cgltf_accessor* normal = nullptr;
    const cgltf_accessor* uv = nullptr;
    for (cgltf_size i = 0; i < prim.attributes_count; ++i) {
        const cgltf_attribute& a = prim.attributes[i];
        if (a.type == cgltf_attribute_type_position) position = a.data;
        if (a.type == cgltf_attribute_type_normal) normal = a.data;
        if (a.type == cgltf_attribute_type_texcoord && a.index == 0) uv = a.data;
    }
    if (position == nullptr) throw std::runtime_error("mesh-loader: glTF primitive has no POSITION: " + path);

    // Normals take the inverse transpose of the linear part: the columns' cross products, signed by det.
    const float c0[3] = {m[0], m[1], m[2]}, c1[3] = {m[4], m[5], m[6]}, c2[3] = {m[8], m[9], m[10]};
    const auto cross = [](const float* a, const float* b, float* o) {
        o[0] = a[1] * b[2] - a[2] * b[1];
        o[1] = a[2] * b[0] - a[0] * b[2];
        o[2] = a[0] * b[1] - a[1] * b[0];
    };
    float n0[3], n1[3], n2[3];
    cross(c1, c2, n0);
    cross(c2, c0, n1);
    cross(c0, c1, n2);
    const float det = c0[0] * n0[0] + c0[1] * n0[1] + c0[2] * n0[2];
    const float sign = det < 0.0f ? -1.0f : 1.0f;

    const std::uint32_t base = static_cast<std::uint32_t>(g.VertexCount());
    for (cgltf_size i = 0; i < position->count; ++i) {
        float p[3] = {0.0f, 0.0f, 0.0f};
        cgltf_accessor_read_float(position, i, p, 3);
        PushZUp(g.vertices, m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12],
                m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13], m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14]);
        float n[3] = {0.0f, 0.0f, 0.0f};
        if (normal != nullptr) {
            cgltf_accessor_read_float(normal, i, n, 3);
            float w[3];
            for (int k = 0; k < 3; ++k) w[k] = sign * (n0[k] * n[0] + n1[k] * n[1] + n2[k] * n[2]);
            const float len = std::sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
            for (int k = 0; k < 3; ++k) n[k] = len > 0.0f ? w[k] / len : 0.0f;
        }
        PushZUp(g.normals, n[0], n[1], n[2]);
        float t[2] = {0.0f, 0.0f};
        if (uv != nullptr) cgltf_accessor_read_float(uv, i, t, 2);
        // Stored UVs run v up as OBJ and COLLADA do; glTF's v runs down the image.
        g.uvs.push_back(t[0]);
        g.uvs.push_back(1.0f - t[1]);
    }
    all_normals = all_normals && normal != nullptr;
    any_uv = any_uv || uv != nullptr;

    const cgltf_size corners = prim.indices != nullptr ? prim.indices->count : position->count;
    const auto corner = [&](cgltf_size k) {
        const cgltf_size v = prim.indices != nullptr ? cgltf_accessor_read_index(prim.indices, k) : k;
        return base + static_cast<std::uint32_t>(v);
    };
    MeshPart part;
    part.first_index = static_cast<std::uint32_t>(g.indices.size());
    const auto triangle = [&](cgltf_size a, cgltf_size b, cgltf_size c) {
        g.indices.push_back(corner(a));
        g.indices.push_back(corner(det < 0.0f ? c : b));
        g.indices.push_back(corner(det < 0.0f ? b : c));
    };
    if (prim.type == cgltf_primitive_type_triangles)
        for (cgltf_size k = 0; k + 2 < corners; k += 3) triangle(k, k + 1, k + 2);
    else if (prim.type == cgltf_primitive_type_triangle_strip)
        for (cgltf_size k = 0; k + 2 < corners; ++k) k % 2 == 0 ? triangle(k, k + 1, k + 2) : triangle(k + 1, k, k + 2);
    else
        for (cgltf_size k = 1; k + 1 < corners; ++k) triangle(0, k, k + 1);
    part.index_count = static_cast<std::uint32_t>(g.indices.size()) - part.first_index;
    if (prim.material != nullptr) ReadMaterial(*prim.material, data, path, part);
    g.parts.push_back(std::move(part));
}

void AppendNode(const cgltf_node& node, MeshGeometry& g, bool& all_normals, bool& any_uv, const cgltf_data& data,
                const std::string& path) {
    if (node.mesh != nullptr) {
        float m[16];
        cgltf_node_transform_world(&node, m);
        for (cgltf_size i = 0; i < node.mesh->primitives_count; ++i)
            AppendPrimitive(node.mesh->primitives[i], m, g, all_normals, any_uv, data, path);
    }
    for (cgltf_size i = 0; i < node.children_count; ++i)
        AppendNode(*node.children[i], g, all_normals, any_uv, data, path);
}

}  // namespace

MeshGeometry LoadGltf(const std::string& path) {
    cgltf_options options{};
    cgltf_data* raw = nullptr;
    Check(cgltf_parse_file(&options, path.c_str(), &raw), path);
    const std::unique_ptr<cgltf_data, void (*)(cgltf_data*)> data(raw, cgltf_free);
    Check(cgltf_load_buffers(&options, data.get(), path.c_str()), path);
    Check(cgltf_validate(data.get()), path);

    MeshGeometry g;
    bool all_normals = true, any_uv = false;
    const cgltf_scene* scene = data->scene != nullptr ? data->scene : data->scenes_count > 0 ? data->scenes : nullptr;
    if (scene != nullptr) {
        for (cgltf_size i = 0; i < scene->nodes_count; ++i)
            AppendNode(*scene->nodes[i], g, all_normals, any_uv, *data, path);
    } else {
        for (cgltf_size i = 0; i < data->nodes_count; ++i)
            if (data->nodes[i].parent == nullptr) AppendNode(data->nodes[i], g, all_normals, any_uv, *data, path);
    }
    if (g.indices.empty()) throw std::runtime_error("mesh-loader: glTF has no triangles: " + path);
    if (!all_normals) g.normals.clear();
    if (!any_uv) g.uvs.clear();
    return g;
}

}  // namespace nuka::import
