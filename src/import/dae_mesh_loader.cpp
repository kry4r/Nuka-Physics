// ---------------------------------------------------------------------------
// nuka::import - COLLADA mesh loading on tinyxml2
// ---------------------------------------------------------------------------

#include "import/mesh_file_loader.hpp"

#include <tinyxml2.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace nuka::import {

namespace {

using tinyxml2::XMLElement;
using Mat4 = std::array<double, 16>;  // row-major, the order <matrix> is written in
using Bindings = std::unordered_map<std::string, const XMLElement*>;

constexpr Mat4 kIdentity{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};

Mat4 Mul(const Mat4& a, const Mat4& b) {
    Mat4 c{};
    for (int r = 0; r < 4; ++r)
        for (int k = 0; k < 4; ++k)
            for (int j = 0; j < 4; ++j) c[r * 4 + j] += a[r * 4 + k] * b[k * 4 + j];
    return c;
}

void IndexIds(const XMLElement* e, std::unordered_map<std::string, const XMLElement*>& ids) {
    for (; e != nullptr; e = e->NextSiblingElement()) {
        if (const char* id = e->Attribute("id")) ids.emplace(id, e);
        IndexIds(e->FirstChildElement(), ids);
    }
}

struct Source {
    std::vector<double> values;
    std::size_t stride = 1;
    std::size_t offset = 0;
};

// Where one corner attribute comes from: its slot in a <p> tuple and its <source>.
struct Channel {
    std::size_t offset = 0;
    const Source* source = nullptr;
};

struct Reader {
    std::string path;
    std::unordered_map<std::string, const XMLElement*> ids;
    std::unordered_map<const XMLElement*, Source> sources;
    double unit = 1.0;
    MeshGeometry g;
    bool all_normals = true;
    bool any_uv = false;

    [[noreturn]] void Fail(const std::string& what) const {
        throw std::runtime_error("mesh-loader: COLLADA " + what + ": " + path);
    }

    const XMLElement* Ref(const char* url) const {
        if (url == nullptr) return nullptr;
        const auto it = ids.find(url[0] == '#' ? url + 1 : url);
        return it != ids.end() ? it->second : nullptr;
    }

    std::vector<double> Numbers(const XMLElement* e) const {
        std::vector<double> out;
        const char* s = e != nullptr && e->GetText() != nullptr ? e->GetText() : "";
        for (char* end = nullptr;; s = end) {
            while (std::isspace(static_cast<unsigned char>(*s))) ++s;
            if (*s == '\0') break;
            out.push_back(std::strtod(s, &end));
            if (end == s) Fail("<" + std::string(e->Name()) + "> holds a non-number");
        }
        return out;
    }

    const Source& SourceOf(const XMLElement* e) {
        if (e == nullptr) Fail("an input names no source");
        const auto found = sources.find(e);
        if (found != sources.end()) return found->second;
        Source s;
        s.values = Numbers(e->FirstChildElement("float_array"));
        const XMLElement* common = e->FirstChildElement("technique_common");
        if (const XMLElement* accessor = common != nullptr ? common->FirstChildElement("accessor") : nullptr) {
            s.stride = std::max(1u, accessor->UnsignedAttribute("stride", 1u));
            s.offset = accessor->UnsignedAttribute("offset", 0u);
        }
        return sources.emplace(e, std::move(s)).first->second;
    }

    // Components [0, n) of element `i` of the channel's source.
    void Read(const Channel& ch, std::int64_t i, std::size_t n, double* out) const {
        const std::size_t at = ch.source->offset + static_cast<std::size_t>(i) * ch.source->stride;
        if (i < 0 || n > ch.source->stride || at + n > ch.source->values.size()) Fail("an index runs past its source");
        for (std::size_t k = 0; k < n; ++k) out[k] = ch.source->values[at + k];
    }

    static void Push(std::vector<float>& out, double x, double y, double z) {
        for (const double c : {x, y, z}) out.push_back(static_cast<float>(c));
    }

    void AppendPrimitive(const XMLElement* prim, const Mat4& m, const double cof[9], bool flip, const Bindings& bound);
    void AppendGeometry(const XMLElement* geometry, const Mat4& m, const Bindings& bound);
    Bindings BindingsOf(const XMLElement* instance) const;
    void Walk(const XMLElement* node, Mat4 m);
};

// A newparam of the effect (or of its profile_COMMON) by sid.
const XMLElement* EffectParam(const XMLElement* effect, const char* sid) {
    if (sid == nullptr) return nullptr;
    for (const XMLElement* scope : {effect, effect->FirstChildElement("profile_COMMON")})
        for (const XMLElement* p = scope != nullptr ? scope->FirstChildElement("newparam") : nullptr; p;
             p = p->NextSiblingElement("newparam"))
            if (const char* name = p->Attribute("sid"); name != nullptr && std::string(name) == sid) return p;
    return nullptr;
}

// The image file a <texture texture="..."/> names: through its sampler and surface newparams (1.4), the
// sampler's <instance_image> (1.5), or straight to an <image> id. Relative paths sit beside the file.
std::string TexturePath(const Reader& r, const XMLElement* effect, const char* sampler) {
    const XMLElement* image = r.Ref(sampler);
    const XMLElement* param = EffectParam(effect, sampler);
    const XMLElement* sampler2d = param != nullptr ? param->FirstChildElement("sampler2D") : nullptr;
    if (sampler2d != nullptr) {
        const XMLElement* source = sampler2d->FirstChildElement("source");
        const XMLElement* surface_param = EffectParam(effect, source != nullptr ? source->GetText() : nullptr);
        const XMLElement* surface = surface_param != nullptr ? surface_param->FirstChildElement("surface") : nullptr;
        const XMLElement* init = surface != nullptr ? surface->FirstChildElement("init_from") : nullptr;
        const XMLElement* instance = sampler2d->FirstChildElement("instance_image");
        image = init != nullptr       ? r.Ref(init->GetText())
                : instance != nullptr ? r.Ref(instance->Attribute("url"))
                                      : nullptr;
    }
    const XMLElement* init = image != nullptr ? image->FirstChildElement("init_from") : nullptr;
    const XMLElement* ref = init != nullptr ? init->FirstChildElement("ref") : nullptr;
    const char* uri = ref != nullptr ? ref->GetText() : init != nullptr ? init->GetText() : nullptr;
    if (uri == nullptr) return {};
    std::string file;
    for (const char* c = uri; *c != '\0'; ++c) {
        const bool hex = c[0] == '%' && std::isxdigit(static_cast<unsigned char>(c[1])) &&
                         std::isxdigit(static_cast<unsigned char>(c[2]));
        file.push_back(hex ? static_cast<char>(std::strtol(std::string(c + 1, 2).c_str(), nullptr, 16)) : *c);
        c += hex ? 2 : 0;
    }
    if (file.rfind("file://", 0) == 0) file.erase(0, file.size() > 9 && file[9] == ':' ? 8 : 7);
    const std::filesystem::path dae = std::filesystem::absolute(r.path);
    return (dae.parent_path() / file).lexically_normal().string();
}

// The material's diffuse color (emission for <constant>), or a white base under its diffuse texture.
bool ReadMaterial(const Reader& r, const XMLElement* material, MeshPart& part) {
    const XMLElement* instance = material != nullptr ? material->FirstChildElement("instance_effect") : nullptr;
    const XMLElement* effect = r.Ref(instance != nullptr ? instance->Attribute("url") : nullptr);
    const XMLElement* common = effect != nullptr ? effect->FirstChildElement("profile_COMMON") : nullptr;
    const XMLElement* technique = common != nullptr ? common->FirstChildElement("technique") : nullptr;
    for (const XMLElement* s = technique != nullptr ? technique->FirstChildElement() : nullptr; s;
         s = s->NextSiblingElement()) {
        const std::string model = s->Name();
        if (model != "phong" && model != "lambert" && model != "blinn" && model != "constant") continue;
        const XMLElement* slot = s->FirstChildElement(model == "constant" ? "emission" : "diffuse");
        const XMLElement* texture = slot != nullptr ? slot->FirstChildElement("texture") : nullptr;
        if (texture != nullptr) {
            part.albedo_map = TexturePath(r, effect, texture->Attribute("texture"));
            return !part.albedo_map.empty();
        }
        const XMLElement* color = slot != nullptr ? slot->FirstChildElement("color") : nullptr;
        if (color == nullptr) return false;
        const std::vector<double> rgba = r.Numbers(color);
        if (rgba.size() < 3u) r.Fail("a <color> needs at least 3 numbers");
        for (std::size_t k = 0; k < 4u; ++k) part.color[k] = k < rgba.size() ? static_cast<float>(rgba[k]) : 1.0f;
        return true;
    }
    return false;
}

void Reader::AppendPrimitive(const XMLElement* prim, const Mat4& m, const double cof[9], bool flip,
                             const Bindings& bound) {
    Channel pos, nrm, tex;
    std::size_t stride = 0;
    const auto bind = [&](const char* semantic, std::size_t offset, const XMLElement* source) {
        const std::string s = semantic != nullptr ? semantic : "";
        if (s == "POSITION") pos = Channel{offset, &SourceOf(source)};
        if (s == "NORMAL") nrm = Channel{offset, &SourceOf(source)};
        if (s == "TEXCOORD" && tex.source == nullptr) tex = Channel{offset, &SourceOf(source)};
    };
    for (const XMLElement* in = prim->FirstChildElement("input"); in; in = in->NextSiblingElement("input")) {
        const std::size_t offset = in->UnsignedAttribute("offset", 0u);
        stride = std::max(stride, offset + 1u);
        const XMLElement* target = Ref(in->Attribute("source"));
        if (target == nullptr) Fail("an input names no element");
        if (std::string(in->Attribute("semantic") != nullptr ? in->Attribute("semantic") : "") != "VERTEX") {
            bind(in->Attribute("semantic"), offset, target);
            continue;
        }
        for (const XMLElement* v = target->FirstChildElement("input"); v; v = v->NextSiblingElement("input"))
            bind(v->Attribute("semantic"), offset, Ref(v->Attribute("source")));
    }
    if (pos.source == nullptr) Fail("a primitive has no POSITION");
    all_normals = all_normals && nrm.source != nullptr;
    any_uv = any_uv || tex.source != nullptr;

    // One vertex per distinct (position, normal, uv) index tuple, in first-use order.
    std::map<std::array<std::int64_t, 3>, std::uint32_t> seen;
    const auto vertex = [&](const std::vector<double>& p, std::size_t c) {
        const auto index = [&](const Channel& ch) -> std::int64_t {
            if (ch.source == nullptr) return -1;
            if (c * stride + ch.offset >= p.size()) Fail("a <p> is shorter than its inputs");
            return static_cast<std::int64_t>(p[c * stride + ch.offset]);
        };
        const std::array<std::int64_t, 3> key{index(pos), index(nrm), index(tex)};
        const auto [it, fresh] = seen.emplace(key, static_cast<std::uint32_t>(g.VertexCount()));
        if (!fresh) return it->second;
        double v[3];
        Read(pos, key[0], 3u, v);
        Push(g.vertices, unit * (m[0] * v[0] + m[1] * v[1] + m[2] * v[2] + m[3]),
             unit * (m[4] * v[0] + m[5] * v[1] + m[6] * v[2] + m[7]),
             unit * (m[8] * v[0] + m[9] * v[1] + m[10] * v[2] + m[11]));
        double n[3] = {0.0, 0.0, 0.0};
        if (nrm.source != nullptr) {
            double w[3];
            Read(nrm, key[1], 3u, v);
            for (int k = 0; k < 3; ++k) w[k] = cof[k * 3] * v[0] + cof[k * 3 + 1] * v[1] + cof[k * 3 + 2] * v[2];
            const double len = std::sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
            for (int k = 0; k < 3; ++k) n[k] = len > 0.0 ? w[k] / len : 0.0;
        }
        Push(g.normals, n[0], n[1], n[2]);
        double t[2] = {0.0, 0.0};
        if (tex.source != nullptr) Read(tex, key[2], 2u, t);
        g.uvs.push_back(static_cast<float>(t[0]));
        g.uvs.push_back(static_cast<float>(t[1]));
        return it->second;
    };
    const auto triangle = [&](const std::vector<double>& p, std::size_t a, std::size_t b, std::size_t c) {
        const std::uint32_t ia = vertex(p, a);
        const std::uint32_t ib = vertex(p, b);
        const std::uint32_t ic = vertex(p, c);
        g.indices.push_back(ia);
        g.indices.push_back(flip ? ic : ib);
        g.indices.push_back(flip ? ib : ic);
    };
    const auto corners = [&](const std::vector<double>& p) {
        if (p.size() % stride != 0u) Fail("a <p> is not a whole number of corners");
        return p.size() / stride;
    };
    const auto fan = [&](const std::vector<double>& p, std::size_t first, std::size_t n) {
        for (std::size_t k = 1; k + 1 < n; ++k) triangle(p, first, first + k, first + k + 1);
    };

    MeshPart part;
    part.first_index = static_cast<std::uint32_t>(g.indices.size());
    const std::string kind = prim->Name();
    if (kind == "triangles") {
        const std::vector<double> p = Numbers(prim->FirstChildElement("p"));
        for (std::size_t c = 0; c + 2 < corners(p); c += 3) triangle(p, c, c + 1, c + 2);
    } else if (kind == "polylist") {
        const std::vector<double> p = Numbers(prim->FirstChildElement("p"));
        std::size_t first = 0;
        for (const double n : Numbers(prim->FirstChildElement("vcount"))) {
            if (first + static_cast<std::size_t>(n) > corners(p)) Fail("a <vcount> runs past its <p>");
            fan(p, first, static_cast<std::size_t>(n));
            first += static_cast<std::size_t>(n);
        }
    } else {
        // <polygons>, <tristrips> and <trifans> hold one <p> per polygon, strip or fan; <ph> wraps a holed one.
        for (const XMLElement* e = prim->FirstChildElement(); e; e = e->NextSiblingElement()) {
            const std::string tag = e->Name();
            const XMLElement* run = tag == "p" ? e : tag == "ph" ? e->FirstChildElement("p") : nullptr;
            if (run == nullptr) continue;
            const std::vector<double> p = Numbers(run);
            const std::size_t n = corners(p);
            if (kind != "tristrips") fan(p, 0, n);
            for (std::size_t k = 0; kind == "tristrips" && k + 2 < n; ++k)
                k % 2 == 0 ? triangle(p, k, k + 1, k + 2) : triangle(p, k + 1, k, k + 2);
        }
    }
    part.index_count = static_cast<std::uint32_t>(g.indices.size()) - part.first_index;

    const char* symbol = prim->Attribute("material");
    const auto binding = symbol != nullptr ? bound.find(symbol) : bound.end();
    const XMLElement* material = binding != bound.end() ? binding->second : Ref(symbol);
    const char* name = material != nullptr ? material->Attribute("name") : nullptr;
    const char* id = material != nullptr ? material->Attribute("id") : nullptr;
    part.material = name != nullptr ? name : id != nullptr ? id : symbol != nullptr ? symbol : "";
    part.has_color = ReadMaterial(*this, material, part);
    g.parts.push_back(std::move(part));
}

void Reader::AppendGeometry(const XMLElement* geometry, const Mat4& m, const Bindings& bound) {
    if (geometry == nullptr) Fail("an instance names no geometry");
    const XMLElement* mesh = geometry->FirstChildElement("mesh");
    if (mesh == nullptr) return;
    // Normals take the inverse transpose of the linear part: its cofactors, signed by the determinant.
    const double r0[3] = {m[0], m[1], m[2]}, r1[3] = {m[4], m[5], m[6]}, r2[3] = {m[8], m[9], m[10]};
    double cof[9];
    const auto cross = [](const double* a, const double* b, double* o) {
        o[0] = a[1] * b[2] - a[2] * b[1];
        o[1] = a[2] * b[0] - a[0] * b[2];
        o[2] = a[0] * b[1] - a[1] * b[0];
    };
    cross(r1, r2, cof);
    cross(r2, r0, cof + 3);
    cross(r0, r1, cof + 6);
    const double det = r0[0] * cof[0] + r0[1] * cof[1] + r0[2] * cof[2];
    if (det < 0.0)
        for (double& c : cof) c = -c;
    for (const XMLElement* prim = mesh->FirstChildElement(); prim; prim = prim->NextSiblingElement()) {
        const std::string kind = prim->Name();
        if (kind == "triangles" || kind == "polylist" || kind == "polygons" || kind == "tristrips" ||
            kind == "trifans")
            AppendPrimitive(prim, m, cof, det < 0.0, bound);
    }
}

Bindings Reader::BindingsOf(const XMLElement* instance) const {
    Bindings b;
    const XMLElement* bind = instance->FirstChildElement("bind_material");
    const XMLElement* common = bind != nullptr ? bind->FirstChildElement("technique_common") : nullptr;
    for (const XMLElement* im = common != nullptr ? common->FirstChildElement("instance_material") : nullptr; im;
         im = im->NextSiblingElement("instance_material"))
        if (const char* symbol = im->Attribute("symbol")) b[symbol] = Ref(im->Attribute("target"));
    return b;
}

void Reader::Walk(const XMLElement* node, Mat4 m) {
    if (node == nullptr) Fail("an instance names no node");
    for (const XMLElement* t = node->FirstChildElement(); t; t = t->NextSiblingElement()) {
        const std::string tag = t->Name();
        if (tag == "lookat" || tag == "skew") Fail("<" + tag + "> transforms are not supported");
        if (tag != "matrix" && tag != "translate" && tag != "rotate" && tag != "scale") continue;
        const std::vector<double> v = Numbers(t);
        if (v.size() != (tag == "matrix" ? 16u : tag == "rotate" ? 4u : 3u))
            Fail("<" + tag + "> has the wrong arity");
        Mat4 a = kIdentity;
        if (tag == "matrix") std::copy(v.begin(), v.end(), a.begin());
        if (tag == "translate") a[3] = v[0], a[7] = v[1], a[11] = v[2];
        if (tag == "scale") a[0] = v[0], a[5] = v[1], a[10] = v[2];
        if (tag == "rotate") {
            const double len = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
            if (len == 0.0) Fail("a <rotate> axis is zero");
            const double x = v[0] / len, y = v[1] / len, z = v[2] / len;
            const double c = std::cos(v[3] * 3.14159265358979323846 / 180.0);
            const double s = std::sin(v[3] * 3.14159265358979323846 / 180.0), k = 1.0 - c;
            a = Mat4{c + x * x * k,     x * y * k - z * s, x * z * k + y * s, 0,
                     y * x * k + z * s, c + y * y * k,     y * z * k - x * s, 0,
                     z * x * k - y * s, z * y * k + x * s, c + z * z * k,     0,
                     0,                 0,                 0,                 1};
        }
        m = Mul(m, a);
    }
    for (const XMLElement* c = node->FirstChildElement(); c; c = c->NextSiblingElement()) {
        const std::string tag = c->Name();
        if (tag == "node") Walk(c, m);
        if (tag == "instance_node") Walk(Ref(c->Attribute("url")), m);
        if (tag == "instance_geometry") AppendGeometry(Ref(c->Attribute("url")), m, BindingsOf(c));
        if (tag == "instance_controller") {
            const XMLElement* controller = Ref(c->Attribute("url"));
            const XMLElement* skin = controller != nullptr ? controller->FirstChildElement("skin") : nullptr;
            if (skin == nullptr) Fail("an instance_controller names no skin");
            Mat4 shape = kIdentity;
            if (const XMLElement* bsm = skin->FirstChildElement("bind_shape_matrix")) {
                const std::vector<double> v = Numbers(bsm);
                if (v.size() != 16u) Fail("<bind_shape_matrix> has the wrong arity");
                std::copy(v.begin(), v.end(), shape.begin());
            }
            AppendGeometry(Ref(skin->Attribute("source")), Mul(m, shape), BindingsOf(c));
        }
    }
}

}  // namespace

MeshGeometry LoadDae(const std::string& path) {
    tinyxml2::XMLDocument doc;
    if (doc.LoadFile(path.c_str()) != tinyxml2::XML_SUCCESS)
        throw std::runtime_error("mesh-loader: COLLADA " + std::string(doc.ErrorStr()) + ": " + path);
    Reader r;
    r.path = path;
    const XMLElement* root = doc.FirstChildElement("COLLADA");
    if (root == nullptr) r.Fail("document has no <COLLADA> root");
    IndexIds(root, r.ids);
    // <up_axis> is ignored as RViz does; robot meshes tag Y_UP while their data is already +Z up.
    if (const XMLElement* asset = root->FirstChildElement("asset"))
        if (const XMLElement* unit = asset->FirstChildElement("unit")) r.unit = unit->DoubleAttribute("meter", 1.0);
    const XMLElement* scene = root->FirstChildElement("scene");
    const XMLElement* instance = scene != nullptr ? scene->FirstChildElement("instance_visual_scene") : nullptr;
    const XMLElement* visual = r.Ref(instance != nullptr ? instance->Attribute("url") : nullptr);
    if (visual == nullptr) {
        const XMLElement* library = root->FirstChildElement("library_visual_scenes");
        visual = library != nullptr ? library->FirstChildElement("visual_scene") : nullptr;
    }
    if (visual != nullptr) {
        for (const XMLElement* node = visual->FirstChildElement("node"); node; node = node->NextSiblingElement("node"))
            r.Walk(node, kIdentity);
    } else {
        for (const XMLElement* lib = root->FirstChildElement("library_geometries"); lib;
             lib = lib->NextSiblingElement("library_geometries"))
            for (const XMLElement* geo = lib->FirstChildElement("geometry"); geo;
                 geo = geo->NextSiblingElement("geometry"))
                r.AppendGeometry(geo, kIdentity, {});
    }
    if (r.g.indices.empty()) r.Fail("file has no triangles");
    if (!r.all_normals) r.g.normals.clear();
    if (!r.any_uv) r.g.uvs.clear();
    return std::move(r.g);
}

}  // namespace nuka::import
