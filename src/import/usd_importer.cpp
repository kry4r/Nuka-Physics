// ---------------------------------------------------------------------------
// nuka::import - USD / USDA importer implementation
// ---------------------------------------------------------------------------

#include "import/usd_importer.hpp"

#include "import/usd_crate_reader.hpp"
#include "import/usd_stage_adapter.hpp"
#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace nuka::import {

namespace {

struct UsdPrim {
    std::string type;
    std::string name;
    std::string path;
    std::string parent_path;
    bool has_rigid_body_api = false;
    bool has_collision_api = false;
    bool rigid_body_enabled = false;
    bool kinematic_enabled = false;
    float mass = 1.0f;
    math::Vec3 diagonal_inertia = {1.0f, 1.0f, 1.0f};
    math::Vec3 translate = math::Vec3::Zero();
    math::Vec3 scale = {1.0f, 1.0f, 1.0f};   // xformOp:scale (baked into grafted mesh)
    float cube_size = 1.0f;
    float radius = 0.5f;
    float height = 1.0f;
    math::Vec3 color = {1.0f, 1.0f, 1.0f};
    float alpha = 1.0f;
    float roughness = 0.5f;
    float metallic = 0.0f;
    bool has_friction = false;           // physics:dynamicFriction authored?
    float friction_mu = 1.0f;            // UsdPhysics PhysicsMaterialAPI friction
    float intensity = 1.0f;
    float focal_length = 50.0f;
    float near_clip = 0.01f;
    float far_clip = 1000.0f;
    std::string body0_path;
    std::string body1_path;
    std::string joint_path;
    std::string body_path;
    std::string axis_token = "Z";
    std::string nuka_type;
    std::string purpose;                 // USD `purpose` token: default|render|proxy|guide
    std::string decompose_mode;          // nuka:decompose token
    int decompose_max_pieces = 32;       // nuka:decompose:max_pieces
    bool has_initial_position = false;
    float initial_position = 0.0f;
    float lower_limit = -3.14159f;
    float upper_limit = 3.14159f;
    float gain = 1.0f;
    float force_limit = 0.0f;
    float sample_rate_hz = 0.0f;

    // -- Mesh geometry (def Mesh; #32) --------------------------------------
    // Parsed straight from the USD attributes; triangulation to mesh_indices is
    // done at SceneIR-build time. mesh_points is flat x,y,z triples in file
    // order (no dedup).
    std::vector<float> mesh_points;            // point3f[] points
    std::vector<int> face_vertex_indices;      // int[] faceVertexIndices
    std::vector<int> face_vertex_counts;       // int[] faceVertexCounts (per-face vert count)

    // -- references (#32) ---------------------------------------------------
    // (prepend|append)? references = @relpath@</PrimPath>. The relpath is
    // resolved against the file the reference is authored in; the referenced
    // subtree is grafted under this prim with this prim's xformOp:scale baked
    // into the grafted mesh points.
    std::string reference_asset;               // relpath inside the @...@
    std::string reference_prim_path;           // the </...> target prim path
};

std::string Trim(std::string_view text) {
    size_t begin = 0;
    while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin]))) {
        ++begin;
    }
    size_t end = text.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1]))) {
        --end;
    }
    return std::string(text.substr(begin, end - begin));
}

bool StartsWith(std::string_view text, std::string_view prefix) {
    return text.size() >= prefix.size() && text.substr(0, prefix.size()) == prefix;
}

std::string Lowercase(std::string_view text) {
    std::string out(text);
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return out;
}

bool ParseDefLine(const std::string& line, std::string& type, std::string& name) {
    if (!StartsWith(line, "def ")) {
        return false;
    }
    std::istringstream stream(line);
    std::string def_token;
    stream >> def_token >> type;
    const size_t first_quote = line.find('"');
    const size_t second_quote = line.find('"', first_quote + 1);
    if (first_quote == std::string::npos || second_quote == std::string::npos) {
        return false;
    }
    name = line.substr(first_quote + 1, second_quote - first_quote - 1);
    return !type.empty() && !name.empty();
}

void ApplyApiSchemas(const std::string& line, UsdPrim& prim) {
    if (line.find("apiSchemas") == std::string::npos) {
        return;
    }
    prim.has_rigid_body_api =
        prim.has_rigid_body_api || line.find("PhysicsRigidBodyAPI") != std::string::npos;
    prim.has_collision_api =
        prim.has_collision_api || line.find("PhysicsCollisionAPI") != std::string::npos;
}

bool ParseBoolValue(const std::string& line, std::string_view key, bool& value) {
    if (line.find(key) == std::string::npos) {
        return false;
    }
    const size_t equals = line.find('=');
    if (equals == std::string::npos) {
        return false;
    }
    const std::string rhs = Lowercase(Trim(std::string_view(line).substr(equals + 1)));
    if (StartsWith(rhs, "true")) {
        value = true;
        return true;
    }
    if (StartsWith(rhs, "false")) {
        value = false;
        return true;
    }
    return false;
}

bool ParseFloatValue(const std::string& line, std::string_view key, float& value) {
    if (line.find(key) == std::string::npos) {
        return false;
    }
    const size_t equals = line.find('=');
    if (equals == std::string::npos) {
        return false;
    }
    std::istringstream stream(Trim(std::string_view(line).substr(equals + 1)));
    stream >> value;
    return !stream.fail();
}

bool ParseIntValue(const std::string& line, std::string_view key, int& value) {
    if (line.find(key) == std::string::npos) {
        return false;
    }
    const size_t equals = line.find('=');
    if (equals == std::string::npos) {
        return false;
    }
    std::istringstream stream(Trim(std::string_view(line).substr(equals + 1)));
    stream >> value;
    return !stream.fail();
}

bool ParseVec3Value(const std::string& line, std::string_view key, math::Vec3& value) {
    if (line.find(key) == std::string::npos) {
        return false;
    }
    const size_t open = line.find('(');
    const size_t close = line.find(')', open + 1);
    if (open == std::string::npos || close == std::string::npos) {
        return false;
    }
    std::string tuple = line.substr(open + 1, close - open - 1);
    std::replace(tuple.begin(), tuple.end(), ',', ' ');
    std::istringstream stream(tuple);
    stream >> value.x >> value.y >> value.z;
    return !stream.fail();
}

bool ParseRelationship(const std::string& line, std::string_view key, std::string& path) {
    if (line.find(key) == std::string::npos) {
        return false;
    }
    const size_t open = line.find('<');
    const size_t close = line.find('>', open + 1);
    if (open == std::string::npos || close == std::string::npos) {
        return false;
    }
    path = line.substr(open + 1, close - open - 1);
    return !path.empty();
}

bool ParseTokenValue(const std::string& line, std::string_view key, std::string& value) {
    if (line.find(key) == std::string::npos) {
        return false;
    }
    const size_t first_quote = line.find('"');
    const size_t second_quote = line.find('"', first_quote + 1);
    if (first_quote == std::string::npos || second_quote == std::string::npos) {
        return false;
    }
    value = line.substr(first_quote + 1, second_quote - first_quote - 1);
    return !value.empty();
}

// Extract the substring between the value-opening '[' and the matching final
// ']' on a (possibly multi-line-joined) logical line. Returns false if no
// balanced value-bracket pair is present.
//
// The search for the opening '[' starts AFTER the '=' so it never matches the
// '[]' of the SDF array-type token (e.g. `int[] faceVertexIndices = [ ... ]`):
// matching the first '[' in the whole line would otherwise capture the type
// token's bracket and leak `] faceVertexIndices = [ ...` into the body, which
// poisons the strict int reader. When there is no '=' (no value authored) we
// fall back to the first '[' so a value-less declaration still reports "no body"
// cleanly rather than mis-parsing.
bool ExtractBracketBody(const std::string& line, std::string& body) {
    const size_t equals = line.find('=');
    const size_t search_from = (equals == std::string::npos) ? 0 : equals + 1;
    const size_t open = line.find('[', search_from);
    if (open == std::string::npos) {
        return false;
    }
    const size_t close = line.rfind(']');
    if (close == std::string::npos || close < open) {
        return false;
    }
    body = line.substr(open + 1, close - open - 1);
    return true;
}

// Parse the body of a `point3f[] points = [ (x,y,z), (x,y,z), ... ]` array into
// flat x,y,z triples appended to out (file order, no dedup). Tuples are
// delimited by parentheses; commas inside and between tuples are treated as
// whitespace. Returns false (and leaves out untouched) on a malformed tuple.
bool ParseVec3Array(const std::string& body, std::vector<float>& out) {
    std::vector<float> parsed;
    size_t pos = 0;
    while (true) {
        const size_t open = body.find('(', pos);
        if (open == std::string::npos) {
            break;
        }
        const size_t close = body.find(')', open + 1);
        if (close == std::string::npos) {
            return false;
        }
        std::string tuple = body.substr(open + 1, close - open - 1);
        std::replace(tuple.begin(), tuple.end(), ',', ' ');
        std::istringstream stream(tuple);
        float x = 0.0f;
        float y = 0.0f;
        float z = 0.0f;
        stream >> x >> y >> z;
        if (stream.fail()) {
            return false;
        }
        parsed.push_back(x);
        parsed.push_back(y);
        parsed.push_back(z);
        pos = close + 1;
    }
    out.insert(out.end(), parsed.begin(), parsed.end());
    return true;
}

// Parse the body of an `int[] ... = [ i, j, k, ... ]` array into ints appended
// to out (file order). Commas are treated as whitespace.
bool ParseIntArray(const std::string& body, std::vector<int>& out) {
    std::string normalized = body;
    std::replace(normalized.begin(), normalized.end(), ',', ' ');
    std::istringstream stream(normalized);
    std::vector<int> parsed;
    int value = 0;
    while (stream >> value) {
        parsed.push_back(value);
    }
    if (stream.bad()) {
        return false;
    }
    out.insert(out.end(), parsed.begin(), parsed.end());
    return true;
}

// Parse `(prepend|append)? references = @relpath@</PrimPath>` from a (logical)
// line. The asset path lives between the two '@' delimiters; the target prim
// path is the </...> relationship that follows.
bool ParseReference(const std::string& line, std::string& asset, std::string& prim_path) {
    if (line.find("references") == std::string::npos) {
        return false;
    }
    const size_t first_at = line.find('@');
    if (first_at == std::string::npos) {
        return false;
    }
    const size_t second_at = line.find('@', first_at + 1);
    if (second_at == std::string::npos) {
        return false;
    }
    asset = Trim(line.substr(first_at + 1, second_at - first_at - 1));
    if (asset.empty()) {
        return false;
    }
    const size_t open = line.find('<', second_at + 1);
    const size_t close = (open == std::string::npos) ? std::string::npos
                                                     : line.find('>', open + 1);
    if (open != std::string::npos && close != std::string::npos) {
        prim_path = line.substr(open + 1, close - open - 1);
    } else {
        prim_path.clear();
    }
    return true;
}

// Route a (logical) line's mesh-array / reference content into the prim. This is
// additive to ApplyPropertyLine (the scalar/relationship handler) -- a points /
// faceVertex* / references line carries none of the scalar keys, so calling both
// is safe.
void ApplyMeshAndReferenceLine(const std::string& line, UsdPrim& prim) {
    std::string body;
    if (line.find("point3f[] points") != std::string::npos ||
        line.find("float3[] points") != std::string::npos) {
        if (ExtractBracketBody(line, body)) {
            (void)ParseVec3Array(body, prim.mesh_points);
        }
        return;
    }
    if (line.find("faceVertexIndices") != std::string::npos) {
        if (ExtractBracketBody(line, body)) {
            (void)ParseIntArray(body, prim.face_vertex_indices);
        }
        return;
    }
    if (line.find("faceVertexCounts") != std::string::npos) {
        if (ExtractBracketBody(line, body)) {
            (void)ParseIntArray(body, prim.face_vertex_counts);
        }
        return;
    }
    if (line.find("references") != std::string::npos) {
        std::string asset;
        std::string ref_prim;
        if (ParseReference(line, asset, ref_prim)) {
            prim.reference_asset = asset;
            prim.reference_prim_path = ref_prim;
        }
    }
}

void ApplyPropertyLine(const std::string& line, UsdPrim& prim) {
    ApplyApiSchemas(line, prim);
    (void)ParseBoolValue(line, "physics:rigidBodyEnabled", prim.rigid_body_enabled);
    (void)ParseBoolValue(line, "physics:kinematicEnabled", prim.kinematic_enabled);
    (void)ParseFloatValue(line, "physics:mass", prim.mass);
    (void)ParseVec3Value(line, "physics:diagonalInertia", prim.diagonal_inertia);
    (void)ParseVec3Value(line, "xformOp:translate", prim.translate);
    (void)ParseVec3Value(line, "xformOp:scale", prim.scale);
    (void)ParseFloatValue(line, "double size", prim.cube_size);
    (void)ParseFloatValue(line, "float size", prim.cube_size);
    (void)ParseFloatValue(line, "double radius", prim.radius);
    (void)ParseFloatValue(line, "float radius", prim.radius);
    (void)ParseFloatValue(line, "double height", prim.height);
    (void)ParseFloatValue(line, "float height", prim.height);
    (void)ParseVec3Value(line, "inputs:diffuseColor", prim.color);
    (void)ParseVec3Value(line, "inputs:color", prim.color);
    (void)ParseFloatValue(line, "inputs:alpha", prim.alpha);
    (void)ParseFloatValue(line, "inputs:roughness", prim.roughness);
    (void)ParseFloatValue(line, "inputs:metallic", prim.metallic);
    // UsdPhysics PhysicsMaterialAPI: a scalar friction coefficient on a Material
    // prim. Mapped to MaterialRecord.friction_mu so the cooker can resolve it.
    if (ParseFloatValue(line, "physics:dynamicFriction", prim.friction_mu)) {
        prim.has_friction = true;
    }
    (void)ParseFloatValue(line, "inputs:intensity", prim.intensity);
    (void)ParseFloatValue(line, "focalLength", prim.focal_length);
    (void)ParseFloatValue(line, "clippingRange:min", prim.near_clip);
    (void)ParseFloatValue(line, "clippingRange:max", prim.far_clip);
    (void)ParseRelationship(line, "physics:body0", prim.body0_path);
    (void)ParseRelationship(line, "physics:body1", prim.body1_path);
    (void)ParseRelationship(line, "nuka:joint", prim.joint_path);
    (void)ParseRelationship(line, "nuka:body", prim.body_path);
    (void)ParseTokenValue(line, "physics:axis", prim.axis_token);
    (void)ParseTokenValue(line, "nuka:type", prim.nuka_type);
    // USD `uniform token purpose = "render"|"proxy"|"guide"` (default unset).
    // guide/proxy geometry is skipped at SceneIR build; render forces a geom to
    // be visual-only (non-colliding). ParseTokenValue keys on the quoted value,
    // so a bare token decl with no value leaves purpose empty (the default).
    (void)ParseTokenValue(line, "uniform token purpose", prim.purpose);
    // nuka:decompose:max_pieces (int) must be tried before nuka:decompose
    // (token) since the former line contains the latter as a substring; the
    // token parser requires quotes (absent on the int line) so it is harmless,
    // but parse the int form explicitly first for clarity.
    (void)ParseIntValue(line, "nuka:decompose:max_pieces", prim.decompose_max_pieces);
    if (line.find("nuka:decompose:max_pieces") == std::string::npos) {
        (void)ParseTokenValue(line, "nuka:decompose", prim.decompose_mode);
    }
    (void)ParseFloatValue(line, "physics:lowerLimit", prim.lower_limit);
    (void)ParseFloatValue(line, "physics:upperLimit", prim.upper_limit);
    if (ParseFloatValue(line, "nuka:initialPosition", prim.initial_position)) {
        prim.has_initial_position = true;
    }
    (void)ParseFloatValue(line, "nuka:gain", prim.gain);
    (void)ParseFloatValue(line, "nuka:forceLimit", prim.force_limit);
    (void)ParseFloatValue(line, "nuka:sampleRate", prim.sample_rate_hz);
    // Mesh geometry arrays + references (#32) -- additive, see helper note.
    ApplyMeshAndReferenceLine(line, prim);
}

std::string JoinPath(const std::string& parent, const std::string& name) {
    if (parent.empty() || parent == "/") {
        return "/" + name;
    }
    return parent + "/" + name;
}

// Net bracket balance of a string: (count of '[') - (count of ']').
int BracketBalance(const std::string& text) {
    int balance = 0;
    for (const char c : text) {
        if (c == '[') {
            ++balance;
        } else if (c == ']') {
            --balance;
        }
    }
    return balance;
}

// Resolve a reference asset path (relative or absolute) against the directory of
// the file that authored the reference.
std::string ResolveReferencePath(const std::string& base_dir, const std::string& asset) {
    std::filesystem::path p(asset);
    if (p.is_absolute() || base_dir.empty()) {
        return p.lexically_normal().string();
    }
    return (std::filesystem::path(base_dir) / p).lexically_normal().string();
}

// Forward declaration: parsing a file may recurse into referenced files, whose
// references are resolved against THEIR own directory.
std::vector<UsdPrim> ParseUsdaFileWithReferences(const std::string& base_dir,
                                                  const std::string& text);

// Parse USDA text into a flat prim list. Multi-line bracketed array values
// (point3f[] points, int[] faceVertex*) are joined into a single logical line
// before being handed to the (additive) property handlers, so the single-line
// path is just the degenerate "opens and closes on one physical line" case.
std::vector<UsdPrim> ParseUsdaText(const std::string& text) {
    constexpr size_t kNoPendingPrim = static_cast<size_t>(-1);
    std::vector<UsdPrim> prims;
    std::vector<size_t> stack;
    size_t pending_prim = kNoPendingPrim;

    std::istringstream input(text);
    std::string raw_line;
    while (std::getline(input, raw_line)) {
        std::string line = Trim(raw_line);
        if (line.empty() || StartsWith(line, "#")) {
            continue;
        }

        // Multi-line array join: if this line opens more '[' than it closes, an
        // array value spans subsequent physical lines -- accumulate them (as one
        // space-joined logical line) until the brackets balance. Brackets only
        // appear inside property values, never in the structural '{'/'}' prim
        // nesting, so this never swallows a scope close.
        int balance = BracketBalance(line);
        while (balance > 0 && std::getline(input, raw_line)) {
            const std::string continuation = Trim(raw_line);
            balance += BracketBalance(continuation);
            line += ' ';
            line += continuation;
        }

        std::string prim_type;
        std::string prim_name;
        if (ParseDefLine(line, prim_type, prim_name)) {
            const std::string parent_path = stack.empty() ? std::string{} : prims[stack.back()].path;
            UsdPrim prim;
            prim.type = std::move(prim_type);
            prim.name = std::move(prim_name);
            prim.parent_path = parent_path;
            prim.path = JoinPath(parent_path, prim.name);
            prims.push_back(std::move(prim));
            pending_prim = prims.size() - 1;
            ApplyPropertyLine(line, prims[pending_prim]);
            if (line.find('{') != std::string::npos) {
                stack.push_back(pending_prim);
                pending_prim = kNoPendingPrim;
            }
            continue;
        }

        if (pending_prim != kNoPendingPrim) {
            ApplyPropertyLine(line, prims[pending_prim]);
        } else if (!stack.empty()) {
            ApplyPropertyLine(line, prims[stack.back()]);
        }
        if (line.find('{') != std::string::npos && pending_prim != kNoPendingPrim) {
            stack.push_back(pending_prim);
            pending_prim = kNoPendingPrim;
        }
        if (line.find('}') != std::string::npos && !stack.empty()) {
            stack.pop_back();
        }
    }
    return prims;
}

// Bake a per-axis scale into a flat x,y,z vertex array in place. A (1,1,1) scale
// is an exact no-op.
void BakeScaleIntoPoints(std::vector<float>& points, const math::Vec3& scale) {
    if (scale.x == 1.0f && scale.y == 1.0f && scale.z == 1.0f) {
        return;
    }
    const size_t triples = points.size() / 3u;
    for (size_t t = 0; t < triples; ++t) {
        points[t * 3u + 0u] *= scale.x;
        points[t * 3u + 1u] *= scale.y;
        points[t * 3u + 2u] *= scale.z;
    }
}

// Graft a referenced prim subtree under a referencing prim. `ref_prims` is the
// fully-parsed (and reference-resolved) prim list of the referenced file;
// `target_path` is the </PrimPath> selected from it. The selected prim and all
// of its descendants are appended to `out` with their paths re-rooted under
// `host.path`, and `effective_scale` (the host's own xformOp:scale times every
// ancestor Xform's scale -- see ComputeEffectiveScale) is baked into every
// grafted Mesh prim's points.
void GraftReference(const UsdPrim& host,
                    const math::Vec3& effective_scale,
                    const std::vector<UsdPrim>& ref_prims,
                    const std::string& target_path,
                    std::vector<UsdPrim>& out) {
    // Find the selected root prim by exact path match.
    const UsdPrim* root = nullptr;
    for (const auto& prim : ref_prims) {
        if (prim.path == target_path) {
            root = &prim;
            break;
        }
    }
    if (root == nullptr) {
        return;
    }

    // Remap a referenced-file path onto the host: the selected root maps to the
    // host's own path; descendants keep their suffix below the root.
    const std::string& root_path = root->path;
    const auto remap = [&](const std::string& path) -> std::string {
        if (path == root_path) {
            return host.path;
        }
        // path begins with root_path + "/" for descendants.
        return host.path + path.substr(root_path.size());
    };

    for (const auto& prim : ref_prims) {
        const bool is_root = prim.path == root_path;
        const bool is_descendant =
            prim.path.size() > root_path.size() &&
            prim.path.compare(0, root_path.size(), root_path) == 0 &&
            prim.path[root_path.size()] == '/';
        if (!is_root && !is_descendant) {
            continue;
        }
        UsdPrim grafted = prim;
        grafted.path = remap(prim.path);
        grafted.parent_path = is_root ? host.parent_path : remap(prim.parent_path);
        // Bake the host's effective (accumulated ancestor) scale into the
        // grafted mesh vertices.
        BakeScaleIntoPoints(grafted.mesh_points, effective_scale);
        out.push_back(std::move(grafted));
    }
}

// The effective xformOp:scale for `prim`: the prim's own scale times every
// ancestor Xform's scale up the hierarchy. USD applies a mesh's whole ancestor
// transform chain; the cup authors its scale on the parent Xform while the
// reference sits on the child Mesh, so a self-only scale would be identity. Only
// scale is accumulated (this asset's rotate/translate are 0); per-component
// product, no rotate/translate baking (out of scope -- would be gold-plating).
math::Vec3 ComputeEffectiveScale(const UsdPrim& prim,
                                 const std::vector<UsdPrim>& prims) {
    std::unordered_map<std::string, const UsdPrim*> by_path;
    by_path.reserve(prims.size());
    for (const auto& p : prims) {
        by_path[p.path] = &p;
    }
    math::Vec3 scale = prim.scale;
    std::string parent = prim.parent_path;
    while (!parent.empty() && parent != "/") {
        const auto it = by_path.find(parent);
        if (it == by_path.end()) {
            break;
        }
        scale.x *= it->second->scale.x;
        scale.y *= it->second->scale.y;
        scale.z *= it->second->scale.z;
        parent = it->second->parent_path;
    }
    return scale;
}

std::vector<UsdPrim> ParseUsdaFileWithReferences(const std::string& base_dir,
                                                 const std::string& text) {
    std::vector<UsdPrim> prims = ParseUsdaText(text);

    // Resolve references: for each prim carrying a reference, load + parse the
    // referenced file (resolved against THIS file's directory; its own
    // references resolve against ITS directory), select the target subtree, and
    // graft it under the referencing prim with the host's (accumulated) scale
    // baked in.
    std::vector<UsdPrim> grafted;
    for (const auto& prim : prims) {
        if (prim.reference_asset.empty()) {
            continue;
        }
        const std::string resolved = ResolveReferencePath(base_dir, prim.reference_asset);
        const math::Vec3 effective_scale = ComputeEffectiveScale(prim, prims);

        // A USDC (crate) binary reference target (the newton-assets cup's
        // ./mesh.usd; v0.8 C7a): parse the crate's Mesh geometry with the
        // self-written reader and synthesize a single ref Mesh prim at the
        // referenced prim path, so the SAME GraftReference + scale-bake +
        // SceneIR-build path the ASCII reference graft uses applies unchanged.
        // Detect by the crate MAGIC (not the extension): the cup's binary file
        // is named "mesh.usd", and a ".usd" file may be either ASCII or crate.
        if (DetectUsdStageFormat(resolved) == UsdStageFormat::UsdcCrate ||
            IsUsdcCrateFile(resolved)) {
            const CrateMesh mesh = ReadUsdcMesh(resolved, prim.reference_prim_path);
            UsdPrim ref_mesh;
            ref_mesh.type = "Mesh";
            // The path inside the referenced file (e.g. "/Model"); GraftReference
            // selects the prim whose path == reference_prim_path, so name them so.
            ref_mesh.path = mesh.prim_path;
            const size_t slash = mesh.prim_path.find_last_of('/');
            ref_mesh.name = (slash == std::string::npos)
                                ? mesh.prim_path
                                : mesh.prim_path.substr(slash + 1);
            ref_mesh.parent_path =
                (slash == std::string::npos || slash == 0) ? "/"
                                                           : mesh.prim_path.substr(0, slash);
            ref_mesh.mesh_points = mesh.points;
            ref_mesh.face_vertex_indices = mesh.face_vertex_indices;
            ref_mesh.face_vertex_counts = mesh.face_vertex_counts;
            const std::vector<UsdPrim> ref_prims = {ref_mesh};
            GraftReference(prim, effective_scale, ref_prims, prim.reference_prim_path, grafted);
            continue;
        }

        const UsdStageData stage = LoadUsdStageData(resolved);  // other binary -> throws
        const std::string ref_dir = std::filesystem::path(resolved).parent_path().string();
        const std::vector<UsdPrim> ref_prims =
            ParseUsdaFileWithReferences(ref_dir, stage.text);
        GraftReference(prim, effective_scale, ref_prims, prim.reference_prim_path, grafted);
    }
    if (!grafted.empty()) {
        prims.insert(prims.end(),
                     std::make_move_iterator(grafted.begin()),
                     std::make_move_iterator(grafted.end()));
    }
    return prims;
}

scene::BodyId FindNearestBodyAncestor(
    std::string path,
    const std::unordered_map<std::string, scene::BodyId>& body_map) {

    while (!path.empty()) {
        const auto it = body_map.find(path);
        if (it != body_map.end()) {
            return it->second;
        }
        const size_t slash = path.find_last_of('/');
        if (slash == std::string::npos || slash == 0) {
            break;
        }
        path = path.substr(0, slash);
    }
    return scene::kInvalidBody;
}

bool IsRigidBodyPrim(const UsdPrim& prim) {
    return prim.has_rigid_body_api && prim.rigid_body_enabled;
}

bool MeshHasGeometry(const UsdPrim& prim) {
    return prim.type == "Mesh" && !prim.mesh_points.empty();
}

bool ShapeTypeFromUsdPrim(const UsdPrim& prim, scene::ShapeType& shape_type) {
    // A def Mesh that carries geometry is always a TriMesh shape, even without a
    // PhysicsCollisionAPI (cup-like USD assets have no physics APIs; #32). All
    // other primitive shapes still require the collision API as before.
    if (MeshHasGeometry(prim)) {
        shape_type = scene::ShapeType::TriMesh;
        return true;
    }
    if (!prim.has_collision_api) {
        return false;
    }
    if (prim.type == "Cube") {
        shape_type = scene::ShapeType::Box;
        return true;
    }
    if (prim.type == "Sphere") {
        shape_type = scene::ShapeType::Sphere;
        return true;
    }
    if (prim.type == "Capsule" || prim.type == "Cylinder") {
        shape_type = scene::ShapeType::Capsule;
        return true;
    }
    if (prim.type == "Mesh") {
        shape_type = scene::ShapeType::TriMesh;
        return true;
    }
    return false;
}

// Fan-triangulate a USD face list into a flat triangle-index array. counts is
// the per-face vertex count (3 -> 1 tri pass-through; 4 -> 2 tris; N -> N-2).
// When counts is empty the indices are treated as already-triangulated triples.
// Deterministic: file order, no reordering.
std::vector<uint32_t> FanTriangulate(const std::vector<int>& indices,
                                     const std::vector<int>& counts) {
    std::vector<uint32_t> out;
    if (counts.empty()) {
        out.reserve(indices.size());
        for (const int idx : indices) {
            if (idx >= 0) {
                out.push_back(static_cast<uint32_t>(idx));
            }
        }
        return out;
    }
    size_t cursor = 0;
    for (const int raw_count : counts) {
        if (raw_count < 3) {
            cursor += static_cast<size_t>(raw_count < 0 ? 0 : raw_count);
            continue;
        }
        const size_t count = static_cast<size_t>(raw_count);
        if (cursor + count > indices.size()) {
            break;  // malformed: not enough indices for this face.
        }
        for (size_t v = 1; v + 1 < count; ++v) {
            out.push_back(static_cast<uint32_t>(indices[cursor]));
            out.push_back(static_cast<uint32_t>(indices[cursor + v]));
            out.push_back(static_cast<uint32_t>(indices[cursor + v + 1]));
        }
        cursor += count;
    }
    return out;
}

bool IsJointPrim(const UsdPrim& prim) {
    return StartsWith(prim.type, "Physics") && prim.type.find("Joint") != std::string::npos;
}

scene::JointType JointTypeFromUsdType(const std::string& type) {
    if (type == "PhysicsRevoluteJoint") {
        return scene::JointType::Revolute;
    }
    if (type == "PhysicsPrismaticJoint") {
        return scene::JointType::Prismatic;
    }
    if (type == "PhysicsSphericalJoint") {
        return scene::JointType::Spherical;
    }
    if (type == "PhysicsFixedJoint") {
        return scene::JointType::Fixed;
    }
    return scene::JointType::Fixed;
}

math::Vec3 AxisFromUsdToken(const std::string& axis) {
    if (axis == "X" || axis == "x") {
        return math::Vec3::UnitX();
    }
    if (axis == "Y" || axis == "y") {
        return math::Vec3::UnitY();
    }
    return math::Vec3::UnitZ();
}

scene::ActuatorType ActuatorTypeFromToken(const std::string& token) {
    const std::string lower = Lowercase(token);
    if (lower == "position") {
        return scene::ActuatorType::Position;
    }
    if (lower == "velocity") {
        return scene::ActuatorType::Velocity;
    }
    if (lower == "force") {
        return scene::ActuatorType::Force;
    }
    return scene::ActuatorType::Motor;
}

scene::SensorType SensorTypeFromToken(const std::string& token) {
    const std::string lower = Lowercase(token);
    if (lower == "lidar") {
        return scene::SensorType::Lidar;
    }
    if (lower == "camera") {
        return scene::SensorType::Camera;
    }
    if (lower == "force_torque" || lower == "forcetorque") {
        return scene::SensorType::ForceTorque;
    }
    if (lower == "contact") {
        return scene::SensorType::Contact;
    }
    if (lower == "framepose") {
        return scene::SensorType::FramePose;
    }
    return scene::SensorType::Imu;
}

scene::DecomposeMode DecomposeModeFromToken(const std::string& token) {
    const std::string lower = Lowercase(token);
    if (lower == "force") {
        return scene::DecomposeMode::Force;
    }
    if (lower == "skip") {
        return scene::DecomposeMode::Skip;
    }
    return scene::DecomposeMode::Auto;
}

scene::LightType LightTypeFromUsdType(const std::string& type) {
    if (type == "DistantLight") {
        return scene::LightType::Directional;
    }
    if (type == "RectLight" || type == "DiskLight") {
        return scene::LightType::Area;
    }
    return scene::LightType::Point;
}

scene::SceneIR BuildSceneFromUsdPrims(const std::vector<UsdPrim>& prims) {
    scene::SceneIR scene;
    std::unordered_map<std::string, scene::BodyId> body_map;
    std::unordered_map<std::string, scene::JointId> joint_map;

    for (const auto& prim : prims) {
        if (prim.type == "Material") {
            scene::MaterialRecord material;
            material.name = prim.name;
            material.base_color = prim.color;
            material.alpha = prim.alpha;
            material.roughness = prim.roughness;
            material.metallic = prim.metallic;
            if (prim.has_friction) {
                material.friction_mu = prim.friction_mu;
            }
            scene.AddMaterial(std::move(material));
        }
    }

    for (const auto& prim : prims) {
        if (!IsRigidBodyPrim(prim)) {
            continue;
        }
        scene::RigidBodyRecord body;
        body.name = prim.name;
        body.parent_id = FindNearestBodyAncestor(prim.parent_path, body_map);
        body.local_transform = math::Transform{prim.translate, math::Quat::Identity()};
        body.mass = prim.mass;
        body.inertia = prim.diagonal_inertia;
        body.is_static = prim.kinematic_enabled || prim.mass <= 0.0f;
        const scene::BodyId body_id = scene.AddRigidBody(std::move(body));
        body_map[prim.path] = body_id;
    }

    for (const auto& prim : prims) {
        // USD `purpose`: guide (viewport-only annotations) and proxy (low-detail
        // collision stand-in) geometry are not part of the render/physics scene
        // here -- skip them. render-purpose geometry is kept but forced
        // visual-only below (contype=0, so the facade projects a VisualMesh).
        const std::string purpose = Lowercase(prim.purpose);
        if (purpose == "guide" || purpose == "proxy") {
            continue;
        }
        scene::ShapeType shape_type = scene::ShapeType::Box;
        if (!ShapeTypeFromUsdPrim(prim, shape_type)) {
            continue;
        }
        scene::BodyId body_id = FindNearestBodyAncestor(prim.parent_path, body_map);
        if (body_id == scene::kInvalidBody) {
            // A geometry-bearing Mesh with no rigid-body ancestor (cup-like USD
            // with no physics APIs; #32) gets an implicit static body so the
            // shape has a target. Other shapes without a body are skipped as
            // before. Static is fine -- a later demo step adds a freejoint to
            // make the cup dynamic.
            if (!MeshHasGeometry(prim)) {
                continue;
            }
            scene::RigidBodyRecord implicit_body;
            implicit_body.name = prim.name + "_body";
            implicit_body.parent_id = scene::kInvalidBody;
            implicit_body.local_transform =
                math::Transform{prim.translate, math::Quat::Identity()};
            implicit_body.is_static = true;
            body_id = scene.AddRigidBody(std::move(implicit_body));
            body_map[prim.path] = body_id;
        }
        scene::CollisionShapeRecord shape;
        shape.name = prim.name;
        shape.body_id = body_id;
        shape.type = shape_type;
        shape.local_transform = math::Transform{prim.translate, math::Quat::Identity()};
        // render-purpose geometry is visual-only: turn collision off so the
        // SceneIR facade projects it to a VisualMeshComponent (M2b).
        if (purpose == "render") {
            shape.contype = 0;
            shape.conaffinity = 0;
        }
        if (shape_type == scene::ShapeType::Box) {
            const float half_size = prim.cube_size * 0.5f;
            shape.half_extents = {half_size, half_size, half_size};
        } else if (shape_type == scene::ShapeType::Sphere) {
            shape.radius = prim.radius;
        } else if (shape_type == scene::ShapeType::Capsule) {
            shape.radius = prim.radius;
            shape.half_height = prim.height * 0.5f;
        }
        if (shape_type == scene::ShapeType::TriMesh) {
            shape.decompose_mode = DecomposeModeFromToken(prim.decompose_mode);
            if (prim.decompose_max_pieces > 0) {
                shape.decompose_max_pieces =
                    static_cast<uint32_t>(prim.decompose_max_pieces);
            }
            // Mesh geometry (#32): points -> flat vertices (file order, no
            // dedup); faceVertexIndices + faceVertexCounts -> fan triangulation.
            shape.mesh_vertices = prim.mesh_points;
            shape.mesh_indices =
                FanTriangulate(prim.face_vertex_indices, prim.face_vertex_counts);
        }
        scene.AddCollisionShape(std::move(shape));
    }

    for (const auto& prim : prims) {
        if (!IsJointPrim(prim)) {
            continue;
        }
        const auto parent_it = body_map.find(prim.body0_path);
        const auto child_it = body_map.find(prim.body1_path);
        if (parent_it == body_map.end() || child_it == body_map.end()) {
            throw std::runtime_error("USD: joint '" + prim.name + "' references an unknown rigid body");
        }
        scene::JointRecord joint;
        joint.name = prim.name;
        joint.type = JointTypeFromUsdType(prim.type);
        joint.parent_body = parent_it->second;
        joint.child_body = child_it->second;
        joint.axis = AxisFromUsdToken(prim.axis_token);
        if (prim.has_initial_position) {
            joint.initial_position = prim.initial_position;
        }
        joint.lower_limit = prim.lower_limit;
        joint.upper_limit = prim.upper_limit;
        const scene::JointId joint_id = scene.AddJoint(std::move(joint));
        joint_map[prim.path] = joint_id;
    }

    for (const auto& prim : prims) {
        if (prim.type == "Camera") {
            scene::CameraRecord camera;
            camera.name = prim.name;
            camera.attached_body = FindNearestBodyAncestor(prim.parent_path, body_map);
            camera.local_transform = math::Transform{prim.translate, math::Quat::Identity()};
            camera.vertical_fov_degrees = prim.focal_length > 0.0f ? 50.0f : 45.0f;
            camera.near_clip = prim.near_clip;
            camera.far_clip = prim.far_clip;
            scene.AddCamera(std::move(camera));
        } else if (prim.type.find("Light") != std::string::npos) {
            scene::LightRecord light;
            light.name = prim.name;
            light.type = LightTypeFromUsdType(prim.type);
            light.local_transform = math::Transform{prim.translate, math::Quat::Identity()};
            light.color = prim.color;
            light.intensity = prim.intensity;
            scene.AddLight(std::move(light));
        } else if (prim.type == "NukaActuator") {
            const auto joint_it = joint_map.find(prim.joint_path);
            if (joint_it == joint_map.end()) {
                throw std::runtime_error("USD: actuator '" + prim.name + "' references an unknown joint");
            }
            scene::ActuatorRecord actuator;
            actuator.name = prim.name;
            actuator.joint_id = joint_it->second;
            actuator.type = ActuatorTypeFromToken(prim.nuka_type);
            actuator.gain = prim.gain;
            actuator.force_limit = prim.force_limit;
            scene.AddActuator(std::move(actuator));
        } else if (prim.type == "NukaSensor") {
            const auto body_it = body_map.find(prim.body_path);
            if (body_it == body_map.end()) {
                throw std::runtime_error("USD: sensor '" + prim.name + "' references an unknown body");
            }
            scene::SensorRecord sensor;
            sensor.name = prim.name;
            sensor.attached_body = body_it->second;
            sensor.type = SensorTypeFromToken(prim.nuka_type);
            sensor.sample_rate_hz = prim.sample_rate_hz;
            scene.AddSensor(std::move(sensor));
        }
    }
    return scene;
}

} // namespace

scene::SceneIR LoadUsd(const std::string& path) {
    const UsdStageData stage = LoadUsdStageData(path);
    const std::string base_dir = std::filesystem::path(path).parent_path().string();
    auto scene = BuildSceneFromUsdPrims(ParseUsdaFileWithReferences(base_dir, stage.text));

    // A valid USD scene must yield either rigid bodies (physics scenes) OR at
    // least one mesh shape (cup-like geometry-only assets; #32). It is an error
    // only when it yields NEITHER.
    bool has_mesh_geometry = false;
    for (const auto& shape : scene.Shapes()) {
        if (!shape.mesh_vertices.empty()) {
            has_mesh_geometry = true;
            break;
        }
    }
    if (scene.RigidBodyCount() == 0u && !has_mesh_geometry) {
        throw std::runtime_error(
            "USD: no enabled UsdPhysics rigid bodies and no mesh geometry found in " + path);
    }
    return scene;
}

} // namespace nuka::import
