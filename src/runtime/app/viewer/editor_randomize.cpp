#include "runtime/app/viewer/editor_randomize.hpp"

#include <algorithm>

namespace nuka::runtime::app::viewer {
namespace {

namespace json = scene::json;

json::Value Pair(float lo, float hi) {
    json::Value a = json::Value::Array();
    a.PushBack(json::Value::Float(lo));
    a.PushBack(json::Value::Float(hi));
    return a;
}

// Reads [lo, hi] into `lo` and `hi`; a factor must stay positive.
bool ReadRange(const json::Value& v, const std::string& name, bool factor, float* lo, float* hi,
               std::string* error) {
    const bool pair = v.Size() == 2u;
    const float a = pair ? v.Elements()[0].AsFloat() : 0.0f, b = pair ? v.Elements()[1].AsFloat() : 0.0f;
    if (!pair || a > b || (factor && !(a > 0.0f))) {
        *error = "the " + name + " range is [lo, hi] with " + (factor ? "0 < lo <= hi" : "lo <= hi");
        return false;
    }
    *lo = a;
    *hi = b;
    return true;
}

}  // namespace

json::Value RandomizeToJson(const RandomizeConfig& c) {
    json::Value render = json::Value::Object();
    render.Set("on", json::Value::Bool(c.render));
    for (size_t k = 0; k < kJitterCount; ++k) render.Set(kRenderJitters[k], json::Value::Float(c.jitter[k]));
    json::Value physics = json::Value::Object();
    physics.Set("on", json::Value::Bool(c.physics));
    for (size_t k = 0; k < kRangeCount; ++k)
        physics.Set(kPhysicsRanges[k], Pair(c.range[2u * k], c.range[2u * k + 1u]));
    json::Value params = json::Value::Object();
    for (const ParamRange& p : c.params) {
        json::Value entry = json::Value::Object();
        entry.Set("on", json::Value::Bool(p.on));
        entry.Set("range", Pair(p.lo, p.hi));
        params.Set(p.key, std::move(entry));
    }
    json::Value v = json::Value::Object();
    v.Set("seed", json::Value::Int(c.seed));
    v.Set("render", std::move(render));
    v.Set("physics", std::move(physics));
    v.Set("params", std::move(params));
    v.Set("reset_draws", json::Value::Bool(c.reset_draws));
    return v;
}

bool RandomizeFromJson(const json::Value& v, const std::vector<std::string>& keys, RandomizeConfig* c,
                       std::string* error) {
    RandomizeConfig next = *c;
    if (const json::Value* f = v.Find("seed")) next.seed = static_cast<uint32_t>(f->AsInt());
    if (const json::Value* f = v.Find("reset_draws")) next.reset_draws = f->AsBool();
    if (const json::Value* r = v.Find("render")) {
        if (const json::Value* f = r->Find("on")) next.render = f->AsBool();
        for (size_t k = 0; k < kJitterCount; ++k) {
            const json::Value* f = r->Find(kRenderJitters[k]);
            if (f == nullptr) continue;
            if (f->AsFloat() < 0.0f) {
                *error = std::string("the ") + kRenderJitters[k] + " jitter cannot be negative";
                return false;
            }
            next.jitter[k] = f->AsFloat();
        }
    }
    if (const json::Value* p = v.Find("physics")) {
        if (const json::Value* f = p->Find("on")) next.physics = f->AsBool();
        // Mass is a factor, armature and gravity offsets.
        for (size_t k = 0; k < kRangeCount; ++k) {
            const json::Value* f = p->Find(kPhysicsRanges[k]);
            float* range = &next.range[2u * k];
            if (f != nullptr && !ReadRange(*f, kPhysicsRanges[k], k == 0u, range, range + 1, error)) return false;
        }
    }
    if (const json::Value* params = v.Find("params")) {
        for (const auto& [key, entry] : params->Items()) {
            if (std::find(keys.begin(), keys.end(), key) == keys.end()) {
                *error = "no parameter '" + key + "' takes a range";
                return false;
            }
            auto it = std::find_if(next.params.begin(), next.params.end(),
                                   [&](const ParamRange& p) { return p.key == key; });
            if (it == next.params.end()) it = next.params.insert(next.params.end(), ParamRange{key});
            if (const json::Value* f = entry.Find("on")) it->on = f->AsBool();
            const json::Value* range = entry.Find("range");
            if (range != nullptr && !ReadRange(*range, key, true, &it->lo, &it->hi, error)) return false;
        }
    }
    *c = std::move(next);
    return true;
}

bool RandomizableParam(const nuka_param_info_t& info) {
    return info.tier == NUKA_PARAM_LIVE && info.type == NUKA_PARAM_F32 && info.scene_field[0] != '\0';
}

}  // namespace nuka::runtime::app::viewer
