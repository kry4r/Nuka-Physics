#include "runtime/app/viewer/clips.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <stdexcept>

namespace nuka::runtime::app::viewer {

namespace json = scene::json;

namespace {

void InsertKey(std::vector<ClipKey>* keys, const ClipKey& key) {
    const auto at = std::lower_bound(keys->begin(), keys->end(), key.time - kClipKeyEpsilon,
                                     [](const ClipKey& k, float t) { return k.time < t; });
    if (at != keys->end() && std::abs(at->time - key.time) < kClipKeyEpsilon) *at = key;
    else keys->insert(at, key);
}

}  // namespace

float SampleTrack(const ClipTrack& track, float time) {
    const std::vector<ClipKey>& keys = track.keys;
    if (time <= keys.front().time) return keys.front().value;
    if (time >= keys.back().time) return keys.back().value;
    const auto next =
        std::upper_bound(keys.begin(), keys.end(), time, [](float t, const ClipKey& k) { return t < k.time; });
    const ClipKey& a = *std::prev(next);
    const ClipKey& b = *next;
    const float h = b.time - a.time, s = (time - a.time) / h, r = 1.0f - s;
    switch (a.interp) {
        case ClipInterp::Step:
            return a.value;
        case ClipInterp::Linear:
            return a.value + (b.value - a.value) * s;
        case ClipInterp::Bezier: {
            // Control values a third of the way along each end's slope.
            const float p1 = a.value + a.out_slope * h / 3.0f, p2 = b.value - b.in_slope * h / 3.0f;
            return r * r * r * a.value + 3.0f * r * r * s * p1 + 3.0f * r * s * s * p2 + s * s * s * b.value;
        }
    }
    return a.value;
}

const ClipTrack* FindTrack(const Clip& clip, const std::string& joint) {
    const auto it = std::find_if(clip.tracks.begin(), clip.tracks.end(), [&](const ClipTrack& t) { return t.joint == joint; });
    return it == clip.tracks.end() ? nullptr : &*it;
}

ClipTrack* FindTrack(Clip* clip, const std::string& joint) {
    const auto it = std::find_if(clip->tracks.begin(), clip->tracks.end(), [&](const ClipTrack& t) { return t.joint == joint; });
    return it == clip->tracks.end() ? nullptr : &*it;
}

ClipKey* FindKey(Clip* clip, const std::string& joint, float time) {
    ClipTrack* track = FindTrack(clip, joint);
    if (track == nullptr) return nullptr;
    const auto it = std::find_if(track->keys.begin(), track->keys.end(),
                                 [&](const ClipKey& k) { return std::abs(k.time - time) < kClipKeyEpsilon; });
    return it == track->keys.end() ? nullptr : &*it;
}

void SetKey(Clip* clip, const std::string& joint, const ClipKey& key) {
    if (ClipTrack* track = FindTrack(clip, joint)) InsertKey(&track->keys, key);
    else clip->tracks.push_back(ClipTrack{joint, {key}});
}

bool RemoveKey(Clip* clip, const std::string& joint, float time) {
    const auto track = std::find_if(clip->tracks.begin(), clip->tracks.end(), [&](const ClipTrack& t) { return t.joint == joint; });
    if (track == clip->tracks.end()) return false;
    const auto key = std::find_if(track->keys.begin(), track->keys.end(),
                                  [&](const ClipKey& k) { return std::abs(k.time - time) < kClipKeyEpsilon; });
    if (key == track->keys.end()) return false;
    track->keys.erase(key);
    if (track->keys.empty()) clip->tracks.erase(track);
    return true;
}

bool MoveKeys(Clip* clip, std::vector<ClipKeyRef>* keys, float dt, float dv) {
    std::vector<const ClipKey*> found;
    for (const ClipKeyRef& r : *keys) {
        const ClipKey* k = FindKey(clip, r.joint, r.time);
        if (k == nullptr || std::find(found.begin(), found.end(), k) != found.end()) return false;
        found.push_back(k);
    }
    // Every key leaves its track before any lands, so keys moving past each other keep their values.
    std::vector<std::pair<ClipTrack*, ClipKey>> moved;
    for (ClipKeyRef& r : *keys) {
        ClipTrack* track = FindTrack(clip, r.joint);
        const auto it = std::find_if(track->keys.begin(), track->keys.end(),
                                     [&](const ClipKey& k) { return std::abs(k.time - r.time) < kClipKeyEpsilon; });
        ClipKey key = *it;
        track->keys.erase(it);
        key.time = std::clamp(key.time + dt, 0.0f, clip->duration);
        key.value += dv;
        r.time = key.time;
        moved.emplace_back(track, key);
    }
    for (const auto& [track, key] : moved) InsertKey(&track->keys, key);
    return true;
}

size_t KeyCount(const Clip& clip) {
    size_t n = 0u;
    for (const ClipTrack& t : clip.tracks) n += t.keys.size();
    return n;
}

const char* InterpName(ClipInterp interp) {
    switch (interp) {
        case ClipInterp::Step:
            return "step";
        case ClipInterp::Linear:
            return "linear";
        case ClipInterp::Bezier:
            return "bezier";
    }
    return "linear";
}

bool ParseInterp(const std::string& name, ClipInterp* out) {
    for (const ClipInterp i : {ClipInterp::Step, ClipInterp::Linear, ClipInterp::Bezier}) {
        if (name != InterpName(i)) continue;
        *out = i;
        return true;
    }
    return false;
}

json::Value KeyJson(const ClipKey& key) {
    json::Value v = json::Value::Object();
    v.Set("time", json::Value::Float(key.time));
    v.Set("value", json::Value::Float(key.value));
    v.Set("interp", json::Value::Str(InterpName(key.interp)));
    v.Set("in", json::Value::Float(key.in_slope));
    v.Set("out", json::Value::Float(key.out_slope));
    return v;
}

ClipKey KeyFromJson(const json::Value& v) {
    ClipKey key;
    key.time = v.At("time").AsFloat();
    key.value = v.At("value").AsFloat();
    if (const json::Value* interp = v.Find("interp"); interp != nullptr && !ParseInterp(interp->AsString(), &key.interp))
        throw std::runtime_error("unknown interpolation '" + interp->AsString() + "'; it is step, linear or bezier");
    if (const json::Value* in = v.Find("in")) key.in_slope = in->AsFloat();
    if (const json::Value* out = v.Find("out")) key.out_slope = out->AsFloat();
    return key;
}

json::Value ClipJson(const Clip& clip) {
    json::Value tracks = json::Value::Array();
    for (const ClipTrack& t : clip.tracks) {
        json::Value keys = json::Value::Array();
        for (const ClipKey& k : t.keys) keys.PushBack(KeyJson(k));
        json::Value track = json::Value::Object();
        track.Set("joint", json::Value::Str(t.joint));
        track.Set("keys", std::move(keys));
        tracks.PushBack(std::move(track));
    }
    json::Value v = json::Value::Object();
    v.Set("name", json::Value::Str(clip.name));
    v.Set("duration", json::Value::Float(clip.duration));
    v.Set("loop", json::Value::Bool(clip.loop));
    v.Set("tracks", std::move(tracks));
    return v;
}

Clip ClipFromJson(const json::Value& v) {
    Clip clip;
    clip.name = v.At("name").AsString();
    clip.duration = v.At("duration").AsFloat();
    clip.loop = v.At("loop").AsBool();
    for (const json::Value& t : v.At("tracks").Elements())
        for (const json::Value& k : t.At("keys").Elements()) SetKey(&clip, t.At("joint").AsString(), KeyFromJson(k));
    return clip;
}

}  // namespace nuka::runtime::app::viewer
