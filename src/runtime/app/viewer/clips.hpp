#pragma once
// Joint animation clips: per-joint tracks of keyed drive targets, how they sample and their JSON.

#include "scene/format/json.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace nuka::runtime::app::viewer {

enum class ClipInterp : uint8_t { Step, Linear, Bezier };

// A key holds `value` [rad or m] at `time` [s]; `interp` shapes the way to the next key, a Bezier
// leaving with `out_slope` and arriving with the next key's `in_slope` [per s].
struct ClipKey {
    float time = 0.0f;
    float value = 0.0f;
    ClipInterp interp = ClipInterp::Linear;
    float in_slope = 0.0f;
    float out_slope = 0.0f;
};

struct ClipTrack {
    std::string joint;
    std::vector<ClipKey> keys;  // by time, never empty
};

struct Clip {
    std::string name;
    float duration = 1.0f;  // s
    bool loop = false;
    std::vector<ClipTrack> tracks;
};

// A key by its joint and time, as selections and requests name it.
struct ClipKeyRef {
    std::string joint;
    float time = 0.0f;
};

// Keys closer than this in time are the same key.
inline constexpr float kClipKeyEpsilon = 1e-4f;

// The track's value at `time`; before its first key that key's, after its last that key's.
float SampleTrack(const ClipTrack& track, float time);
const ClipTrack* FindTrack(const Clip& clip, const std::string& joint);
ClipTrack* FindTrack(Clip* clip, const std::string& joint);
ClipKey* FindKey(Clip* clip, const std::string& joint, float time);
// Puts `key` on the joint's track in time order, replacing a key at its time.
void SetKey(Clip* clip, const std::string& joint, const ClipKey& key);
// Removes the joint's key at `time`, and its track once it has none; false without such a key.
bool RemoveKey(Clip* clip, const std::string& joint, float time);
// Shifts `keys` by `dt` within [0, duration] and their values by `dv`, replacing the keys where they
// land, and moves the refs along; false, changing nothing, when one is missing or named twice.
bool MoveKeys(Clip* clip, std::vector<ClipKeyRef>* keys, float dt, float dv);
size_t KeyCount(const Clip& clip);

const char* InterpName(ClipInterp interp);
// False for a name other than step, linear or bezier.
bool ParseInterp(const std::string& name, ClipInterp* out);

// {time, value, interp, in, out}; reading takes interp, in and out as optional.
scene::json::Value KeyJson(const ClipKey& key);
ClipKey KeyFromJson(const scene::json::Value& v);
// {name, duration, loop, tracks: [{joint, keys}]}; reading throws on a missing field.
scene::json::Value ClipJson(const Clip& clip);
Clip ClipFromJson(const scene::json::Value& v);

}  // namespace nuka::runtime::app::viewer
