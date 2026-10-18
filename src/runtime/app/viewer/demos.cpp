#include "runtime/app/viewer/demos.hpp"

#include "math/transform.hpp"
#include "runtime/app/viewer/editor_profile.hpp"
#include "scene/format/json.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <type_traits>

namespace nuka::runtime::app::viewer {
namespace {

namespace json = scene::json;

// One environment's row of a per-link field.
template <class T>
void DownloadEnv(nuka_world_handle world, nuka_state_field_t field, uint32_t links, uint32_t env, std::vector<T>* out) {
    out->resize(links);
    CheckResult(nuka_world_download_field(world, field, out->data(), out->size() * sizeof(T),
                                          static_cast<size_t>(env) * links * sizeof(T)),
                "nuka_world_download_field");
}

template <class T>
json::Value Array(const std::vector<T>& items) {
    json::Value a = json::Value::Array();
    for (const T& item : items) {
        if constexpr (std::is_same_v<T, std::string>)
            a.PushBack(json::Value::Str(item));
        else
            a.PushBack(json::Value::Int(static_cast<int64_t>(item)));
    }
    return a;
}

}  // namespace

void DownloadDemoSample(nuka_world_handle world, const WorldLayout& layout, uint32_t env,
                        const std::vector<uint32_t>& roots, DemoSample* out) {
    const uint32_t links = layout.links_per_env;
    out->env = env;
    DownloadEnv(world, NUKA_FIELD_JOINT_POSITION, links, env, &out->q);
    DownloadEnv(world, NUKA_FIELD_JOINT_VELOCITY, links, env, &out->qd);
    DownloadEnv(world, NUKA_FIELD_DRIVE_TARGET, links, env, &out->target);
    std::vector<math::Transform> poses;
    std::vector<std::array<float, 6>> velocities;
    DownloadEnv(world, NUKA_FIELD_ARTICULATION_LINK_POSE, links, env, &poses);
    DownloadEnv(world, NUKA_FIELD_LINK_VELOCITY, links, env, &velocities);
    out->root.clear();
    for (const uint32_t r : roots) {
        const math::Transform& p = poses[r];
        out->root.insert(out->root.end(), {p.position.x, p.position.y, p.position.z, p.rotation.w, p.rotation.x,
                                           p.rotation.y, p.rotation.z});
        out->root.insert(out->root.end(), velocities[r].begin(), velocities[r].end());
    }
}

// A sample that does not follow its environment's last one starts an episode: the playhead
// went back, or the samples between were never recorded.
void CutEpisodes(const std::vector<DemoSample>& samples, DemoSet* set) {
    const size_t first = set->episodes.size();
    for (const DemoSample& s : samples) {
        auto it = std::find_if(set->episodes.rbegin(), set->episodes.rend() - static_cast<std::ptrdiff_t>(first),
                               [&](const DemoEpisode& e) { return e.env == s.env; });
        if (it == set->episodes.rend() - static_cast<std::ptrdiff_t>(first) || it->start + it->Steps() + 1u != s.tick) {
            DemoEpisode e;
            e.env = s.env;
            e.start = s.tick - 1u;
            e.roots = static_cast<uint32_t>(s.root.size() / kRootValues);
            set->episodes.push_back(std::move(e));
            it = set->episodes.rbegin();
        }
        DemoEpisode& e = *it;
        e.time.push_back(s.time);
        for (const uint32_t link : set->links) {
            e.q.push_back(s.q[link]);
            e.qd.push_back(s.qd[link]);
            e.action.push_back(s.target[link]);
        }
        for (size_t r = 0; r < s.root.size(); r += kRootValues) {
            e.root_pose.insert(e.root_pose.end(), s.root.begin() + static_cast<std::ptrdiff_t>(r),
                               s.root.begin() + static_cast<std::ptrdiff_t>(r + 7u));
            e.root_vel.insert(e.root_vel.end(), s.root.begin() + static_cast<std::ptrdiff_t>(r + 7u),
                              s.root.begin() + static_cast<std::ptrdiff_t>(r + kRootValues));
        }
        e.command.insert(e.command.end(), s.command.begin(), s.command.end());
        for (size_t a = 0; a < set->axes.size(); ++a) e.axes.push_back(a < s.axes.size() ? s.axes[a] : 0.0f);
    }
}

// Each episode's arrays follow one another in `data`: time (float64), then q, qd, action,
// root_pose, root_vel, command and axes (float32), each row-major.
void WriteDemoExchange(const DemoSet& set, const std::string& manifest, const std::string& data) {
    std::ofstream out(Utf8Path(data), std::ios::binary | std::ios::trunc);
    const auto put = [&](const auto& v) {
        out.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(v[0])));
    };
    json::Value episodes = json::Value::Array();
    for (const DemoEpisode& e : set.episodes) {
        json::Value v = json::Value::Object();
        v.Set("env", json::Value::Int(e.env));
        v.Set("start", json::Value::Int(static_cast<int64_t>(e.start)));
        v.Set("steps", json::Value::Int(static_cast<int64_t>(e.Steps())));
        v.Set("roots", json::Value::Int(e.roots));
        v.Set("offset", json::Value::Int(static_cast<int64_t>(out.tellp())));
        episodes.PushBack(std::move(v));
        put(e.time);
        put(e.q);
        put(e.qd);
        put(e.action);
        put(e.root_pose);
        put(e.root_vel);
        put(e.command);
        put(e.axes);
    }
    json::Value m = json::Value::Object();
    m.Set("scene", json::Value::Str(set.scene));
    m.Set("dt", json::Value::Float(set.dt));
    m.Set("joints", Array(set.joints));
    m.Set("links", Array(set.links));
    m.Set("axes", Array(set.axes));
    m.Set("episodes", std::move(episodes));
    std::ofstream(Utf8Path(manifest), std::ios::binary | std::ios::trunc) << m.Dump();
}

DemoSet ReadDemoExchange(const std::string& manifest, const std::string& data) {
    std::ifstream text(Utf8Path(manifest), std::ios::binary);
    const json::Value m = json::Value::Parse(std::string(std::istreambuf_iterator<char>(text), {}));
    std::ifstream raw(Utf8Path(data), std::ios::binary);
    const std::string bytes(std::istreambuf_iterator<char>(raw), {});
    DemoSet set;
    set.scene = m.At("scene").AsString();
    set.dt = m.At("dt").AsFloat();
    for (const json::Value& j : m.At("joints").Elements()) set.joints.push_back(j.AsString());
    for (const json::Value& l : m.At("links").Elements()) set.links.push_back(static_cast<uint32_t>(l.AsInt()));
    for (const json::Value& a : m.At("axes").Elements()) set.axes.push_back(a.AsString());
    const size_t joints = set.joints.size(), axes = set.axes.size();
    for (const json::Value& v : m.At("episodes").Elements()) {
        DemoEpisode e;
        e.env = static_cast<uint32_t>(v.At("env").AsInt());
        e.start = static_cast<uint64_t>(v.At("start").AsInt());
        e.roots = static_cast<uint32_t>(v.At("roots").AsInt());
        const auto steps = static_cast<size_t>(v.At("steps").AsInt());
        auto at = static_cast<size_t>(v.At("offset").AsInt());
        const auto take = [&](auto* out, size_t count) {
            const size_t size = count * sizeof((*out)[0]);
            if (at + size > bytes.size()) throw std::runtime_error("the demonstration data ends early");
            out->resize(count);
            std::memcpy(out->data(), bytes.data() + at, size);
            at += size;
        };
        take(&e.time, steps);
        take(&e.q, steps * joints);
        take(&e.qd, steps * joints);
        take(&e.action, steps * joints);
        take(&e.root_pose, steps * e.roots * 7u);
        take(&e.root_vel, steps * e.roots * 6u);
        take(&e.command, steps * 3u);
        take(&e.axes, steps * axes);
        set.episodes.push_back(std::move(e));
    }
    return set;
}

}  // namespace nuka::runtime::app::viewer
