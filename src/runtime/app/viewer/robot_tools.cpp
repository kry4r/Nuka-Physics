#include "runtime/app/viewer/robot_tools.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <utility>

namespace nuka::runtime::app::viewer {
namespace {

namespace json = scene::json;

constexpr std::pair<scene::ActuatorType, const char*> kActuatorTypes[] = {
    {scene::ActuatorType::Position, "position"},
    {scene::ActuatorType::Motor, "motor"},
    {scene::ActuatorType::Velocity, "velocity"},
    {scene::ActuatorType::Force, "force"}};

constexpr std::pair<scene::JointType, const char*> kJointTypes[] = {
    {scene::JointType::Fixed, "fixed"},         {scene::JointType::Revolute, "revolute"},
    {scene::JointType::Prismatic, "prismatic"}, {scene::JointType::Spherical, "spherical"},
    {scene::JointType::Free, "free"}};

const scene::JointRecord* IncomingJoint(const scene::SceneIR& scene, scene::BodyId body) {
    for (const scene::JointRecord& j : scene.Joints())
        if (j.child_body == body) return &j;
    return nullptr;
}

// One environment's row of a per-link field.
void DownloadLinks(nuka_world_handle world, nuka_state_field_t field, const WorldLayout& layout, uint32_t env,
                   std::vector<float>* out) {
    out->resize(layout.links_per_env);
    CheckResult(nuka_world_download_field(world, field, out->data(), out->size() * sizeof(float),
                                          static_cast<size_t>(env) * layout.links_per_env * sizeof(float)),
                "nuka_world_download_field");
}

float At(const std::vector<float>& v, uint32_t i) { return i < v.size() ? v[i] : 0.0f; }

constexpr double kPi = 3.14159265358979323846;

// Step figures against the settled value, the mean of the last tenth of the samples.
void FitStep(const GainTrace& t, GainMetrics* m) {
    const size_t n = t.q.size(), tail = std::max<size_t>(n / 10u, 1u);
    double settled = 0.0;
    for (size_t i = n - tail; i < n; ++i) settled += t.q[i];
    settled /= static_cast<double>(tail);
    const double start = t.q[0], change = settled - start;
    m->steady_error = static_cast<float>(t.base + t.amplitude - settled);
    if (std::abs(change) < 1e-6) return;
    double peak = -std::numeric_limits<double>::infinity();
    size_t at = 0u, outside = 0u;
    float t10 = GainMetrics::kNone, t90 = GainMetrics::kNone;
    for (size_t i = 0; i < n; ++i) {
        const double s = (t.q[i] - start) / change;
        if (std::isnan(t10) && s >= 0.1) t10 = t.time[i];
        if (std::isnan(t90) && s >= 0.9) t90 = t.time[i];
        if (s > peak) {
            peak = s;
            at = i;
        }
        if (std::abs(t.q[i] - settled) > 0.02 * std::abs(change)) outside = i;
    }
    m->rise = t90 - t10;
    if (outside + 1u < n) m->settling = t.time[outside + 1u];
    // A peak inside the settled tail is the approach, not an overshoot.
    const bool overshot = peak > 1.0 && at < n - tail;
    m->overshoot = overshot ? static_cast<float>((peak - 1.0) * 100.0) : 0.0f;
    if (!overshot) return;
    m->peak_time = t.time[at];
    const double l = std::log(peak - 1.0);
    const double zeta = -l / std::sqrt(kPi * kPi + l * l);
    m->zeta = static_cast<float>(zeta);
    m->omega = static_cast<float>(kPi / (static_cast<double>(t.time[at]) * std::sqrt(1.0 - zeta * zeta)));
}

// Least squares of the second half onto a sine, a cosine and a constant at the command's frequency.
void FitSine(const GainTrace& t, GainMetrics* m) {
    const size_t n = t.q.size(), from = n / 2u;
    const double w = 2.0 * kPi * t.frequency;
    double a[3][4] = {};
    double error = 0.0;
    for (size_t i = from; i < n; ++i) {
        const double basis[3] = {std::sin(w * t.time[i]), std::cos(w * t.time[i]), 1.0};
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) a[r][c] += basis[r] * basis[c];
            a[r][3] += basis[r] * t.q[i];
        }
        error += (static_cast<double>(t.target[i]) - t.q[i]) * (static_cast<double>(t.target[i]) - t.q[i]);
    }
    m->rmse = static_cast<float>(std::sqrt(error / static_cast<double>(n - from)));
    for (int p = 0; p < 3; ++p) {
        int best = p;
        for (int r = p + 1; r < 3; ++r)
            if (std::abs(a[r][p]) > std::abs(a[best][p])) best = r;
        std::swap(a[p], a[best]);
        if (std::abs(a[p][p]) < 1e-12) return;
        for (int r = 0; r < 3; ++r) {
            if (r == p) continue;
            const double f = a[r][p] / a[p][p];
            for (int c = p; c < 4; ++c) a[r][c] -= f * a[p][c];
        }
    }
    // a·sin + b·cos is R·sin(wt + phase); a negative amplitude flips the command.
    const double sine = a[0][3] / a[0][0], cosine = a[1][3] / a[1][1];
    const double phase = std::atan2(cosine, sine) * 180.0 / kPi + (t.amplitude < 0.0f ? 180.0 : 0.0);
    m->gain_db = static_cast<float>(20.0 * std::log10(std::hypot(sine, cosine) / std::abs(t.amplitude)));
    m->lag = static_cast<float>(std::remainder(-phase, 360.0));
}

}  // namespace

const char* ActuatorTypeName(scene::ActuatorType type) {
    for (const auto& [t, name] : kActuatorTypes)
        if (t == type) return name;
    return "";
}

bool ParseActuatorType(const std::string& name, scene::ActuatorType* out) {
    for (const auto& [t, n] : kActuatorTypes) {
        if (name == n) {
            *out = t;
            return true;
        }
    }
    return false;
}

const char* JointTypeName(scene::JointType type) {
    for (const auto& [t, name] : kJointTypes)
        if (t == type) return name;
    return "";
}

bool ParseJointType(const std::string& name, scene::JointType* out) {
    for (const auto& [t, n] : kJointTypes) {
        if (name == n) {
            *out = t;
            return true;
        }
    }
    return false;
}

std::vector<RobotJoint> RobotJoints(const scene::SceneIR& scene, const WorldLayout& layout,
                                    const std::function<std::string(scene::EntityId)>& path_of) {
    std::vector<RobotJoint> joints;
    for (uint32_t l = 0; l < layout.links_per_env; ++l) {
        const nuka_link_binding_t& link = layout.links[l];
        const scene::JointRecord* j = link.is_root != 0u ? nullptr : IncomingJoint(scene, link.body_index);
        if (j == nullptr || (j->type != scene::JointType::Revolute && j->type != scene::JointType::Prismatic)) continue;
        RobotJoint r;
        r.link = l;
        r.name = j->name;
        r.path = path_of(scene.EntityOfBody(link.body_index));
        r.prismatic = j->type == scene::JointType::Prismatic;
        r.limited = j->has_lower_limit && j->has_upper_limit;
        r.lower = j->lower_limit;
        r.upper = j->upper_limit;
        for (const scene::ActuatorRecord& a : scene.Actuators()) {
            if (a.joint_id != j->id) continue;
            r.drive = ActuatorTypeName(a.type);
            r.kp = a.type == scene::ActuatorType::Position ? a.gain : 0.0f;
            r.force_limit = a.force_limit;
        }
        r.damping = j->damping;
        joints.push_back(std::move(r));
    }
    return joints;
}

std::vector<std::string> RobotRoots(const scene::SceneIR& scene, const WorldLayout& layout,
                                    const std::function<std::string(scene::EntityId)>& path_of) {
    std::vector<std::string> roots;
    for (uint32_t l = 0; l < layout.links_per_env; ++l)
        if (layout.links[l].is_root != 0u) roots.push_back(path_of(scene.EntityOfBody(layout.links[l].body_index)));
    return roots;
}

std::vector<RobotLink> RobotLinks(const scene::SceneIR& scene, const WorldLayout& layout,
                                  const std::function<std::string(scene::EntityId)>& path_of) {
    std::vector<RobotLink> links(layout.links_per_env);
    for (uint32_t l = 0; l < layout.links_per_env; ++l) {
        const nuka_link_binding_t& link = layout.links[l];
        RobotLink& r = links[l];
        r.path = path_of(scene.EntityOfBody(link.body_index));
        r.mass = scene.GetBody(link.body_index).mass;
        r.type = "root";
        const scene::JointRecord* j = link.is_root != 0u ? nullptr : IncomingJoint(scene, link.body_index);
        if (j == nullptr) continue;
        r.joint = j->name;
        r.type = JointTypeName(j->type);
        for (uint32_t p = 0; p < layout.links_per_env; ++p)
            if (layout.links[p].body_index == j->parent_body) r.parent = static_cast<int>(p);
    }
    return links;
}

json::Value LinkJson(const RobotLink& link, const std::vector<RobotLink>& links) {
    json::Value v = json::Value::Object();
    v.Set("path", json::Value::Str(link.path));
    v.Set("parent", json::Value::Str(link.parent < 0 ? std::string() : links[static_cast<size_t>(link.parent)].path));
    v.Set("joint", json::Value::Str(link.joint));
    v.Set("type", json::Value::Str(link.type));
    v.Set("mass", json::Value::Float(link.mass));
    return v;
}

int FindJoint(const std::vector<RobotJoint>& joints, const std::string& name) {
    for (size_t i = 0; i < joints.size(); ++i)
        if (joints[i].name == name) return static_cast<int>(i);
    return -1;
}

void DownloadJointFrame(nuka_world_handle world, const WorldLayout& layout, uint32_t env, JointFrame* out) {
    DownloadLinks(world, NUKA_FIELD_JOINT_POSITION, layout, env, &out->q);
    DownloadLinks(world, NUKA_FIELD_JOINT_VELOCITY, layout, env, &out->qd);
    DownloadLinks(world, NUKA_FIELD_DRIVE_TARGET, layout, env, &out->target);
    DownloadLinks(world, NUKA_FIELD_ACTUATOR_EFFORT, layout, env, &out->effort);
    DownloadLinks(world, NUKA_FIELD_ACTUATOR_SATURATED, layout, env, &out->saturated);
}

void DownloadJointSample(nuka_world_handle world, const WorldLayout& layout, uint32_t env, JointSample* out) {
    DownloadLinks(world, NUKA_FIELD_JOINT_POSITION, layout, env, &out->q);
    DownloadLinks(world, NUKA_FIELD_DRIVE_TARGET, layout, env, &out->target);
    DownloadLinks(world, NUKA_FIELD_ACTUATOR_EFFORT, layout, env, &out->effort);
}

json::Value JointJson(const RobotJoint& joint, const JointFrame& live) {
    json::Value v = json::Value::Object();
    v.Set("name", json::Value::Str(joint.name));
    v.Set("link", json::Value::Int(joint.link));
    v.Set("path", json::Value::Str(joint.path));
    v.Set("type", json::Value::Str(joint.prismatic ? "prismatic" : "revolute"));
    v.Set("lower", joint.limited ? json::Value::Float(joint.lower) : json::Value::Null());
    v.Set("upper", joint.limited ? json::Value::Float(joint.upper) : json::Value::Null());
    v.Set("drive", joint.drive.empty() ? json::Value::Null() : json::Value::Str(joint.drive));
    v.Set("kp", json::Value::Float(joint.kp));
    v.Set("force_limit", json::Value::Float(joint.force_limit));
    v.Set("damping", json::Value::Float(joint.damping));
    v.Set("q", json::Value::Float(At(live.q, joint.link)));
    v.Set("qd", json::Value::Float(At(live.qd, joint.link)));
    v.Set("target", json::Value::Float(At(live.target, joint.link)));
    v.Set("effort", json::Value::Float(At(live.effort, joint.link)));
    v.Set("saturated", json::Value::Bool(At(live.saturated, joint.link) != 0.0f));
    return v;
}

GainMetrics FitGain(const GainTrace& trace) {
    GainMetrics m;
    const size_t n = trace.q.size();
    if (n < 3u) return m;
    double saturated = 0.0;
    for (size_t i = 1; i < n; ++i) saturated += trace.saturated[i] != 0.0f ? 1.0 : 0.0;
    m.saturation = static_cast<float>(saturated / static_cast<double>(n - 1u));
    if (trace.sine) FitSine(trace, &m);
    else FitStep(trace, &m);
    return m;
}

json::Value GainRunJson(const GainRun& run, bool trace) {
    const auto number = [](float x) { return std::isfinite(x) ? json::Value::Float(x) : json::Value::Null(); };
    const GainTrace& t = run.trace;
    const GainMetrics& m = run.metrics;
    json::Value v = json::Value::Object();
    v.Set("joint", json::Value::Str(run.joint));
    v.Set("kind", json::Value::Str(t.sine ? "sine" : "step"));
    v.Set("amplitude", json::Value::Float(t.amplitude));
    v.Set("frequency", t.sine ? json::Value::Float(t.frequency) : json::Value::Null());
    v.Set("duration", json::Value::Float(run.duration));
    v.Set("steps", json::Value::Int(t.q.empty() ? 0 : static_cast<int64_t>(t.q.size()) - 1));
    v.Set("kp", json::Value::Float(run.kp));
    v.Set("damping", json::Value::Float(run.damping));
    v.Set("force_limit", json::Value::Float(run.force_limit));
    for (const auto& [key, x] : std::initializer_list<std::pair<const char*, float>>{
             {"rise", m.rise}, {"overshoot", m.overshoot}, {"peak_time", m.peak_time}, {"settling", m.settling},
             {"steady_error", m.steady_error}, {"zeta", m.zeta}, {"omega", m.omega}, {"gain_db", m.gain_db},
             {"lag", m.lag}, {"rmse", m.rmse}, {"saturation", m.saturation}})
        v.Set(key, number(x));
    if (!trace) return v;
    for (const auto& [key, xs] : std::initializer_list<std::pair<const char*, const std::vector<float>*>>{
             {"time", &t.time}, {"target", &t.target}, {"q", &t.q}, {"qd", &t.qd}, {"effort", &t.effort}}) {
        json::Value list = json::Value::Array();
        for (const float x : *xs) list.PushBack(json::Value::Float(x));
        v.Set(key, std::move(list));
    }
    return v;
}

}  // namespace nuka::runtime::app::viewer
