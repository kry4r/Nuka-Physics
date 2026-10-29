#include "nk/model/articulation_ik.hpp"

#include "phi/articulation_contract.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace nuka::nk {
namespace {

constexpr uint32_t kNoLink = ~0u;
using JointType = phi::ArticulationJointType;

JointType TypeOf(const ModelArticulation& art, uint32_t link) {
    return static_cast<JointType>(art.joint_type[link]);
}

bool Moves(const ModelArticulation& art, uint32_t link) {
    return TypeOf(art, link) == JointType::Revolute || TypeOf(art, link) == JointType::Prismatic;
}

math::Vec3 UnitAxis(math::Vec3 axis) {
    const float length_sq = axis.LengthSq();
    return length_sq > 1.0e-12f ? axis * (1.0f / std::sqrt(length_sq)) : math::Vec3::Zero();
}

// The link frame in its parent's at coordinate q or spherical rotation, as JointRelativeFrame poses it.
math::Transform RelativeFrame(const ModelArticulation& art, uint32_t link, float q, const math::Quat& rotation) {
    const math::Transform& local = art.link_local_pose[link];
    math::Transform result{local.position + art.parent_offset[link], local.rotation};
    const math::Vec3 axis = UnitAxis(art.joint_axis[link]);
    if (TypeOf(art, link) == JointType::Revolute) {
        const float sine = std::sin(0.5f * q);
        const math::Quat turn(std::cos(0.5f * q), axis.x * sine, axis.y * sine, axis.z * sine);
        result.rotation = (local.rotation * turn).Normalized();
    } else if (TypeOf(art, link) == JointType::Prismatic) {
        result.position += local.rotation.Normalized().Rotate(axis * q);
    } else if (TypeOf(art, link) == JointType::Spherical) {
        result.rotation = (local.rotation * rotation).Normalized();
    }
    return result;
}

// Per link: its parent and articulation within the environment, and the coordinate a mimic
// follower takes as q = scale * q[source] + shift, composed down its chain like the device does.
struct Topology {
    std::vector<uint32_t> parent;
    std::vector<uint32_t> tree;
    std::vector<uint32_t> source;
    std::vector<float> scale;
    std::vector<float> shift;
};

Topology BuildTopology(const ModelArticulation& art) {
    const auto count = static_cast<uint32_t>(art.parent_link.size());
    Topology t;
    t.parent.resize(count);
    t.tree.resize(count);
    t.source.resize(count);
    t.scale.assign(count, 1.0f);
    t.shift.assign(count, 0.0f);
    // Links run root first, articulation by articulation, with parents local to theirs.
    uint32_t root = 0u, trees = 0u;
    for (uint32_t link = 0u; link < count; ++link) {
        if (art.parent_link[link] == kNoLink) {
            root = link;
            ++trees;
        }
        t.parent[link] = art.parent_link[link] == kNoLink ? kNoLink : root + art.parent_link[link];
        t.tree[link] = trees - 1u;
    }
    for (uint32_t link = 0u; link < count; ++link) {
        uint32_t at = link;
        for (uint32_t hop = 0u; hop < count && !art.mimic_source_link.empty(); ++hop) {
            const uint32_t from = art.mimic_source_link[at];
            if (from >= count || t.tree[from] != t.tree[at] || !Moves(art, at) || !Moves(art, from)) break;
            t.shift[link] += t.scale[link] * art.mimic_offset[at];
            t.scale[link] *= art.mimic_multiplier[at];
            at = from;
        }
        t.source[link] = at;
    }
    return t;
}

std::vector<math::Transform> Poses(const ModelArticulation& art, const Topology& t, const float* q,
                                   const math::Quat* rotations, const math::Transform* base_poses) {
    std::vector<math::Transform> pose(t.parent.size());
    for (uint32_t link = 0u; link < pose.size(); ++link) {
        const math::Transform relative = RelativeFrame(art, link, q[link], rotations[link]);
        if (t.parent[link] != kNoLink)
            pose[link] = pose[t.parent[link]] * relative;
        else
            pose[link] = TypeOf(art, link) == JointType::FloatingBase ? base_poses[t.tree[link]] : relative;
    }
    return pose;
}

// The world rotation vector that turns `from` onto `to`.
math::Vec3 RotationError(const math::Quat& to, const math::Quat& from) {
    math::Quat d = (to * from.Conjugate()).Normalized();
    if (d.w < 0.0f) d = math::Quat(-d.w, -d.x, -d.y, -d.z);
    const math::Vec3 v{d.x, d.y, d.z};
    const float sine = v.Length();
    return sine > 1.0e-9f ? v * (2.0f * std::atan2(sine, d.w) / sine) : v * 2.0f;
}

// Solves m x = g in place for a symmetric positive definite n x n row-major m.
void SolveSpd(std::vector<double>& m, std::vector<double>& g, size_t n) {
    for (size_t j = 0; j < n; ++j) {
        double d = m[j * n + j];
        for (size_t k = 0; k < j; ++k) d -= m[j * n + k] * m[j * n + k];
        m[j * n + j] = std::sqrt(d);
        for (size_t i = j + 1u; i < n; ++i) {
            double v = m[i * n + j];
            for (size_t k = 0; k < j; ++k) v -= m[i * n + k] * m[j * n + k];
            m[i * n + j] = v / m[j * n + j];
        }
    }
    for (size_t i = 0; i < n; ++i) {
        for (size_t k = 0; k < i; ++k) g[i] -= m[i * n + k] * g[k];
        g[i] /= m[i * n + i];
    }
    for (size_t i = n; i-- > 0u;) {
        for (size_t k = i + 1u; k < n; ++k) g[i] -= m[k * n + i] * g[k];
        g[i] /= m[i * n + i];
    }
}

}  // namespace

std::vector<math::Transform> ArticulationLinkPoses(const ModelArticulation& art, const float* q,
                                                   const math::Quat* rotations,
                                                   const math::Transform* base_poses) {
    return Poses(art, BuildTopology(art), q, rotations, base_poses);
}

bool SolveArticulationIk(const ModelArticulation& art, const float* q0, const math::Quat* rotations,
                         const math::Transform* base_poses,
                         const std::vector<IkTarget>& targets, const IkOptions& o, IkSolution* out) {
    const Topology t = BuildTopology(art);
    const auto count = static_cast<uint32_t>(t.parent.size());
    // A target's chain is the moving joints from its link up to the root, not the root's own; the
    // free coordinates are their sources.
    std::vector<uint32_t> column(count, kNoLink);
    std::vector<uint32_t> coords;
    std::vector<std::vector<uint32_t>> chains(targets.size());
    for (size_t i = 0; i < targets.size(); ++i) {
        uint32_t root = o.root;
        for (uint32_t link = targets[i].link; root == kNoLink; link = t.parent[link])
            if (t.parent[link] == kNoLink) root = link;
        for (uint32_t link = targets[i].link; link != root; link = t.parent[link]) {
            if (link == kNoLink) return false;
            if (!Moves(art, link)) continue;
            chains[i].push_back(link);
            const uint32_t source = t.source[link];
            if (column[source] == kNoLink) {
                column[source] = static_cast<uint32_t>(coords.size());
                coords.push_back(source);
            }
        }
    }
    const size_t n = coords.size();
    constexpr double kOpen = std::numeric_limits<double>::infinity();
    std::vector<double> x(n), lower(n, -kOpen), upper(n, kOpen);
    for (size_t c = 0; c < n; ++c) {
        const uint32_t link = coords[c];
        if ((art.joint_limit_flags[link] & 1u) != 0u) lower[c] = art.joint_limit_lower[link];
        if ((art.joint_limit_flags[link] & 2u) != 0u) upper[c] = art.joint_limit_upper[link];
        x[c] = std::clamp(static_cast<double>(q0[link]), lower[c], upper[c]);
    }
    std::vector<float> q(q0, q0 + count);
    std::vector<uint32_t> moved;
    for (uint32_t link = 0u; link < count; ++link)
        if (Moves(art, link) && column[t.source[link]] != kNoLink) moved.push_back(link);
    const auto set = [&] {
        for (uint32_t link : moved) {
            const auto v = static_cast<float>(x[column[t.source[link]]]);
            q[link] = link == t.source[link] ? v : t.scale[link] * v + t.shift[link];
        }
    };
    set();

    std::vector<math::Transform> pose;
    std::vector<math::Vec3> point(targets.size()), ep(targets.size()), er(targets.size());
    size_t rows = 0u;
    for (const IkTarget& g : targets) rows += g.orient ? 6u : 3u;
    // Poses the links and measures the targets; true once every target is within tolerance.
    const auto measure = [&] {
        pose = Poses(art, t, q.data(), rotations, base_poses);
        bool done = true;
        for (size_t i = 0; i < targets.size(); ++i) {
            const IkTarget& g = targets[i];
            point[i] = pose[g.link].TransformPoint(g.point);
            ep[i] = g.position - point[i];
            er[i] = g.orient ? RotationError(g.rotation, pose[g.link].rotation) : math::Vec3::Zero();
            done = done && ep[i].Length() < o.position_tolerance && er[i].Length() < o.rotation_tolerance;
        }
        return done;
    };
    uint32_t steps = 0u;
    std::vector<double> a(rows * n), b(rows), m(n * n), dx(n);
    while (!measure() && steps < o.iterations && n > 0u) {
        std::fill(a.begin(), a.end(), 0.0);
        size_t r = 0u;
        for (size_t i = 0; i < targets.size(); ++i) {
            const IkTarget& g = targets[i];
            const double w = g.weight, wr = static_cast<double>(g.weight) * o.rotation_weight;
            for (uint32_t link : chains[i]) {
                const size_t c = column[t.source[link]];
                const double s = t.scale[link];
                const math::Vec3 axis = pose[link].rotation.Rotate(UnitAxis(art.joint_axis[link]));
                const bool revolute = TypeOf(art, link) == JointType::Revolute;
                const math::Vec3 linear = revolute ? axis.Cross(point[i] - pose[link].position) : axis;
                const float lin[3] = {linear.x, linear.y, linear.z}, ang[3] = {axis.x, axis.y, axis.z};
                for (size_t k = 0; k < 3u; ++k) {
                    a[(r + k) * n + c] += w * s * lin[k];
                    if (g.orient && revolute) a[(r + 3u + k) * n + c] += wr * s * ang[k];
                }
            }
            const float pe[3] = {ep[i].x, ep[i].y, ep[i].z}, re[3] = {er[i].x, er[i].y, er[i].z};
            for (size_t k = 0; k < 3u; ++k) {
                b[r + k] = w * pe[k];
                if (g.orient) b[r + 3u + k] = wr * re[k];
            }
            r += g.orient ? 6u : 3u;
        }
        // (A^T A + damping I) dx = A^T b, the step scaled down to max_step.
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j <= i; ++j) {
                double v = i == j ? static_cast<double>(o.damping) : 0.0;
                for (size_t k = 0; k < rows; ++k) v += a[k * n + i] * a[k * n + j];
                m[i * n + j] = m[j * n + i] = v;
            }
            dx[i] = 0.0;
            for (size_t k = 0; k < rows; ++k) dx[i] += a[k * n + i] * b[k];
        }
        SolveSpd(m, dx, n);
        double largest = 0.0;
        for (double d : dx) largest = std::max(largest, std::abs(d));
        const double shrink = largest > o.max_step ? o.max_step / largest : 1.0;
        for (size_t c = 0; c < n; ++c) x[c] = std::clamp(x[c] + shrink * dx[c], lower[c], upper[c]);
        set();
        ++steps;
    }

    out->q = std::move(q);
    out->moved = std::move(moved);
    out->position_error.resize(targets.size());
    out->rotation_error.resize(targets.size());
    for (size_t i = 0; i < targets.size(); ++i) {
        out->position_error[i] = ep[i].Length();
        out->rotation_error[i] = er[i].Length();
    }
    out->iterations = steps;
    out->joints = static_cast<uint32_t>(n);
    out->at_limit = 0u;
    for (size_t c = 0; c < n; ++c)
        if (x[c] <= lower[c] + 1.0e-6 || x[c] >= upper[c] - 1.0e-6) ++out->at_limit;
    return true;
}

}  // namespace nuka::nk
