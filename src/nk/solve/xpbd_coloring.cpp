// ---------------------------------------------------------------------------
// nk::XpbdColoring implementation. Greedy graph coloring per XPBD family +
// in-place color-contiguous reorder of the single-env constraint SoA.
// ---------------------------------------------------------------------------

#include "nk/solve/xpbd_coloring.hpp"

#include <cstdint>
#include <vector>

#include "nk/model/model.hpp"

namespace nuka::nk {

namespace {

// Greedy coloring: conflict = shared particle, constraints walked in ascending
// index (deterministic), lowest free color each. Returns colors + out_num_colors.
std::vector<uint32_t> GreedyColor(const std::vector<std::vector<uint32_t>>& members,
                                  uint32_t particle_count,
                                  uint32_t* out_num_colors) {
    const uint32_t n = static_cast<uint32_t>(members.size());
    std::vector<uint32_t> color(n, 0u);
    uint32_t num_colors = 0u;
    // per-particle list of colors already assigned to a constraint touching it.
    std::vector<std::vector<uint32_t>> particle_colors(particle_count);
    std::vector<uint8_t> used;  // used[c] == 1 => color c taken by a neighbor.
    for (uint32_t i = 0u; i < n; ++i) {
        used.assign(num_colors + 1u, 0u);
        for (const uint32_t p : members[i]) {
            if (p >= particle_count) continue;
            for (const uint32_t c : particle_colors[p]) {
                if (c < used.size()) used[c] = 1u;
            }
        }
        uint32_t chosen = 0u;
        while (chosen < num_colors && used[chosen] != 0u) ++chosen;
        color[i] = chosen;
        if (chosen >= num_colors) num_colors = chosen + 1u;
        for (const uint32_t p : members[i]) {
            if (p < particle_count) particle_colors[p].push_back(chosen);
        }
    }
    *out_num_colors = num_colors;
    return color;
}

// Stable permutation that groups constraint indices by ascending color (ties in
// original index order) + the flat {offset,count}-per-color segment table.
std::vector<uint32_t> ColorOrder(const std::vector<uint32_t>& color,
                                 uint32_t num_colors,
                                 std::vector<uint32_t>* out_segments) {
    const uint32_t n = static_cast<uint32_t>(color.size());
    std::vector<uint32_t> count(num_colors, 0u);
    for (uint32_t i = 0u; i < n; ++i) ++count[color[i]];
    std::vector<uint32_t> offset(num_colors, 0u);
    uint32_t acc = 0u;
    out_segments->assign(static_cast<size_t>(num_colors) * 2u, 0u);
    for (uint32_t c = 0u; c < num_colors; ++c) {
        offset[c] = acc;
        (*out_segments)[static_cast<size_t>(c) * 2u + 0u] = acc;
        (*out_segments)[static_cast<size_t>(c) * 2u + 1u] = count[c];
        acc += count[c];
    }
    std::vector<uint32_t> order(n, 0u);
    std::vector<uint32_t> cursor = offset;
    for (uint32_t i = 0u; i < n; ++i) {
        const uint32_t c = color[i];
        order[cursor[c]++] = i;  // new slot for old constraint i.
    }
    return order;
}

// Apply a permutation `order` (new slot -> old index) to a flat per-constraint
// vector with `stride` elements per constraint.
template <typename T>
void Permute(std::vector<T>& v, const std::vector<uint32_t>& order, uint32_t stride) {
    if (v.empty() || order.empty()) return;
    std::vector<T> out(v.size());
    const uint32_t n = static_cast<uint32_t>(order.size());
    for (uint32_t s = 0u; s < n; ++s) {
        const uint32_t old = order[s];
        for (uint32_t j = 0u; j < stride; ++j) {
            out[static_cast<size_t>(s) * stride + j] =
                v[static_cast<size_t>(old) * stride + j];
        }
    }
    v.swap(out);
}

}  // namespace

void XpbdColoring::Build(Model* model) {
    if (model == nullptr) return;
    Model::ModelParticles& mp = model->particles;
    ModelCapacities& cap = model->capacities;
    const uint32_t P = cap.particles_per_env;

    model->dist_color_segments.clear();
    model->bend_color_segments.clear();
    model->vol_color_segments.clear();
    model->sm_color_segments.clear();
    cap.xpbd_dist_colors = 0u;
    cap.xpbd_bend_colors = 0u;
    cap.xpbd_vol_colors = 0u;
    cap.xpbd_sm_colors = 0u;

    // DISTANCE: 2 particles per constraint (dist_a/b + rest/alpha parallel SoA).
    const uint32_t dn = static_cast<uint32_t>(mp.dist_a.size());
    if (dn > 0u) {
        std::vector<std::vector<uint32_t>> members(dn);
        for (uint32_t i = 0u; i < dn; ++i) members[i] = {mp.dist_a[i], mp.dist_b[i]};
        uint32_t nc = 0u;
        const std::vector<uint32_t> color = GreedyColor(members, P, &nc);
        const std::vector<uint32_t> order =
            ColorOrder(color, nc, &model->dist_color_segments);
        Permute(mp.dist_a, order, 1u);
        Permute(mp.dist_b, order, 1u);
        Permute(mp.dist_rest, order, 1u);
        Permute(mp.dist_alpha, order, 1u);
        cap.xpbd_dist_colors = nc;
    }

    // BEND: 4 particles per constraint (bend_particles[4*c]; gradients[4*c]).
    const uint32_t bn = static_cast<uint32_t>(mp.bend_alpha.size());
    if (bn > 0u) {
        std::vector<std::vector<uint32_t>> members(bn);
        for (uint32_t i = 0u; i < bn; ++i) {
            const size_t b = static_cast<size_t>(i) * 4u;
            members[i] = {mp.bend_particles[b + 0u], mp.bend_particles[b + 1u],
                          mp.bend_particles[b + 2u], mp.bend_particles[b + 3u]};
        }
        uint32_t nc = 0u;
        const std::vector<uint32_t> color = GreedyColor(members, P, &nc);
        const std::vector<uint32_t> order =
            ColorOrder(color, nc, &model->bend_color_segments);
        Permute(mp.bend_particles, order, 4u);
        Permute(mp.bend_gradients, order, 4u);
        Permute(mp.bend_alpha, order, 1u);
        cap.xpbd_bend_colors = nc;
    }

    // VOLUME: 4 particles per tet (vol_particles[4*c]; rest6/alpha parallel SoA).
    const uint32_t vn = static_cast<uint32_t>(mp.vol_alpha.size());
    if (vn > 0u) {
        std::vector<std::vector<uint32_t>> members(vn);
        for (uint32_t i = 0u; i < vn; ++i) {
            const size_t b = static_cast<size_t>(i) * 4u;
            members[i] = {mp.vol_particles[b + 0u], mp.vol_particles[b + 1u],
                          mp.vol_particles[b + 2u], mp.vol_particles[b + 3u]};
        }
        uint32_t nc = 0u;
        const std::vector<uint32_t> color = GreedyColor(members, P, &nc);
        const std::vector<uint32_t> order =
            ColorOrder(color, nc, &model->vol_color_segments);
        Permute(mp.vol_particles, order, 4u);
        Permute(mp.vol_rest6, order, 1u);
        Permute(mp.vol_alpha, order, 1u);
        cap.xpbd_vol_colors = nc;
    }

    // SHAPE-MATCH: cluster members = CSR slice; clusters conflict iff they share a
    // particle. Reorder per-cluster arrays + rebuild the flat pool (offsets shift).
    const uint32_t scn = static_cast<uint32_t>(mp.sm_cluster_size.size());
    if (scn > 0u) {
        std::vector<std::vector<uint32_t>> members(scn);
        for (uint32_t c = 0u; c < scn; ++c) {
            const uint32_t off = mp.sm_cluster_offset[c];
            const uint32_t sz = mp.sm_cluster_size[c];
            members[c].reserve(sz);
            for (uint32_t j = 0u; j < sz; ++j) {
                const size_t m = static_cast<size_t>(off) + j;
                if (m < mp.sm_particles.size()) members[c].push_back(mp.sm_particles[m]);
            }
        }
        uint32_t nc = 0u;
        const std::vector<uint32_t> color = GreedyColor(members, P, &nc);
        const std::vector<uint32_t> order =
            ColorOrder(color, nc, &model->sm_color_segments);
        // Rebuild the flat member pool in the new cluster order; recompute each
        // cluster's offset into the rebuilt pool (size/stiffness/centroid permute).
        std::vector<uint32_t> new_particles;
        std::vector<math::Vec3> new_rest_q;
        std::vector<float> new_mass;
        std::vector<uint32_t> new_offset(scn, 0u);
        std::vector<uint32_t> new_size(scn, 0u);
        std::vector<float> new_stiffness(scn, 0.0f);
        std::vector<math::Vec3> new_centroid(scn, math::Vec3{0.0f, 0.0f, 0.0f});
        new_particles.reserve(mp.sm_particles.size());
        new_rest_q.reserve(mp.sm_rest_q.size());
        new_mass.reserve(mp.sm_mass.size());
        for (uint32_t s = 0u; s < scn; ++s) {
            const uint32_t old = order[s];
            const uint32_t off = mp.sm_cluster_offset[old];
            const uint32_t sz = mp.sm_cluster_size[old];
            new_offset[s] = static_cast<uint32_t>(new_particles.size());
            new_size[s] = sz;
            new_stiffness[s] = mp.sm_stiffness[old];
            new_centroid[s] = mp.sm_rest_centroid[old];
            for (uint32_t j = 0u; j < sz; ++j) {
                const size_t m = static_cast<size_t>(off) + j;
                new_particles.push_back(mp.sm_particles[m]);
                new_rest_q.push_back(mp.sm_rest_q[m]);
                new_mass.push_back(mp.sm_mass[m]);
            }
        }
        mp.sm_particles.swap(new_particles);
        mp.sm_rest_q.swap(new_rest_q);
        mp.sm_mass.swap(new_mass);
        mp.sm_cluster_offset.swap(new_offset);
        mp.sm_cluster_size.swap(new_size);
        mp.sm_stiffness.swap(new_stiffness);
        mp.sm_rest_centroid.swap(new_centroid);
        cap.xpbd_sm_colors = nc;
    }
}

}  // namespace nuka::nk
