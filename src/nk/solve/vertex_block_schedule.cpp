#include "nk/solve/vertex_block_schedule.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <map>
#include <stdexcept>
#include <vector>

#include "nk/model/model.hpp"

namespace nuka::nk {

namespace {
// Pairs each triangle of a fracturing field with its bend element, for the crack kernels' face lookups.
void BuildFieldFacePartners(Model& model) {
    auto& p = model.particles;
    p.mf_face_partner.clear();
    if (model.capacities.mf_spares_per_env == 0u) return;
    const uint32_t count = static_cast<uint32_t>(p.mf_elements.size());
    p.mf_face_partner.assign(count, ~0u);
    std::map<std::array<uint32_t, 3>, uint32_t> faces;
    for (uint32_t i = 0u; i < count; ++i) {
        const auto& e = p.mf_elements[i];
        if (e.kind == kMfTriangle) faces[{e.vertex[0], e.vertex[1], e.vertex[2]}] = i;
    }
    for (uint32_t i = 0u; i < count; ++i) {
        const auto& e = p.mf_elements[i];
        if (e.kind != kMfBend) continue;
        const uint32_t face = faces.at({e.vertex[0], e.vertex[1], e.vertex[2]});
        p.mf_face_partner[i] = face;
        p.mf_face_partner[face] = i;
    }
}
}  // namespace

void BuildVertexBlockSchedule(Model* model) {
    if (model == nullptr) return;
    auto& p = model->particles;
    auto& cap = model->capacities;
    const uint32_t vertices = cap.vbd_vertices_per_env;
    const uint32_t field = cap.mf_vertices_per_env, first = vertices - field;
    p.vbd_incidence_offsets.clear();
    p.vbd_incidence.clear();
    p.vbd_color_vertices.clear();
    p.vbd_color_segments.clear();
    p.mf_incidence_offsets.clear();
    p.mf_incidence.clear();
    p.mf_incidence_anchor.clear();
    p.mf_color.clear();
    p.surface_corner_source.clear();
    cap.vbd_elements_per_env = static_cast<uint32_t>(p.vbd_elements.size());
    cap.mf_elements_per_env = static_cast<uint32_t>(p.mf_elements.size());
    cap.mf_materials = static_cast<uint32_t>(p.mf_materials.size());
    uint64_t state_end = 0u, trial_end = 0u;
    for (MfElement& e : p.mf_elements) {
        const uint32_t count = MfElementStateCount(e, p.mf_materials[e.material]);
        if (count == 0u || state_end + count > UINT32_MAX || trial_end + count + kMfTrialLosses > UINT32_MAX)
            throw std::invalid_argument("material-field state exceeds its schema or index range");
        e.state_offset = static_cast<uint32_t>(state_end);
        e.trial_offset = static_cast<uint32_t>(trial_end);
        e.state_count = count;
        e.state_version = kMfStateVersion;
        state_end += count;
        trial_end += count + kMfTrialLosses;
    }
    cap.mf_state_scalars_per_env = static_cast<uint32_t>(state_end);
    cap.mf_trial_scalars_per_env = static_cast<uint32_t>(trial_end);
    cap.vbd_incidence_per_env = 0u;
    cap.mf_incidence_per_env = 0u;
    cap.vbd_dynamic_vertices_per_env = 0u;
    cap.vbd_colors = 0u;
    if (vertices == 0u) return;
    if (uint64_t{cap.vbd_particle_begin} + vertices > p.inv_mass.size() || field > vertices)
        throw std::invalid_argument("vertex-block range exceeds the particle set");

    std::vector<uint32_t> degree(vertices, 0u);
    for (const VbdElement& e : p.vbd_elements)
        for (uint32_t j = 0u; j < VbdElementVertexCount(e.kind); ++j) {
            if (e.vertex[j] >= first)
                throw std::invalid_argument("vertex-block element index exceeds its range");
            ++degree[e.vertex[j]];
        }
    for (const MfElement& e : p.mf_elements)
        for (uint32_t j = 0u; j < MfElementVertexCount(e); ++j) {
            if (e.vertex[j] < first || e.vertex[j] >= vertices)
                throw std::invalid_argument("material-field element leaves the field vertices");
            ++degree[e.vertex[j]];
        }
    p.vbd_incidence_offsets.assign(vertices + 1u, 0u);
    for (uint32_t v = 0u; v < vertices; ++v)
        p.vbd_incidence_offsets[v + 1u] = p.vbd_incidence_offsets[v] + (v < first ? degree[v] : 0u);
    p.vbd_incidence.resize(p.vbd_incidence_offsets.back());
    std::vector<uint32_t> cursor(p.vbd_incidence_offsets.begin(), p.vbd_incidence_offsets.end() - 1);
    for (uint32_t element = 0u; element < p.vbd_elements.size(); ++element) {
        const VbdElement& e = p.vbd_elements[element];
        for (uint32_t j = 0u; j < VbdElementVertexCount(e.kind); ++j)
            p.vbd_incidence[cursor[e.vertex[j]]++] = PackVbdIncidence(element, j);
    }
    if (field > 0u) {
        p.mf_incidence_offsets.assign(field + 1u, 0u);
        for (uint32_t v = 0u; v < field; ++v)
            p.mf_incidence_offsets[v + 1u] = p.mf_incidence_offsets[v] + degree[first + v];
        p.mf_incidence.resize(p.mf_incidence_offsets.back());
        std::vector<uint32_t> field_cursor(p.mf_incidence_offsets.begin(), p.mf_incidence_offsets.end() - 1);
        for (uint32_t element = 0u; element < p.mf_elements.size(); ++element) {
            const MfElement& e = p.mf_elements[element];
            for (uint32_t j = 0u; j < MfElementVertexCount(e); ++j)
                p.mf_incidence[field_cursor[e.vertex[j] - first]++] = PackMfIncidence(element, j);
        }
    }
    // A split vertex's incidences follow the triangle on their side: a bend's own corners its face, its far
    // face vertices the face across that hinge, a seam's sides the faces of its two bends.
    if (cap.mf_spares_per_env > 0u) {
        std::map<std::array<uint32_t, 3>, uint32_t> face_of;
        for (uint32_t element = 0u; element < p.mf_elements.size(); ++element) {
            const MfElement& e = p.mf_elements[element];
            if (e.kind == kMfTriangle) face_of[{e.vertex[0], e.vertex[1], e.vertex[2]}] = element;
        }
        const auto face = [&](uint32_t bend) {
            const MfElement& e = p.mf_elements[bend];
            return face_of.at({e.vertex[0], e.vertex[1], e.vertex[2]});
        };
        p.mf_incidence_anchor.resize(p.mf_incidence.size());
        for (uint32_t v = 0u; v < field; ++v)
            for (uint32_t i = p.mf_incidence_offsets[v]; i < p.mf_incidence_offsets[v + 1u]; ++i) {
                const uint32_t element = MfIncidenceElement(p.mf_incidence[i]);
                const uint32_t local = MfIncidenceLocal(p.mf_incidence[i]);
                const MfElement& e = p.mf_elements[element];
                uint32_t side = element;
                if (e.kind == kMfSeam) side = face(e.vertex[4u + (local & 1u)]);
                else if (e.kind == kMfBend) side = face(element);
                if (e.kind == kMfBend && local >= 3u) {
                    uint32_t hinge = 0u;
                    for (uint32_t far = (local - 3u) / 3u; ((e.hinges >> hinge) & 1u) == 0u || far-- > 0u;) ++hinge;
                    const MfElement& seam = p.mf_elements[p.mf_elements[side].vertex[3u + hinge]];
                    side = face(seam.vertex[4] == element ? seam.vertex[5] : seam.vertex[4]);
                }
                const MfElement& t = p.mf_elements[side];
                const uint32_t slot = t.vertex[0] == first + v ? 0u : t.vertex[1] == first + v ? 1u : 2u;
                p.mf_incidence_anchor[i] = PackMfIncidence(side, slot);
            }
    }

    // Highest degree first, ties by index; each vertex takes the lowest color no neighbor of either medium holds.
    std::vector<uint32_t> order;
    for (uint32_t v = 0u; v < vertices; ++v)
        if (p.inv_mass[cap.vbd_particle_begin + v] > 0.0f) order.push_back(v);
    std::stable_sort(order.begin(), order.end(),
                     [&](uint32_t a, uint32_t b) { return degree[a] > degree[b]; });
    std::vector<uint32_t> color(vertices, kVbdNoColor);
    std::vector<uint32_t> taken;
    uint32_t colors = 0u;
    for (uint32_t v : order) {
        taken.assign(colors + 1u, 0u);
        if (v < first)
            for (uint32_t i = p.vbd_incidence_offsets[v]; i < p.vbd_incidence_offsets[v + 1u]; ++i) {
                const VbdElement& e = p.vbd_elements[VbdIncidenceElement(p.vbd_incidence[i])];
                for (uint32_t j = 0u; j < VbdElementVertexCount(e.kind); ++j)
                    if (color[e.vertex[j]] != kVbdNoColor) taken[color[e.vertex[j]]] = 1u;
            }
        else
            for (uint32_t i = p.mf_incidence_offsets[v - first]; i < p.mf_incidence_offsets[v - first + 1u]; ++i) {
                const MfElement& e = p.mf_elements[MfIncidenceElement(p.mf_incidence[i])];
                for (uint32_t j = 0u; j < MfElementVertexCount(e); ++j)
                    if (color[e.vertex[j]] != kVbdNoColor) taken[color[e.vertex[j]]] = 1u;
            }
        uint32_t chosen = 0u;
        while (taken[chosen] != 0u) ++chosen;
        color[v] = chosen;
        colors = std::max(colors, chosen + 1u);
    }
    // Spares cycle through colors of their own past the cloth's, so a split side clashes only with nearby spares.
    // Each collision corner follows the field triangle corner on its particle.
    if (cap.mf_spares_per_env > 0u) {
        constexpr uint32_t kSpareColors = 8u;
        if (colors + kSpareColors > 256u)
            throw std::invalid_argument("a fracturing field needs at most 248 vertex colors");
        for (uint32_t s = 0u; s < cap.mf_spares_per_env; ++s)
            color[cap.mf_spare_first + s] = colors + s % kSpareColors;
        colors += kSpareColors;
        p.mf_color.assign(color.begin() + first, color.end());
        std::map<std::array<uint32_t, 3>, uint32_t> faces;
        for (uint32_t element = 0u; element < p.mf_elements.size(); ++element) {
            const MfElement& e = p.mf_elements[element];
            if (e.kind != kMfTriangle) continue;
            std::array<uint32_t, 3> key{e.vertex[0], e.vertex[1], e.vertex[2]};
            std::sort(key.begin(), key.end());
            faces[key] = element;
        }
        p.surface_corner_source.assign(p.surface_triangles.size(), ~0u);
        for (const collision::MeshSurfaceInfo& info : p.surface_info)
            for (uint32_t t = info.triangle_offset; t < info.triangle_offset + info.triangle_count; ++t) {
                std::array<uint32_t, 3> key{};
                for (uint32_t k = 0u; k < 3u; ++k)
                    key[k] = info.vertex_offset + p.surface_triangles[size_t{t} * 3u + k] - cap.vbd_particle_begin;
                const std::array<uint32_t, 3> corners = key;
                std::sort(key.begin(), key.end());
                const auto found = faces.find(key);
                if (found == faces.end()) continue;
                for (uint32_t k = 0u; k < 3u; ++k)
                    for (uint32_t j = 0u; j < 3u; ++j)
                        if (p.mf_elements[found->second].vertex[j] == corners[k])
                            p.surface_corner_source[size_t{t} * 3u + k] = found->second * kMfMaxVertices + j;
            }
    }
    for (uint32_t c = 0u; c < colors; ++c) {
        p.vbd_color_segments.push_back(static_cast<uint32_t>(p.vbd_color_vertices.size()));
        for (uint32_t v = 0u; v < vertices; ++v)
            if (color[v] == c) p.vbd_color_vertices.push_back(v);
        p.vbd_color_segments.push_back(static_cast<uint32_t>(p.vbd_color_vertices.size()) -
                                       p.vbd_color_segments.back());
    }
    cap.vbd_incidence_per_env = static_cast<uint32_t>(p.vbd_incidence.size());
    cap.mf_incidence_per_env = static_cast<uint32_t>(p.mf_incidence.size());
    cap.vbd_dynamic_vertices_per_env = static_cast<uint32_t>(p.vbd_color_vertices.size());
    cap.vbd_colors = colors;
    BuildFieldFacePartners(*model);
}

}  // namespace nuka::nk
