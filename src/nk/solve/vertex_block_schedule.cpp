#include "nk/solve/vertex_block_schedule.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "nk/model/model.hpp"

namespace nuka::nk {

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
    cap.vbd_elements_per_env = static_cast<uint32_t>(p.vbd_elements.size());
    cap.mf_elements_per_env = static_cast<uint32_t>(p.mf_elements.size());
    cap.mf_materials = static_cast<uint32_t>(p.mf_materials.size());
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
}

}  // namespace nuka::nk
