// ---------------------------------------------------------------------------
// nuka::constraint::RowBuffers implementation
// ---------------------------------------------------------------------------

#include "constraint/row_buffers.hpp"

#include <algorithm>

namespace nuka::constraint {

void RowBuffers::Clear() {
    rows.clear();
    body_indices.clear();
    jacobian_data.clear();
    materials.clear();
    anchors.clear();
    partitions.clear();
}

uint32_t RowBuffers::AddRow(Row row,
                            RowBodyPair bodies,
                            const RowJacobian6& jacobian_a,
                            const RowJacobian6& jacobian_b,
                            RowMaterial material,
                            RowAnchor anchor) {
    row.body_count = 2u;
    row.body_list_offset = static_cast<uint32_t>(body_indices.size());
    row.jacobian_offset = static_cast<uint32_t>(jacobian_data.size());

    body_indices.push_back(bodies.body_a);
    body_indices.push_back(bodies.body_b);
    jacobian_data.push_back(jacobian_a);
    jacobian_data.push_back(jacobian_b);

    const uint32_t row_index = static_cast<uint32_t>(rows.size());
    rows.push_back(row);
    materials.push_back(material);
    anchors.push_back(anchor);
    return row_index;
}

RowBodyPair RowBuffers::BodiesForRow(uint32_t row_index) const {
    if (row_index >= rows.size()) {
        return {};
    }
    const Row& row = rows[row_index];
    RowBodyPair pair;
    if (row.body_count > 0u &&
        row.body_list_offset < body_indices.size()) {
        pair.body_a = body_indices[row.body_list_offset];
    }
    if (row.body_count > 1u &&
        row.body_list_offset + 1u < body_indices.size()) {
        pair.body_b = body_indices[row.body_list_offset + 1u];
    }
    return pair;
}

RowJacobian6 RowBuffers::JacobianForRowBody(uint32_t row_index,
                                            uint32_t local_body_index) const {
    if (row_index >= rows.size()) {
        return {};
    }
    const Row& row = rows[row_index];
    const uint32_t index = row.jacobian_offset + local_body_index;
    if (local_body_index >= row.body_count || index >= jacobian_data.size()) {
        return {};
    }
    return jacobian_data[index];
}

bool IsFrictionRow(const RowBuffers& buffers, uint32_t row_index) {
    if (row_index >= buffers.materials.size()) {
        return false;
    }
    const RowMaterial& material = buffers.materials[row_index];
    return material.kind == RowKind::Contact &&
           material.friction_row_count > 0u &&
           row_index >= material.first_friction_row &&
           row_index < material.first_friction_row + material.friction_row_count;
}

bool IsContactNormalRow(const RowBuffers& buffers, uint32_t row_index) {
    if (row_index >= buffers.materials.size()) {
        return false;
    }
    const RowMaterial& material = buffers.materials[row_index];
    return material.kind == RowKind::Contact &&
           row_index >= material.group_id &&
           row_index < material.group_id + std::max(material.normal_row_count, 1u);
}

float TotalNormalLambda(const RowBuffers& buffers, uint32_t row_index) {
    if (row_index >= buffers.materials.size()) {
        return 0.0f;
    }
    const RowMaterial& material = buffers.materials[row_index];
    if (material.kind != RowKind::Contact || material.normal_row_count == 0u) {
        return 0.0f;
    }

    float total = 0.0f;
    const uint32_t begin = material.group_id;
    const uint32_t end = std::min<uint32_t>(
        begin + material.normal_row_count,
        static_cast<uint32_t>(buffers.rows.size()));
    for (uint32_t row = begin; row < end; ++row) {
        total += std::max(buffers.rows[row].lambda, 0.0f);
    }
    return total;
}

} // namespace nuka::constraint
