#pragma once
// ---------------------------------------------------------------------------
// nuka::constraint::RowBuffers -- host CSR row storage
// ---------------------------------------------------------------------------

#include "constraint/row.hpp"

#include <cstdint>
#include <vector>

namespace nuka::constraint {

struct RowBufferPartition {
    uint32_t row_offset = 0u;
    uint32_t row_count = 0u;
    uint32_t env_id = 0u;
};

struct RowBodyPair {
    uint32_t body_a = kInvalidBodyIndex;
    uint32_t body_b = kInvalidBodyIndex;
};

struct RowBuffers {
    std::vector<Row> rows;
    std::vector<uint32_t> body_indices;
    std::vector<RowJacobian6> jacobian_data;
    std::vector<RowMaterial> materials;
    std::vector<RowAnchor> anchors;
    std::vector<RowBufferPartition> partitions;

    void Clear();
    uint32_t AddRow(Row row,
                    RowBodyPair bodies,
                    const RowJacobian6& jacobian_a,
                    const RowJacobian6& jacobian_b,
                    RowMaterial material = {},
                    RowAnchor anchor = {});

    uint32_t RowCount() const {
        return static_cast<uint32_t>(rows.size());
    }
    uint32_t BodyIndexCount() const {
        return static_cast<uint32_t>(body_indices.size());
    }
    uint32_t JacobianDataCount() const {
        return static_cast<uint32_t>(jacobian_data.size());
    }

    RowBodyPair BodiesForRow(uint32_t row_index) const;
    RowJacobian6 JacobianForRowBody(uint32_t row_index,
                                    uint32_t local_body_index) const;
};

bool IsFrictionRow(const RowBuffers& buffers, uint32_t row_index);
bool IsContactNormalRow(const RowBuffers& buffers, uint32_t row_index);
float TotalNormalLambda(const RowBuffers& buffers, uint32_t row_index);

} // namespace nuka::constraint
