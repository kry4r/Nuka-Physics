#pragma once
// ---------------------------------------------------------------------------
// nuka::constraint::row_builder -- contact/joint/drive to CSR rows
// ---------------------------------------------------------------------------

#include "constraint/contact_manifold.hpp"
#include "constraint/row_buffers.hpp"
#include "math/transform.hpp"

#include <span>
#include <vector>

namespace nuka::constraint {

uint32_t BuildContactRows(uint32_t contact_points, RowBuffers* out_rows);
void BuildContactRows(const std::vector<ContactManifold>& manifolds,
                      RowBuffers* out_rows);
void BuildContactRows(std::span<const ContactManifold> manifolds,
                      RowBuffers* out_rows);

void AppendRevoluteRows(RowBuffers* out_rows,
                        uint32_t body_a,
                        uint32_t body_b,
                        math::Vec3 axis,
                        math::Transform parent_frame,
                        math::Transform child_frame,
                        float lower_limit,
                        float upper_limit,
                        float current_angle = 0.0f);

void AppendFixedRows(RowBuffers* out_rows,
                     uint32_t body_a,
                     uint32_t body_b,
                     math::Transform parent_frame,
                     math::Transform child_frame);

void AppendDriveRow(RowBuffers* out_rows,
                    uint32_t body_a,
                    uint32_t body_b,
                    math::Vec3 axis,
                    float target_velocity,
                    float max_force);

} // namespace nuka::constraint
