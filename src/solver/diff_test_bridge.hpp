#pragma once
// ---------------------------------------------------------------------------
// nuka::solver::DiffTestBridge -- validation-only row solver diff bridge
// ---------------------------------------------------------------------------

#include "constraint/row_buffers.hpp"
#include "phi/device_context.hpp"
#include "runtime/rigid/body_state.hpp"
#include "solver/gpu/row_solver.cuh"

#include <cstdint>
#include <vector>

namespace nuka::solver {

class DiffTestBridge {
public:
    struct Divergence {
        uint32_t body_id = 0u;
        float position_error = 0.0f;
        float velocity_error = 0.0f;
    };

    explicit DiffTestBridge(const phi::DeviceContext& context);

    gpu::RowSolveReport SolveAndCompare(const constraint::RowBuffers& input_rows,
                                         std::vector<runtime::rigid::BodyState>& bodies,
                                         const gpu::RowSolveConfig& config = {},
                                         float tolerance_position = 1.0e-4f,
                                         float tolerance_velocity = 1.0e-4f);

    const std::vector<Divergence>& LastDivergence() const {
        return last_divergence_;
    }

private:
    phi::DeviceContext context_;
    std::vector<Divergence> last_divergence_;
};

} // namespace nuka::solver
