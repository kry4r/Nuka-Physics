// ---------------------------------------------------------------------------
// nuka::diffsim -- host-side helpers for the single-step backward
// ---------------------------------------------------------------------------
//
// BuildSpatialInertiaMassJacobian: the per-link d(I[36])/d(mass) slope, built
// from the SAME MakeSpatialInertia mapping the engine forward uses. This is the
// representation-consistency guarantee the mass system-ID demo rides on: when the
// FD test perturbs the scalar mass it calls MakeSpatialInertia(mass+delta, ...),
// and the analytic adjoint contracts grad_I against THIS slope -- so the
// parallel-axis + COM coupling (top-left (|c|^2 I - c c^T), off-diagonal +-[c]x,
// bottom-right I3) is differentiated EXACTLY as the forward builds it.
//
// MakeSpatialInertia is exactly AFFINE in mass for fixed (diagonal_inertia,
// inertial_frame): the rotated COM inertia term (top-left) is mass-independent;
// every other term is mass * constant. So the slope is independent of the mass
// value: dI/dmass = MakeSpatialInertia(m+1) - MakeSpatialInertia(m). We evaluate
// it at the link's actual mass (m and m+1 both > 0 so the mass<=0 guard is
// inactive) to stay numerically identical to the engine's float path.
// ---------------------------------------------------------------------------

#include "diffsim/step_backward.hpp"

#include "runtime/articulation/articulation_state.hpp"

namespace nuka::diffsim {

std::vector<float> BuildSpatialInertiaMassJacobian(
    const std::vector<LinkMassParams>& links) {
    std::vector<float> out(links.size() * 36u, 0.0f);
    for (size_t i = 0u; i < links.size(); ++i) {
        const LinkMassParams& lp = links[i];
        // Affine in mass: slope = I(m+1) - I(m), independent of m. Use m and m+1
        // (both > 0) so the engine's mass<=0 guard never trips.
        const float base_mass = (lp.mass > 0.0f) ? lp.mass : 1.0f;
        const articulation::LinkSpatialInertia i_lo =
            articulation::MakeSpatialInertia(base_mass, lp.diagonal_inertia,
                                             lp.inertial_frame);
        const articulation::LinkSpatialInertia i_hi =
            articulation::MakeSpatialInertia(base_mass + 1.0f, lp.diagonal_inertia,
                                             lp.inertial_frame);
        for (uint32_t k = 0u; k < 36u; ++k) {
            out[i * 36u + k] = i_hi.I[k] - i_lo.I[k];
        }
    }
    return out;
}

}  // namespace nuka::diffsim
