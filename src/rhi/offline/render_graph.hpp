#pragma once
// ---------------------------------------------------------------------------
// nuka::rhi::offline::RenderGraph -- the ordered pass list a render runs. A
// RenderProfile selects the passes as DATA (a Beauty profile -> {BeautyShade});
// richer profiles append more passes over the SAME shared core. Executing the
// graph runs each pass in order against one PassContext.
//
// HOST-ONLY / CUDA-FREE: composes render_pass stages (opaque backend handles),
// no CUDA token.
// ---------------------------------------------------------------------------

#include "rhi/offline/render_pass.hpp"
#include "rhi/offline/render_profile.hpp"

#include <memory>
#include <vector>

namespace nuka::rhi::offline {

// An ordered list of passes plus the run loop. Build it from a profile, then
// Execute it once per frame.
class RenderGraph {
public:
    // Append a pass to the end of the run order.
    void AddPass(std::unique_ptr<RenderPass> pass) {
        passes_.push_back(std::move(pass));
    }

    uint32_t PassCount() const { return static_cast<uint32_t>(passes_.size()); }

    // Run every pass in order against `ctx` (each pass writes ctx.frame).
    void Execute(const PassContext& ctx) const {
        for (const std::unique_ptr<RenderPass>& pass : passes_) {
            pass->Execute(ctx);
        }
    }

    // Build the pass list a profile selects. Beauty -> the beauty shade pass.
    // The Sensor output is a declared seam (the batched AOV path fills it); it
    // is not produced here, so this returns an empty graph the renderer rejects
    // rather than silently emitting nothing.
    static RenderGraph FromProfile(const RenderProfile& profile) {
        RenderGraph graph;
        if (profile.output == RenderOutput::Beauty) {
            graph.AddPass(std::make_unique<BeautyShadePass>());
        }
        return graph;
    }

private:
    std::vector<std::unique_ptr<RenderPass>> passes_;
};

}  // namespace nuka::rhi::offline
