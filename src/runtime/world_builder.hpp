#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::BuildWorld – construct runtime world from cooked blob
// ---------------------------------------------------------------------------

#include "runtime/world_template.hpp"
#include "runtime/world_instance.hpp"
#include "runtime/batch_context.hpp"
#include "scene/cooked_blob.hpp"

namespace nuka::runtime {

struct BuiltWorld {
    WorldTemplate template_view;
    WorldInstance  instance;
    BatchContext   batch;
};

/// Build a single-instance world from a cooked scene blob.
BuiltWorld BuildWorld(const scene::CookedBlob& blob);

} // namespace nuka::runtime
