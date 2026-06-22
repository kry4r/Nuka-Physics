// ---------------------------------------------------------------------------
// nk::XpbdColoring — graph coloring of the XPBD constraint families for parallel
// Gauss-Seidel. A general solver property (no per-scene branch): each family's
// constraints are partitioned into COLORS (independent sets sharing no particle)
// so a color's constraints project in parallel without write races, while colors
// run in fixed order — deterministic, no float atomics. Built at World construct
// from the cooked single-env constraint SoA, BEFORE Model::UploadTo.
// ---------------------------------------------------------------------------

#pragma once

namespace nuka::nk {

struct Model;

class XpbdColoring {
public:
    // Reorder each XPBD family's single-env constraint SoA into color-contiguous
    // order and fill the per-family {offset,count}-per-color segment tables +
    // color-count capacities on the Model. No-op when a family is empty.
    static void Build(Model* model);
};

}  // namespace nuka::nk
