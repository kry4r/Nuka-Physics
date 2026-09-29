#pragma once

namespace nuka::nk {

class Model;

// Incidence lists and a greedy distance-1 coloring of the dynamic vertex blocks over shared elements.
// Reads the cooked elements, vertex range and inverse masses; fills the tables and their capacities.
void BuildVertexBlockSchedule(Model* model);

}  // namespace nuka::nk
