#pragma once
// ---------------------------------------------------------------------------
// PHI v2 CUDA backend — explicit op-registration entry points (M3b).
//
// One function per op TU, all EXPLICITLY called from
// RegisterNkArticulationPipelineOps() (articulation.cu), which itself is
// called from cuda_backend.cu::RegisterCudaBackendEntry(). Explicit calls —
// not static initializers — so the static-lib linker can never drop an op TU
// (the same de-risk as the CUDA registry entry itself).
// ---------------------------------------------------------------------------

namespace nuka::phi {

// Umbrella (articulation.cu): registers the whole M3b articulation pipeline.
void RegisterNkArticulationPipelineOps();

// Per-TU registrars.
void RegisterNkAbaOps();              // articulation.cu
void RegisterNkCrbaOps();             // crba.cu
void RegisterNkContactsFootOps();     // contacts_foot.cu (+contacts_union.cu)
void RegisterNkDogDogContactOps();    // dog_dog_contact.cu (WP5/6/8 multi-artic)
void RegisterNkSyncBodyPoseOps();     // sync_body_pose.cu (general contact B2)
void RegisterNkBroadphaseOps();       // broadphase.cu (M5)
void RegisterNkNarrowphaseSdfOps();   // narrowphase_sdf.cu (M5)
void RegisterNkAssembleRowsOps();     // assemble_rows.cu (M4)
void RegisterNkSolveRowsOps();        // solve_rows.cu (M4)
void RegisterNkParticleOps();         // particles.cu (M6)
void RegisterNkReadoutOps();          // readout.cu
void RegisterNkDiffsimBackwardOps();  // diffsim_backward.cu (M9 T7)

} // namespace nuka::phi
