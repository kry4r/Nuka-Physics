// ---------------------------------------------------------------------------
// p01-F T1 -- batched N-articulation replication
//
// Proves that ReplicateArticulationHostState tiles one cooked Go2 articulation
// into N independent copies that the per-articulation Featherstone ABA kernels
// step bit-identically to the single-env reference. This is the foundation for
// stepping thousands of Go2 articulations in parallel on the GPU.
// ---------------------------------------------------------------------------

#include "import/usd_importer.hpp"
#include "phi/scoped_device_guard.hpp"
#include <cuda_runtime.h>
#include "runtime/articulation/articulation_state.hpp"
#include "runtime/articulation/featherstone_aba.hpp"
#include "runtime/world_builder.hpp"
#include "scene/cooker.hpp"
#include "scene/scene_ir.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <string>
#include <vector>

namespace {

namespace articulation = nuka::runtime::articulation;

std::filesystem::path SourcePath(const char* relative_path) {
    return std::filesystem::path(NUKA_SOURCE_DIR) / relative_path;
}

// Cooks a single Go2 articulation host state via the same internal cook path the
// C ABI uses (LoadUsd -> CookScene -> BuildWorld -> BuildArticulationHostState).
articulation::ArticulationHostState CookGo2HostState() {
    const auto scene_path = SourcePath("examples/scenes/go2_stand.usda");
    const auto scene = nuka::import::LoadUsd(scene_path.string());
    const auto blob = nuka::scene::CookScene(scene);
    const auto world = nuka::runtime::BuildWorld(blob);
    return articulation::BuildArticulationHostState(world.template_view.articulations,
                                                    world.template_view.body_table);
}

// Steps the supplied device state through the existing ABA acceleration +
// integration sequence for `steps` iterations. NOTE: ApplyPositionDrives is
// intentionally omitted -- the drive-target buffers live outside
// ArticulationHostState and are not part of the replication contract, so they
// are not replicated here. Gravity (plus any seeded tau) drives the dynamics,
// which is sufficient to exercise the per-articulation grid kernels and prove
// replication + determinism.
void StepAba(cudaStream_t context, int context_dev,
             articulation::ArticulationDeviceBuffers& device_state,
             uint32_t steps,
             float gravity_z,
             float dt) {
    for (uint32_t step = 0u; step < steps; ++step) {
        articulation::FeatherstoneAba::ComputeAccelerations(context, context_dev,
                                                            device_state.View(),
                                                            gravity_z);
        articulation::FeatherstoneAba::Integrate(context, context_dev, device_state.View(), dt);
    }
    cudaStreamSynchronize(context);
}

} // namespace

TEST(BatchedArticulationReplication, NReplicasStepBitIdenticalToSingleEnv) {
    const auto scene_path = SourcePath("examples/scenes/go2_stand.usda");
    if (!std::filesystem::exists(scene_path)) {
        GTEST_SKIP() << "Go2 stand scene is not available";
    }

    constexpr float kGravityZ = -9.81f;
    constexpr float kDt = 1.0f / 240.0f;
    constexpr uint32_t kSteps = 100u;
    constexpr uint32_t kEnvCount = 4096u;

    // --- Cook a pristine single-env base ------------------------------------
    auto base = CookGo2HostState();
    const uint32_t base_link_count = base.TotalLinkCount();
    const uint32_t base_articulation_count = base.ArticulationCount();
    ASSERT_GT(base_link_count, 0u);
    ASSERT_GT(base_articulation_count, 0u);
    // BuildArticulationHostState skips zero-link articulations, so the compacted
    // count/offset arrays could in principle be shorter than the topology vector.
    // For Go2 (one clean articulation) they coincide, which is what the spot-check
    // loop below assumes when it indexes by env * base_articulation_count.
    ASSERT_EQ(base.articulations.size(), static_cast<size_t>(base_articulation_count));

    // Seed a non-zero torque on a couple of non-fixed joints so the trajectory is
    // guaranteed non-trivial even if gravity alone were nearly static. tau is part
    // of the host state, so it tiles automatically and -- with drives omitted --
    // stays constant across the run.
    for (uint32_t link = 0u; link < base_link_count; ++link) {
        if (base.joint_type[link] != articulation::ArticulationJointType::Fixed) {
            base.tau[link] = 0.5f;
        }
    }
    const std::vector<float> base_initial_q = base.q;

    const cudaStream_t context = nullptr;  // BUF-14: stream 0
    const int context_dev = 0;

    // --- Single-env reference run -------------------------------------------
    auto single_device = articulation::UploadArticulationState(context, context_dev, base);
    StepAba(context, context_dev, single_device, kSteps, kGravityZ, kDt);
    articulation::ArticulationHostState reference;
    articulation::DownloadArticulationState(single_device, &reference);
    ASSERT_EQ(reference.q.size(), base.q.size());

    // Guard against a vacuous pass: the reference q must have actually moved,
    // otherwise we would only be comparing initial-to-initial state.
    EXPECT_NE(reference.q, base_initial_q)
        << "reference trajectory is static; replication test would be vacuous";

    // --- Replicate from the pristine base, then step the batch --------------
    auto batched = articulation::ReplicateArticulationHostState(base, kEnvCount);

    // Counts must scale exactly by N.
    ASSERT_EQ(batched.TotalLinkCount(), base_link_count * kEnvCount);
    ASSERT_EQ(batched.ArticulationCount(), base_articulation_count * kEnvCount);

    // Spot-check the offset arithmetic on the host structure before stepping.
    for (uint32_t env = 0u; env < kEnvCount; env += 1023u) {
        const uint32_t artic = env * base_articulation_count;
        ASSERT_LT(artic, batched.articulation_link_offset.size());
        EXPECT_EQ(batched.articulation_link_offset[artic],
                  base.articulation_link_offset[0] + env * base_link_count);
        const uint32_t link = env * base_link_count;
        ASSERT_LT(link, batched.link_to_articulation.size());
        EXPECT_EQ(batched.link_to_articulation[link],
                  base.link_to_articulation[0] + env * base_articulation_count);
        // parent_link is articulation-local: stays equal to the base value.
        EXPECT_EQ(batched.parent_link[link], base.parent_link[0]);
    }

    auto batched_device = articulation::UploadArticulationState(context, context_dev, batched);
    StepAba(context, context_dev, batched_device, kSteps, kGravityZ, kDt);
    articulation::ArticulationHostState batched_result;
    articulation::DownloadArticulationState(batched_device, &batched_result);
    ASSERT_EQ(batched_result.q.size(),
              static_cast<size_t>(base.q.size()) * kEnvCount);

    // --- Assert every replica's q sub-vector is bit-identical to the reference
    const size_t dof_per_env = reference.q.size();
    size_t mismatches = 0u;
    for (uint32_t env = 0u; env < kEnvCount; ++env) {
        const size_t offset = static_cast<size_t>(env) * dof_per_env;
        for (size_t dof = 0u; dof < dof_per_env; ++dof) {
            if (batched_result.q[offset + dof] != reference.q[dof]) {
                if (mismatches < 8u) {
                    ADD_FAILURE() << "replica " << env << " dof " << dof
                                  << " replica=" << batched_result.q[offset + dof]
                                  << " reference=" << reference.q[dof];
                }
                ++mismatches;
            }
        }
    }
    EXPECT_EQ(mismatches, 0u)
        << mismatches << " of " << (static_cast<size_t>(kEnvCount) * dof_per_env)
        << " q values diverged across " << kEnvCount << " replicas";
}
