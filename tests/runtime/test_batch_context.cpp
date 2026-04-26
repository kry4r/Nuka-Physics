// ---------------------------------------------------------------------------
// Tests for nuka::runtime::BatchScheduler
// ---------------------------------------------------------------------------

#include "runtime/batch_scheduler.hpp"

#include <gtest/gtest.h>

using namespace nuka::runtime;

// ---------------------------------------------------------------------------
// Template reuse across instances
// ---------------------------------------------------------------------------

TEST(BatchContext, SupportsTemplateReuseAcrossInstances) {
    auto batch = BuildTestBatch(3);
    EXPECT_EQ(batch.instance_count, 3u);
    EXPECT_EQ(batch.template_count, 1u);
}

// ---------------------------------------------------------------------------
// Single instance
// ---------------------------------------------------------------------------

TEST(BatchContext, SingleInstance) {
    auto batch = BuildTestBatch(1);
    EXPECT_EQ(batch.instance_count, 1u);
    EXPECT_EQ(batch.template_count, 1u);
}

// ---------------------------------------------------------------------------
// Scheduler configure and status
// ---------------------------------------------------------------------------

TEST(BatchContext, SchedulerConfigAndStatus) {
    BatchScheduler scheduler;
    BatchConfig config;
    config.instance_count = 5;
    config.template_reuse = true;

    scheduler.Configure(config);

    EXPECT_EQ(scheduler.InstanceCount(), 5u);

    auto status = scheduler.GetStatus();
    EXPECT_EQ(status.instance_count, 5u);
    EXPECT_EQ(status.template_count, 1u);
}

// ---------------------------------------------------------------------------
// Default state
// ---------------------------------------------------------------------------

TEST(BatchContext, DefaultState) {
    BatchScheduler scheduler;
    EXPECT_EQ(scheduler.InstanceCount(), 1u);

    auto status = scheduler.GetStatus();
    EXPECT_EQ(status.instance_count, 1u);
    EXPECT_EQ(status.template_count, 1u);
}
