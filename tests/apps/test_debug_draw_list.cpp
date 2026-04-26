// ---------------------------------------------------------------------------
// Tests for nuka::app::DebugDrawList
// ---------------------------------------------------------------------------

#include "apps/debug_shell/debug_draw.hpp"

#include <gtest/gtest.h>

using namespace nuka::app;
using namespace nuka::math;
using namespace nuka::collision;

// ---------------------------------------------------------------------------
// Contact overlay command count
// ---------------------------------------------------------------------------

TEST(DebugDraw, ProducesContactOverlayCommands) {
    const auto commands = BuildContactOverlayCommandCount(4);
    EXPECT_EQ(commands, 4u);
}

// ---------------------------------------------------------------------------
// AddLine produces a line command
// ---------------------------------------------------------------------------

TEST(DebugDraw, AddLineCommand) {
    DebugDrawList list;
    list.AddLine({0, 0, 0}, {1, 1, 1});
    ASSERT_EQ(list.CommandCount(), 1u);
    EXPECT_EQ(list.Commands()[0].type, DrawCommandType::Line);
}

// ---------------------------------------------------------------------------
// AddSphere produces a sphere command
// ---------------------------------------------------------------------------

TEST(DebugDraw, AddSphereCommand) {
    DebugDrawList list;
    list.AddSphere({1, 2, 3}, 0.5f);
    ASSERT_EQ(list.CommandCount(), 1u);
    EXPECT_EQ(list.Commands()[0].type, DrawCommandType::Sphere);
    EXPECT_FLOAT_EQ(list.Commands()[0].radius, 0.5f);
}

// ---------------------------------------------------------------------------
// AddAABB produces an AABB command
// ---------------------------------------------------------------------------

TEST(DebugDraw, AddAABBCommand) {
    AABB box;
    box.min = {-1, -1, -1};
    box.max = { 1,  1,  1};

    DebugDrawList list;
    list.AddAABB(box);
    ASSERT_EQ(list.CommandCount(), 1u);
    EXPECT_EQ(list.Commands()[0].type, DrawCommandType::AABB);
}

// ---------------------------------------------------------------------------
// AddFrame produces 3 line commands (X, Y, Z axes)
// ---------------------------------------------------------------------------

TEST(DebugDraw, AddFrameProducesThreeLines) {
    DebugDrawList list;
    list.AddFrame(Transform::Identity(), 1.0f);
    EXPECT_EQ(list.CommandCount(), 3u);
    for (const auto& cmd : list.Commands()) {
        EXPECT_EQ(cmd.type, DrawCommandType::Line);
    }
}

// ---------------------------------------------------------------------------
// Clear resets the command list
// ---------------------------------------------------------------------------

TEST(DebugDraw, ClearResetsCommands) {
    DebugDrawList list;
    list.AddLine({0, 0, 0}, {1, 0, 0});
    list.AddSphere({0, 0, 0}, 1.0f);
    ASSERT_EQ(list.CommandCount(), 2u);

    list.Clear();
    EXPECT_EQ(list.CommandCount(), 0u);
    EXPECT_TRUE(list.Commands().empty());
}
