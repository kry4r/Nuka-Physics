#include <gtest/gtest.h>
#include "core/version.hpp"

TEST(BuildSmoke, EngineVersionStringPresent) {
    EXPECT_STREQ(nuka::core::EngineVersion(), "0.1.0-dev");
}
