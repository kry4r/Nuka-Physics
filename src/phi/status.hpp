#pragma once

#include <cstdint>

namespace nuka::phi {
enum class Status : uint8_t { Ok, Unsupported, Failed, OutOfMemory, InvalidArgument };
}  // namespace nuka::phi
