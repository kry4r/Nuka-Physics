#pragma once

#include <cstdint>
#include <cstring>

namespace nuka::nk {

inline constexpr uint32_t kContactProfileSchemaVersion = 1u;
inline constexpr uint32_t kContactProfileFlagRefsafe = 1u << 0;
inline constexpr uint32_t kContactProfileSupportedFlags = kContactProfileFlagRefsafe;

enum ContactProfileLane : uint32_t {
    kContactProfileSchema = 0u,
    kContactProfileFlags = 1u,
    kContactProfileMu1 = 2u,
    kContactProfileMu2 = 3u,
    kContactProfileSolref0 = 4u,
    kContactProfileSolref1 = 5u,
    kContactProfileSolimp0 = 6u,
    kContactProfileSolimp1 = 7u,
    kContactProfileSolimp2 = 8u,
    kContactProfileSolimp3 = 9u,
    kContactProfileSolimp4 = 10u,
    kContactProfileSolmix = 11u,
    kContactProfileMargin = 12u,
    kContactProfileGap = 13u,
    kContactProfilePriority = 14u,
    kContactProfileCondim = 15u,
    kContactProfileWordCount = 16u,
};

struct ContactProfileV1 {
    uint32_t schema_version = kContactProfileSchemaVersion;
    uint32_t flags = kContactProfileFlagRefsafe;
    float mu1 = 1.0f;
    float mu2 = 1.0f;
    float solref[2] = {0.02f, 1.0f};
    float solimp[5] = {0.9f, 0.95f, 0.001f, 0.5f, 2.0f};
    float solmix = 1.0f;
    float margin = 0.0f;
    float gap = 0.0f;
    int32_t priority = 0;
    uint32_t condim = 3u;
};

static_assert(sizeof(ContactProfileV1) == kContactProfileWordCount * sizeof(uint32_t),
              "ContactProfileV1 must be exactly 16 packed 32-bit words");

inline uint32_t ContactProfileFloatBits(float value) {
    uint32_t bits = 0u;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

inline float ContactProfileBitsFloat(uint32_t bits) {
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

inline void PackContactProfile(const ContactProfileV1& profile, float* words) {
    uint32_t raw[kContactProfileWordCount] = {};
    raw[kContactProfileSchema] = profile.schema_version;
    raw[kContactProfileFlags] = profile.flags;
    raw[kContactProfileMu1] = ContactProfileFloatBits(profile.mu1);
    raw[kContactProfileMu2] = ContactProfileFloatBits(profile.mu2);
    raw[kContactProfileSolref0] = ContactProfileFloatBits(profile.solref[0]);
    raw[kContactProfileSolref1] = ContactProfileFloatBits(profile.solref[1]);
    for (uint32_t i = 0u; i < 5u; ++i)
        raw[kContactProfileSolimp0 + i] = ContactProfileFloatBits(profile.solimp[i]);
    raw[kContactProfileSolmix] = ContactProfileFloatBits(profile.solmix);
    raw[kContactProfileMargin] = ContactProfileFloatBits(profile.margin);
    raw[kContactProfileGap] = ContactProfileFloatBits(profile.gap);
    std::memcpy(&raw[kContactProfilePriority], &profile.priority, sizeof(uint32_t));
    raw[kContactProfileCondim] = profile.condim;
    std::memcpy(words, raw, sizeof(raw));
}

inline ContactProfileV1 UnpackContactProfile(const float* words) {
    uint32_t raw[kContactProfileWordCount] = {};
    std::memcpy(raw, words, sizeof(raw));
    ContactProfileV1 profile;
    profile.schema_version = raw[kContactProfileSchema];
    profile.flags = raw[kContactProfileFlags];
    profile.mu1 = ContactProfileBitsFloat(raw[kContactProfileMu1]);
    profile.mu2 = ContactProfileBitsFloat(raw[kContactProfileMu2]);
    profile.solref[0] = ContactProfileBitsFloat(raw[kContactProfileSolref0]);
    profile.solref[1] = ContactProfileBitsFloat(raw[kContactProfileSolref1]);
    for (uint32_t i = 0u; i < 5u; ++i)
        profile.solimp[i] = ContactProfileBitsFloat(raw[kContactProfileSolimp0 + i]);
    profile.solmix = ContactProfileBitsFloat(raw[kContactProfileSolmix]);
    profile.margin = ContactProfileBitsFloat(raw[kContactProfileMargin]);
    profile.gap = ContactProfileBitsFloat(raw[kContactProfileGap]);
    std::memcpy(&profile.priority, &raw[kContactProfilePriority], sizeof(uint32_t));
    profile.condim = raw[kContactProfileCondim];
    return profile;
}

}  // namespace nuka::nk
