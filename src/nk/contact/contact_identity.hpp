#pragma once

#include "constraint/collidable.hpp"
#include "math/vec3.hpp"

#include <cstdint>
#include <cmath>

#if defined(__CUDACC__)
#define NUKA_CONTACT_HD __host__ __device__
#else
#define NUKA_CONTACT_HD
#endif

namespace nuka::nk {

inline constexpr uint32_t kContactFeatureUnavailable = 0xffffffffu;
inline constexpr uint32_t kContactHandleBits = 28u;
inline constexpr uint32_t kContactHandleMask = (1u << kContactHandleBits) - 1u;
inline constexpr float kContactPointQuantization = 4096.0f;

struct ContactId {
    uint64_t pair = 0u;
    uint64_t feature = 0u;

    NUKA_CONTACT_HD constexpr bool operator==(const ContactId& rhs) const {
        return pair == rhs.pair && feature == rhs.feature;
    }

    NUKA_CONTACT_HD constexpr bool operator!=(const ContactId& rhs) const {
        return !(*this == rhs);
    }
};

NUKA_CONTACT_HD inline bool ContactIdLess(const ContactId& a, const ContactId& b) {
    return a.pair < b.pair || (a.pair == b.pair && a.feature < b.feature);
}

struct CanonicalContactDescriptor {
    constraint::CollidableRef a;
    constraint::CollidableRef b;
    math::Vec3 local_point_a{};
    math::Vec3 local_point_b{};
    math::Vec3 normal{};
    math::Vec3 tangent1{};
    math::Vec3 tangent2{};
    uint32_t feature_a = kContactFeatureUnavailable;
    uint32_t feature_b = kContactFeatureUnavailable;
    uint32_t manifold_slot = 0u;
    uint32_t topology_version = 0u;
    uint32_t material_version = 0u;
};

NUKA_CONTACT_HD inline uint32_t ContactSideKey(const constraint::CollidableRef& ref) {
    return (static_cast<uint32_t>(ref.type) << kContactHandleBits) |
           (ref.handle & kContactHandleMask);
}

NUKA_CONTACT_HD inline uint64_t ContactPairKey(
    const constraint::CollidableRef& a, const constraint::CollidableRef& b) {
    const uint32_t ka = ContactSideKey(a);
    const uint32_t kb = ContactSideKey(b);
    const uint32_t lo = ka < kb ? ka : kb;
    const uint32_t hi = ka < kb ? kb : ka;
    return (static_cast<uint64_t>(lo) << 32u) | static_cast<uint64_t>(hi);
}

NUKA_CONTACT_HD inline void BuildCanonicalTangentBasis(
    const math::Vec3& input_normal, math::Vec3* tangent1, math::Vec3* tangent2) {
    const float length_sq = input_normal.LengthSq();
    if (length_sq <= 1.0e-20f) {
        *tangent1 = math::Vec3::UnitX();
        *tangent2 = math::Vec3::UnitY();
        return;
    }
    const float inv_length = 1.0f / sqrtf(length_sq);
    const math::Vec3 n = input_normal * inv_length;
    const float ax = fabsf(n.x);
    const float ay = fabsf(n.y);
    const float az = fabsf(n.z);
    const math::Vec3 axis = (ax <= ay && ax <= az)
        ? math::Vec3::UnitX()
        : ((ay <= az) ? math::Vec3::UnitY() : math::Vec3::UnitZ());
    const math::Vec3 raw = axis.Cross(n);
    const float inv_tangent_length = 1.0f / sqrtf(raw.LengthSq());
    *tangent1 = raw * inv_tangent_length;
    *tangent2 = n.Cross(*tangent1);
}

NUKA_CONTACT_HD inline CanonicalContactDescriptor CanonicalizeContact(
    CanonicalContactDescriptor descriptor) {
    if (ContactSideKey(descriptor.b) < ContactSideKey(descriptor.a)) {
        const constraint::CollidableRef side = descriptor.a;
        descriptor.a = descriptor.b;
        descriptor.b = side;
        const math::Vec3 point = descriptor.local_point_a;
        descriptor.local_point_a = descriptor.local_point_b;
        descriptor.local_point_b = point;
        const uint32_t feature = descriptor.feature_a;
        descriptor.feature_a = descriptor.feature_b;
        descriptor.feature_b = feature;
        descriptor.normal = -descriptor.normal;
    }
    BuildCanonicalTangentBasis(descriptor.normal,
                               &descriptor.tangent1, &descriptor.tangent2);
    return descriptor;
}

NUKA_CONTACT_HD inline int32_t QuantizeContactCoordinate(float value) {
    const float scaled = value * kContactPointQuantization;
    const float rounded = scaled >= 0.0f ? scaled + 0.5f : scaled - 0.5f;
    return static_cast<int32_t>(rounded);
}

NUKA_CONTACT_HD inline uint64_t ContactHashWord(uint64_t hash, uint64_t word) {
    hash ^= word;
    hash *= 1099511628211ull;
    return hash;
}

NUKA_CONTACT_HD inline uint64_t ContactPointToken(const math::Vec3& point) {
    uint64_t hash = 1469598103934665603ull;
    hash = ContactHashWord(hash, static_cast<uint32_t>(QuantizeContactCoordinate(point.x)));
    hash = ContactHashWord(hash, static_cast<uint32_t>(QuantizeContactCoordinate(point.y)));
    hash = ContactHashWord(hash, static_cast<uint32_t>(QuantizeContactCoordinate(point.z)));
    return hash;
}

NUKA_CONTACT_HD inline uint64_t ContactFeatureToken(
    uint32_t feature, const math::Vec3& fallback_point) {
    if (feature != kContactFeatureUnavailable) {
        return 0x8000000000000000ull | static_cast<uint64_t>(feature);
    }
    return ContactPointToken(fallback_point) & 0x7fffffffffffffffull;
}

NUKA_CONTACT_HD inline ContactId MakeContactId(
    const CanonicalContactDescriptor& input) {
    const CanonicalContactDescriptor descriptor = CanonicalizeContact(input);
    uint64_t feature_hash = 1469598103934665603ull;
    feature_hash = ContactHashWord(feature_hash,
        ContactFeatureToken(descriptor.feature_a, descriptor.local_point_a));
    feature_hash = ContactHashWord(feature_hash,
        ContactFeatureToken(descriptor.feature_b, descriptor.local_point_b));
    feature_hash = ContactHashWord(feature_hash, descriptor.manifold_slot);
    feature_hash = ContactHashWord(feature_hash, descriptor.topology_version);
    feature_hash = ContactHashWord(feature_hash, descriptor.material_version);
    return {ContactPairKey(descriptor.a, descriptor.b), feature_hash};
}

struct ContactMatch {
    uint32_t previous_index = kContactFeatureUnavailable;
    uint64_t cost = ~0ull;
};

NUKA_CONTACT_HD inline bool ContactFeaturesCompatible(
    const CanonicalContactDescriptor& input_a,
    const CanonicalContactDescriptor& input_b) {
    const CanonicalContactDescriptor a = CanonicalizeContact(input_a);
    const CanonicalContactDescriptor b = CanonicalizeContact(input_b);
    const bool side_a_matches =
        a.feature_a == kContactFeatureUnavailable ||
        b.feature_a == kContactFeatureUnavailable ||
        a.feature_a == b.feature_a;
    const bool side_b_matches =
        a.feature_b == kContactFeatureUnavailable ||
        b.feature_b == kContactFeatureUnavailable ||
        a.feature_b == b.feature_b;
    return side_a_matches && side_b_matches;
}

NUKA_CONTACT_HD inline uint64_t ContactMatchCost(
    const CanonicalContactDescriptor& input_a,
    const CanonicalContactDescriptor& input_b) {
    const CanonicalContactDescriptor a = CanonicalizeContact(input_a);
    const CanonicalContactDescriptor b = CanonicalizeContact(input_b);
    const int64_t ax = QuantizeContactCoordinate(a.local_point_a.x);
    const int64_t ay = QuantizeContactCoordinate(a.local_point_a.y);
    const int64_t az = QuantizeContactCoordinate(a.local_point_a.z);
    const int64_t bx = QuantizeContactCoordinate(b.local_point_a.x);
    const int64_t by = QuantizeContactCoordinate(b.local_point_a.y);
    const int64_t bz = QuantizeContactCoordinate(b.local_point_a.z);
    const int64_t dx = ax - bx;
    const int64_t dy = ay - by;
    const int64_t dz = az - bz;
    int64_t normal_dot = static_cast<int64_t>(
        (a.normal.Dot(b.normal) + 1.0f) * 32768.0f);
    normal_dot = normal_dot < 0 ? 0 : (normal_dot > 65536 ? 65536 : normal_dot);
    const uint64_t normal_cost = static_cast<uint64_t>(65536 - normal_dot);
    return static_cast<uint64_t>(dx * dx + dy * dy + dz * dz) + normal_cost;
}

template <uint32_t MaxPoints>
NUKA_CONTACT_HD inline void MatchContactManifolds(
    const CanonicalContactDescriptor* previous, uint32_t previous_count,
    const CanonicalContactDescriptor* current, uint32_t current_count,
    ContactMatch* matches) {
    bool used[MaxPoints] = {};
    const uint32_t pn = previous_count < MaxPoints ? previous_count : MaxPoints;
    const uint32_t cn = current_count < MaxPoints ? current_count : MaxPoints;
    for (uint32_t i = 0u; i < cn; ++i) {
        matches[i] = ContactMatch{};
        const ContactId current_id = MakeContactId(current[i]);
        for (uint32_t j = 0u; j < pn; ++j) {
            if (used[j]) continue;
            const ContactId previous_id = MakeContactId(previous[j]);
            if (current_id.pair != previous_id.pair ||
                !ContactFeaturesCompatible(current[i], previous[j])) continue;
            const uint64_t cost = ContactMatchCost(current[i], previous[j]);
            const bool lower_cost = cost < matches[i].cost;
            const bool same_cost = cost == matches[i].cost;
            const bool lower_id = matches[i].previous_index == kContactFeatureUnavailable ||
                ContactIdLess(previous_id,
                              MakeContactId(previous[matches[i].previous_index]));
            if (lower_cost || (same_cost && lower_id)) {
                matches[i].previous_index = j;
                matches[i].cost = cost;
            }
        }
        if (matches[i].previous_index != kContactFeatureUnavailable) {
            used[matches[i].previous_index] = true;
        }
    }
    for (uint32_t i = cn; i < MaxPoints; ++i) matches[i] = ContactMatch{};
}

}  // namespace nuka::nk

#undef NUKA_CONTACT_HD
