#pragma once
// Random123 Philox4x32-10 provides reproducible counter-indexed random numbers.
// Poisson sampling uses Knuth inversion below rate 30 and Hoermann transformed rejection above it.

#include <cmath>
#include <cstdint>

#if defined(__CUDACC__)
#define NUKA_RANDOM_HD __host__ __device__ __forceinline__
#else
#define NUKA_RANDOM_HD inline
#endif

namespace nuka::sensor::noise {
constexpr uint32_t kPhiloxM0 = 0xD2511F53u;
constexpr uint32_t kPhiloxM1 = 0xCD9E8D57u;
constexpr uint32_t kPhiloxW0 = 0x9E3779B9u;  // golden ratio (key bump, lane 0)
constexpr uint32_t kPhiloxW1 = 0xBB67AE85u;  // sqrt(3)-1   (key bump, lane 1)
struct Philox4x32Counter {
    uint32_t v[4];
};
struct Philox4x32Key {
    uint32_t v[2];
};
NUKA_RANDOM_HD void MulHiLo(uint32_t a, uint32_t b,
                           uint32_t* hi, uint32_t* lo) {
    const uint64_t product = static_cast<uint64_t>(a) * static_cast<uint64_t>(b);
    *hi = static_cast<uint32_t>(product >> 32);
    *lo = static_cast<uint32_t>(product);
}
NUKA_RANDOM_HD Philox4x32Counter PhiloxRound(
    Philox4x32Counter ctr, Philox4x32Key key) {
    uint32_t hi0, lo0, hi1, lo1;
    MulHiLo(kPhiloxM0, ctr.v[0], &hi0, &lo0);
    MulHiLo(kPhiloxM1, ctr.v[2], &hi1, &lo1);
    Philox4x32Counter out;
    out.v[0] = hi1 ^ ctr.v[1] ^ key.v[0];
    out.v[1] = lo1;
    out.v[2] = hi0 ^ ctr.v[3] ^ key.v[1];
    out.v[3] = lo0;
    return out;
}
NUKA_RANDOM_HD Philox4x32Key PhiloxBumpKey(
    Philox4x32Key key) {
    key.v[0] += kPhiloxW0;
    key.v[1] += kPhiloxW1;
    return key;
}
NUKA_RANDOM_HD Philox4x32Counter Philox4x32_10(
    Philox4x32Counter ctr, Philox4x32Key key) {
    ctr = PhiloxRound(ctr, key);
    for (int round = 1; round < 10; ++round) {
        key = PhiloxBumpKey(key);
        ctr = PhiloxRound(ctr, key);
    }
    return ctr;
}
NUKA_RANDOM_HD Philox4x32Key SplitSeed(uint64_t seed) {
    Philox4x32Key key;
    key.v[0] = static_cast<uint32_t>(seed & 0xFFFFFFFFu);
    key.v[1] = static_cast<uint32_t>(seed >> 32);
    return key;
}
NUKA_RANDOM_HD Philox4x32Counter MakeCounter(
    uint32_t element_idx, uint64_t seq) {
    Philox4x32Counter ctr;
    ctr.v[0] = element_idx;
    ctr.v[1] = static_cast<uint32_t>(seq & 0xFFFFFFFFu);
    ctr.v[2] = static_cast<uint32_t>(seq >> 32);
    ctr.v[3] = 0u;
    return ctr;
}
NUKA_RANDOM_HD float Uint32ToUniform01(uint32_t x) {
    return (static_cast<float>(x) + 1.0f) * 2.3283064365386963e-10f;
}
NUKA_RANDOM_HD float NormalSample(uint64_t seed,
                                  uint32_t element_idx, uint64_t seq) {
    const Philox4x32Counter out =
        Philox4x32_10(MakeCounter(element_idx, seq), SplitSeed(seed));
    const float u1 = Uint32ToUniform01(out.v[0]);  // in (0,1], log finite
    const float u2 = Uint32ToUniform01(out.v[1]);
    const float radius = sqrtf(-2.0f * logf(u1));
    const float theta = 6.28318530717958648f * u2;  // 2*pi
    return radius * cosf(theta);
}
NUKA_RANDOM_HD float GaussianSample(uint64_t seed,
                                    uint32_t element_idx, uint64_t seq,
                                    float mean, float stddev) {
    return mean + stddev * NormalSample(seed, element_idx, seq);
}
NUKA_RANDOM_HD uint32_t PoissonSample(uint64_t seed,
                                     uint32_t element_idx, uint64_t seq, float lambda) {
    if (!(lambda > 0.0f) || !(lambda <= 1.0e8f)) {
        return 0u;
    }
    const Philox4x32Key key = SplitSeed(seed);
    Philox4x32Counter ctr = MakeCounter(element_idx, seq);
    uint32_t sub_round = 0u;
    if (lambda < 30.0f) {
        const float limit = expf(-lambda);
        float product = 1.0f;
        uint32_t count = 0u;
        for (;;) {
            ctr.v[3] = sub_round++;
            const Philox4x32Counter out = Philox4x32_10(ctr, key);
            for (int lane = 0; lane < 4; ++lane) {
                product *= Uint32ToUniform01(out.v[lane]);
                if (product <= limit) return count;
                ++count;
            }
        }
    }
    const double rate = lambda;
    const double root = sqrt(rate);
    const double b = 0.931 + 2.53 * root;
    const double a = -0.059 + 0.02483 * b;
    const double inverse_alpha = 1.1239 + 1.1328 / (b - 3.4);
    const double squeeze = 0.9277 - 3.6224 / (b - 2.0);
    for (;;) {
        ctr.v[3] = sub_round++;
        const Philox4x32Counter out = Philox4x32_10(ctr, key);
        const double u = (static_cast<double>(out.v[0]) + 0.5) * 0x1p-32 - 0.5;
        const double v = (static_cast<double>(out.v[1]) + 0.5) * 0x1p-32;
        const double us = 0.5 - fabs(u);
        const double k = floor((2.0 * a / us + b) * u + rate + 0.43);
        if (k < 0.0 || k > 4294967295.0 || (us < 0.013 && v > us)) continue;
        if (us >= 0.07 && v <= squeeze) return static_cast<uint32_t>(k);
        const double log_acceptance = -rate + k * log(rate) - lgamma(k + 1.0);
        if (log(v * inverse_alpha / (a / (us * us) + b)) <= log_acceptance)
            return static_cast<uint32_t>(k);
    }
}

}  // namespace nuka::sensor::noise

#undef NUKA_RANDOM_HD
