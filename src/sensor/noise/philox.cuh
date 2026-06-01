#pragma once
// ---------------------------------------------------------------------------
// philox.cuh -- counter-based Philox4x32-10 RNG + uniform/normal/Poisson helpers
// (v0.5 p04 N1 sim-to-real sensor noise)
// ---------------------------------------------------------------------------
//
// The engine's FIRST RNG. It is COUNTER-BASED and STATELESS: a sample is a PURE
// FUNCTION of (seed, logical_counter). There is NO mutable RNG state, NO global
// counter, and NO dependence on thread scheduling -- which is exactly what
// DeterminismLevel::Strong (D1) requires: the SAME (seed, counter) yields the
// SAME bits on every run AND on the reverse/replay pass (no RNG state to
// checkpoint -- the replay re-derives the identical sample from the same
// indices).
//
// Generator: Philox4x32-10 (Salmon et al., "Parallel Random Numbers: As Easy as
// 1, 2, 3", SC'11 / the Random123 library). 4x 32-bit counter words, 2x 32-bit
// key words, 10 rounds. Bijective on the 128-bit counter space, so distinct
// counters give independent-looking, collision-free streams.
//
// Counter layout (this engine's convention):
//   key  = SplitSeed(seed)               -- the 64-bit seed split into 2x uint32
//   ctr  = { element_idx, seq_lo, seq_hi, 0 }
// element_idx and seq occupy SEPARATE 32-bit lanes (NOT an arithmetic mix like
// idx + seq*count). This is injective by construction: distinct (idx, seq) pairs
// map to distinct counters, so there are no stream collisions and "different seq
// -> different noise" / "different element -> different noise" hold by design.
// The first Philox output word (lane 0) is used for the primary uniform; lane 1
// provides the second uniform for Box-Muller's paired normal.
//
// Distributions:
//   uniform01  : output_word * 2^-32, then clamped to (0,1] so log() is finite.
//   normal     : Box-Muller from two uniforms (BOTH normals are valid; we return
//                the first, the standard single-draw use). Deterministic.
//   poisson    : Knuth's multiplicative algorithm (a deterministic loop over
//                uniforms until the running product drops below e^-lambda). Exact
//                for the small lambda regime this sensor-noise path targets;
//                O(lambda) iterations. No rejection / no branching on RNG state
//                beyond the counted loop, so it is reproducible bit-for-bit.
// ---------------------------------------------------------------------------

#include <cstdint>

// HOST/DEVICE portability. The Philox helpers are __host__ __device__ so the SAME
// pure function runs in a CUDA kernel (the N1 .cu noise path) AND on the host (the
// N2 domain-randomization sampling, compiled by the plain C++ toolchain). When
// this header is included from a non-NVCC translation unit (.cpp), __host__ /
// __device__ / __forceinline__ are not defined, so map them to empty / inline.
// <cmath> supplies the sqrtf/logf/cosf/sinf/expf the normal/Poisson host path uses
// (NVCC injects these via its device math headers; the host needs the explicit
// include).
#ifndef __CUDACC__
#include <cmath>
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#ifndef __forceinline__
#define __forceinline__ inline
#endif
#endif  // __CUDACC__

namespace nuka::sensor::noise {

// Philox4x32-10 round constants (the canonical Random123 values).
constexpr uint32_t kPhiloxM0 = 0xD2511F53u;
constexpr uint32_t kPhiloxM1 = 0xCD9E8D57u;
constexpr uint32_t kPhiloxW0 = 0x9E3779B9u;  // golden ratio (key bump, lane 0)
constexpr uint32_t kPhiloxW1 = 0xBB67AE85u;  // sqrt(3)-1   (key bump, lane 1)

// A 128-bit counter (4x uint32) and a 64-bit key (2x uint32).
struct Philox4x32Counter {
    uint32_t v[4];
};
struct Philox4x32Key {
    uint32_t v[2];
};

// 32x32 -> 64 multiply, returning hi and lo halves. mulhi/mullo are exact.
__host__ __device__ __forceinline__ void MulHiLo(uint32_t a, uint32_t b,
                                                  uint32_t* hi, uint32_t* lo) {
    const uint64_t product = static_cast<uint64_t>(a) * static_cast<uint64_t>(b);
    *hi = static_cast<uint32_t>(product >> 32);
    *lo = static_cast<uint32_t>(product);
}

// One Philox4x32 round (single round function).
__host__ __device__ __forceinline__ Philox4x32Counter PhiloxRound(
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

// Bump the key between rounds (Weyl sequence).
__host__ __device__ __forceinline__ Philox4x32Key PhiloxBumpKey(
    Philox4x32Key key) {
    key.v[0] += kPhiloxW0;
    key.v[1] += kPhiloxW1;
    return key;
}

// Philox4x32-10: 10 rounds. Pure function of (ctr, key).
__host__ __device__ __forceinline__ Philox4x32Counter Philox4x32_10(
    Philox4x32Counter ctr, Philox4x32Key key) {
    // Round 0 uses the raw key, then the key is bumped before each subsequent
    // round (9 bumps for the remaining 9 rounds): the canonical schedule.
    ctr = PhiloxRound(ctr, key);
    for (int round = 1; round < 10; ++round) {
        key = PhiloxBumpKey(key);
        ctr = PhiloxRound(ctr, key);
    }
    return ctr;
}

// Split a 64-bit seed into the 2x uint32 Philox key.
__host__ __device__ __forceinline__ Philox4x32Key SplitSeed(uint64_t seed) {
    Philox4x32Key key;
    key.v[0] = static_cast<uint32_t>(seed & 0xFFFFFFFFu);
    key.v[1] = static_cast<uint32_t>(seed >> 32);
    return key;
}

// Build the per-sample counter from stable logical indices. element_idx and the
// 64-bit sequence index occupy separate lanes (injective, no collisions).
__host__ __device__ __forceinline__ Philox4x32Counter MakeCounter(
    uint32_t element_idx, uint64_t seq) {
    Philox4x32Counter ctr;
    ctr.v[0] = element_idx;
    ctr.v[1] = static_cast<uint32_t>(seq & 0xFFFFFFFFu);
    ctr.v[2] = static_cast<uint32_t>(seq >> 32);
    ctr.v[3] = 0u;
    return ctr;
}

// uint32 -> uniform in (0, 1]. We map to (0,1] (never exactly 0) so a subsequent
// log() in Box-Muller / Knuth-Poisson is always finite. (x+1)*2^-32 lands in
// (2^-32, 1]; for x==0xFFFFFFFF it is exactly 1.0, which log handles fine.
__host__ __device__ __forceinline__ float Uint32ToUniform01(uint32_t x) {
    // 2^-32 == 2.3283064365386963e-10f.
    return (static_cast<float>(x) + 1.0f) * 2.3283064365386963e-10f;
}

// A single standard-normal draw (mean 0, stddev 1) via Box-Muller from the first
// two Philox output lanes. Deterministic. (The second normal,
// r*sin(theta), is equally valid but unused in the single-draw API.)
__host__ __device__ __forceinline__ float NormalSample(uint64_t seed,
                                                       uint32_t element_idx,
                                                       uint64_t seq) {
    const Philox4x32Counter out =
        Philox4x32_10(MakeCounter(element_idx, seq), SplitSeed(seed));
    const float u1 = Uint32ToUniform01(out.v[0]);  // in (0,1], log finite
    const float u2 = Uint32ToUniform01(out.v[1]);
    const float radius = sqrtf(-2.0f * logf(u1));
    const float theta = 6.28318530717958648f * u2;  // 2*pi
    return radius * cosf(theta);
}

// Gaussian(mean, stddev) sample.
__host__ __device__ __forceinline__ float GaussianSample(uint64_t seed,
                                                         uint32_t element_idx,
                                                         uint64_t seq,
                                                         float mean,
                                                         float stddev) {
    return mean + stddev * NormalSample(seed, element_idx, seq);
}

// Poisson(lambda) sample via Knuth's algorithm. Deterministic counted loop:
// multiply successive uniforms until the running product drops below e^-lambda;
// the number of multiplies minus one is the Poisson draw. Each uniform comes
// from a fresh Philox call keyed by an incrementing sub-counter (lane 3) so the
// draws within one sample are independent yet fully reproducible. Targets small
// lambda (sensor shot noise); cost is O(lambda) iterations.
__host__ __device__ __forceinline__ uint32_t PoissonSample(uint64_t seed,
                                                           uint32_t element_idx,
                                                           uint64_t seq,
                                                           float lambda) {
    if (!(lambda > 0.0f)) {
        return 0u;
    }
    const float limit = expf(-lambda);
    const Philox4x32Key key = SplitSeed(seed);
    Philox4x32Counter ctr = MakeCounter(element_idx, seq);
    float product = 1.0f;
    uint32_t count = 0u;
    // sub_round walks lane 3; each Philox call yields 4 uniforms (4 output
    // lanes). Cap iterations defensively (lambda small => essentially never hit).
    uint32_t sub_round = 0u;
    const uint32_t kMaxIters = 4096u;
    while (count < kMaxIters) {
        ctr.v[3] = sub_round++;
        const Philox4x32Counter out = Philox4x32_10(ctr, key);
        for (int lane = 0; lane < 4; ++lane) {
            product *= Uint32ToUniform01(out.v[lane]);
            if (product <= limit) {
                return count;
            }
            ++count;
        }
    }
    return count;
}

}  // namespace nuka::sensor::noise
