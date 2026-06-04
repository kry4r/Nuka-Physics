// ---------------------------------------------------------------------------
// v0.8 C4a: test of the MuJoCo `_efc_row` solref/solimp compliance port
// (src/constraint/solref_solimp.hpp).
// ---------------------------------------------------------------------------
// TWO DISTINCT regimes, intentionally separate (do NOT conflate):
//
//   (1) function-vs-HAND-TABLE  -- tolerance. Each expected (k,b,imp,D,R,aref)
//       is INDEPENDENTLY HAND-EVALUATED from the MuJoCo `_efc_row` formula
//       (the arithmetic is shown in the comment above each row); these are
//       decimal float32 oracle values, so the assert is NEAR, not byte-exact.
//       The hand math was cross-derived from mujoco_warp/_src/constraint.py
//       (the @wp.func _efc_row + the contact call site sign convention) and
//       re-checked with a standalone float32 evaluator -- it does NOT echo the
//       implementation (it is an independent oracle).
//
//   (2) HOST-vs-DEVICE PARITY    -- BYTE-EXACT (memcmp of the raw bits). The
//       same NUKA_SRSI_HD function is called on the host AND from a __global__
//       on identical inputs; the bit patterns must be IDENTICAL. This is the
//       D1 gate that JUSTIFIES the OPEN-F integer-power decision (libm `powf`
//       would NOT be host/device bit-identical). It passes because this TU is
//       compiled --fmad=false (device) / -ffp-contract=off (host): no FMA
//       contraction on either side + integer-power (no powf) + IEEE division
//       => every op is a correctly-rounded IEEE single => identical bits.
//
// NOTE: this is a .cpp compiled as CUDA (LANGUAGE CUDA in CMakeLists) so the
// host EXPECT helpers and the __global__ live in one TU.
// ---------------------------------------------------------------------------

#include "constraint/solref_solimp.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

namespace {

using nuka::constraint::CompliantContactRow;
using nuka::constraint::CompliantRowTerms;
using nuka::constraint::ComputeCompliantRow;
using nuka::constraint::ComputeCompliantRowTerms;

// One test case: the inputs + the HAND-EVALUATED expected terms.
struct Case {
    const char* name;
    float solref[2];
    float solimp[5];
    float pos_imp;
    float pos_aref;
    float vel;
    float invweight;
    float dt;
    bool refsafe;
    // hand-evaluated expected (float32 oracle):
    float exp_k, exp_b, exp_imp, exp_D, exp_R, exp_aref;
};

// ---------------------------------------------------------------------------
// The TABLE. Each expected value is hand-derived; arithmetic shown.
// MuJoCo default solimp = {dmin=0.9, dmax=0.95, width=0.001, mid=0.5, power=2}.
// Default solref = {timeconst=0.02, dampratio=1.0}.
//   For the DEFAULT solref (all cases unless noted):
//     k = 1/(dmax^2 * tc^2 * dr^2) = 1/(0.95^2 * 0.02^2 * 1) = 1/0.000361 = 2770.0835
//     b = 2/(dmax*tc) = 2/(0.95*0.02) = 2/0.019 = 105.26316
//   x = |pos_imp|/width;  pm1 = power-1.
//   sigmoid (power=2): x<mid -> imp_y = (1/mid)*x^2;  x>=mid -> imp_y = 1 - (1/(1-mid))*(1-x)^2
//   imp = dmin + imp_y*(dmax-dmin); clamp to [dmin,dmax]; if x>1 imp=dmax.
//   D = 1/max(invweight*(1-imp)/imp, 1e-15);  R = 1/D.
//   aref = -k*imp*pos_aref - b*vel.
// ---------------------------------------------------------------------------
const Case kCases[] = {
    // --- default, x=1.0 exactly (pos=-0.001, width=0.001). x not >1, x>=mid:
    //     imp_y = 1 - (1/0.5)*(1-1)^2 = 1 - 0 = 1; imp = 0.9 + 1*0.05 = 0.95 (=dmax).
    //     D = 1/(1*(1-0.95)/0.95) = 1/(0.05/0.95) = 1/0.0526316 = 19.0 (float 18.999996)
    //     R = 1/19 = 0.0526316;  vel=0 -> aref = -2770.0835*0.95*(-0.001) = 2.6315794
    {"default_x1_edge", {0.02f, 1.0f}, {0.9f, 0.95f, 0.001f, 0.5f, 2.0f},
     -0.001f, -0.001f, 0.0f, 1.0f, 0.005f, true,
     2770.0835f, 105.26316f, 0.95f, 18.999996f, 0.05263159f, 2.6315794f},

    // --- default, x=0.3 (<mid). imp_y = 2*0.09 = 0.18; imp = 0.9 + 0.18*0.05 = 0.909.
    //     D = 1/(1*(1-0.909)/0.909) = 1/(0.091/0.909) = 1/0.100110 = 9.989008; R = 0.10011004
    //     aref = -2770.0835*0.909*(-0.0003) - 105.26316*(-0.1)
    //          = 0.755419 + 10.526316 = 11.281718
    {"default_x0p3_below_mid", {0.02f, 1.0f}, {0.9f, 0.95f, 0.001f, 0.5f, 2.0f},
     -0.0003f, -0.0003f, -0.1f, 1.0f, 0.005f, true,
     2770.0835f, 105.26316f, 0.909f, 9.989008f, 0.10011004f, 11.281718f},

    // --- default, x=0.7 (>mid). imp_y = 1 - 2*(0.3)^2 = 1 - 0.18 = 0.82; imp = 0.941.
    //     D = 1/(1*(1-0.941)/0.941) = 1/(0.059/0.941) = 1/0.0626993 = 15.949148; R=0.06269927
    //     aref = -2770.0835*0.941*(-0.0007) - 105.26316*0.2
    //          = 1.824749 - 21.052632 = -19.22798
    {"default_x0p7_above_mid", {0.02f, 1.0f}, {0.9f, 0.95f, 0.001f, 0.5f, 2.0f},
     -0.0007f, -0.0007f, 0.2f, 1.0f, 0.005f, true,
     2770.0835f, 105.26316f, 0.941f, 15.949148f, 0.06269927f, -19.22798f},

    // --- default, x=2.0 (>1 -> imp=dmax=0.95). invweight=2.
    //     D = 1/(2*(1-0.95)/0.95) = 1/(2*0.0526316) = 1/0.105263 = 9.499998; R=0.10526318
    //     aref = -2770.0835*0.95*(-0.002) - 105.26316*0.5 = 5.263159 - 52.63158 = -47.368423
    {"default_x2_deep_dmax", {0.02f, 1.0f}, {0.9f, 0.95f, 0.001f, 0.5f, 2.0f},
     -0.002f, -0.002f, 0.5f, 2.0f, 0.005f, true,
     2770.0835f, 105.26316f, 0.95f, 9.499998f, 0.10526318f, -47.368423f},

    // --- power=1 (LINEAR). pm1=0 -> mid^0=1, (1-mid)^0=1.
    //     x=0.4 (<mid). imp_a = (1/1)*x^1 = 0.4. imp = 0.9 + 0.4*0.05 = 0.92.
    //     (x>=mid branch would be imp_b = 1 - 1*(1-x) = x, same line -> continuous.)
    //     D = 1/(1*(1-0.92)/0.92) = 1/(0.08/0.92) = 1/0.0869565 = 11.499993; R=0.086956576
    //     vel=0 -> aref = -2770.0835*0.92*(-0.0004) = 1.0193907
    {"power1_linear_x0p4", {0.02f, 1.0f}, {0.9f, 0.95f, 0.001f, 0.5f, 1.0f},
     -0.0004f, -0.0004f, 0.0f, 1.0f, 0.005f, true,
     2770.0835f, 105.26316f, 0.92f, 11.499993f, 0.086956576f, 1.0193907f},

    // --- NEGATIVE solref[0] (DIRECT stiffness). solref={-50,1}. refsafe clamps
    //     timeconst=max(-50, 2*0.005=0.01)=0.01 (used by b only). solref[0]<=0:
    //     k = -(-50)/dmax^2 = 50/0.9025 = 55.401665. b = 2/(0.95*0.01) = 210.52632.
    //     x=0.5 (==mid -> NOT <mid -> imp_b branch). imp_b = 1 - 2*(0.5)^2 = 0.5;
    //     imp = 0.9 + 0.5*0.05 = 0.925 (float 0.92499995).
    //     D = 1/(1*(1-0.925)/0.925) = 1/(0.075/0.925) = 1/0.0810811 = 12.333324; R=0.08108114
    //     aref = -55.401665*0.925*(-0.0005) - 210.52632*0.3
    //          = 0.025626 - 63.157896 = -63.132275  (float -63.132275)
    {"neg_solref0_direct_k", {-50.0f, 1.0f}, {0.9f, 0.95f, 0.001f, 0.5f, 2.0f},
     -0.0005f, -0.0005f, 0.3f, 1.0f, 0.005f, true,
     55.401665f, 210.52632f, 0.92499995f, 12.333324f, 0.08108114f, -63.132275f},

    // --- NEGATIVE solref[1] (DIRECT damping). solref={0.02,-10}.
    //     k = 1/(0.95^2*0.02^2*?) but dampratio used only in k via dr^2; HOWEVER
    //     solref[1]<=0 overrides b ONLY. k uses dampratio=solref[1]=-10:
    //     k = 1/(0.9025 * 0.0004 * (-10)^2) = 1/(0.9025*0.0004*100) = 1/0.0361 = 27.700834
    //     b: solref[1]<=0 -> b = -(-10)/dmax = 10/0.95 = 10.526316.
    //     x=0.5 same imp=0.925; D=12.333324; R=0.08108114.
    //     aref = -27.700834*0.925*(-0.0005) - 10.526316*0.4
    //          = 0.012812 - 4.210526 = -4.197715  (float -4.197715)
    {"neg_solref1_direct_b", {0.02f, -10.0f}, {0.9f, 0.95f, 0.001f, 0.5f, 2.0f},
     -0.0005f, -0.0005f, 0.4f, 1.0f, 0.005f, true,
     27.700834f, 10.526316f, 0.92499995f, 12.333324f, 0.08108114f, -4.197715f},

    // --- REFSAFE clamp: timeconst=0.005 < 2*dt=0.02 -> clamped to 0.02.
    //     k,b computed with tc=0.02 (== default): k=2770.0835, b=105.26316.
    //     x=0.6 (>mid). imp_b = 1 - 2*(0.4)^2 = 1 - 0.32 = 0.68; imp = 0.9+0.68*0.05=0.934.
    //     D = 1/(1*(1-0.934)/0.934) = 1/(0.066/0.934) = 1/0.0706638 = 14.151519; R=0.070663795
    //     vel=0 -> aref = -2770.0835*0.934*(-0.0006) = 1.5523549
    {"refsafe_clamp_tc", {0.005f, 1.0f}, {0.9f, 0.95f, 0.001f, 0.5f, 2.0f},
     -0.0006f, -0.0006f, 0.0f, 1.0f, 0.01f, true,
     2770.0835f, 105.26316f, 0.934f, 14.151519f, 0.070663795f, 1.5523549f},

    // --- imp clamps at dmin: x=0.0001 (tiny). imp_y = 2*x^2 ~= 2e-8 ~= 0;
    //     imp = 0.9 + ~0 = 0.9 (=dmin floor). D = 1/(1*(1-0.9)/0.9) = 1/(0.1/0.9)
    //     = 1/0.111111 = 8.999997; R = 0.11111115.
    //     vel=0 -> aref = -2770.0835*0.9*(-1e-7) = 0.00024930752
    {"imp_clamp_dmin_tiny_x", {0.02f, 1.0f}, {0.9f, 0.95f, 0.001f, 0.5f, 2.0f},
     -1.0e-7f, -1.0e-7f, 0.0f, 1.0f, 0.005f, true,
     2770.0835f, 105.26316f, 0.9f, 8.999997f, 0.11111115f, 0.00024930752f},
};
constexpr int kNumCases = static_cast<int>(sizeof(kCases) / sizeof(kCases[0]));

// Bit-packed inputs the kernel consumes (POD, copyable to device).
struct KernelInput {
    float solref[2];
    float solimp[5];
    float pos_imp;
    float pos_aref;
    float vel;
    float invweight;
    float dt;
    int refsafe;  // bool as int for trivial device copy
};

// The kernel result -- the full terms, packed so we can memcmp it vs the host.
struct KernelResult {
    float k, b, imp, D, R, aref;
};

__global__ void ComputeKernel(const KernelInput* in, KernelResult* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const KernelInput& q = in[i];
    const CompliantRowTerms t = ComputeCompliantRowTerms(
        q.solref, q.solimp, q.pos_imp, q.pos_aref, q.vel, q.invweight, q.dt,
        q.refsafe != 0);
    out[i].k = t.k;
    out[i].b = t.b;
    out[i].imp = t.imp;
    out[i].D = t.D;
    out[i].R = t.R;
    out[i].aref = t.aref;
}

// ---------------------------------------------------------------------------
// Regime (1): function-vs-hand-table (tolerance).
// ---------------------------------------------------------------------------
TEST(SolrefSolimp, FunctionMatchesHandEvaluatedTable) {
    for (int i = 0; i < kNumCases; ++i) {
        const Case& c = kCases[i];
        const CompliantRowTerms t = ComputeCompliantRowTerms(
            c.solref, c.solimp, c.pos_imp, c.pos_aref, c.vel, c.invweight, c.dt,
            c.refsafe);
        SCOPED_TRACE(c.name);
        // Relative+absolute tolerance for the decimal hand-values (float32).
        EXPECT_NEAR(t.k, c.exp_k, 1e-3f * (1.0f + std::abs(c.exp_k)));
        EXPECT_NEAR(t.b, c.exp_b, 1e-3f * (1.0f + std::abs(c.exp_b)));
        EXPECT_NEAR(t.imp, c.exp_imp, 1e-5f);
        EXPECT_NEAR(t.D, c.exp_D, 1e-3f * (1.0f + std::abs(c.exp_D)));
        EXPECT_NEAR(t.R, c.exp_R, 1e-5f);
        EXPECT_NEAR(t.aref, c.exp_aref, 1e-3f * (1.0f + std::abs(c.exp_aref)));

        // The compact wrapper must agree with the rich terms.
        const CompliantContactRow row = ComputeCompliantRow(
            c.solref, c.solimp, c.pos_imp, c.pos_aref, c.vel, c.invweight, c.dt,
            c.refsafe);
        EXPECT_EQ(std::memcmp(&row.R, &t.R, sizeof(float)), 0) << c.name;
        EXPECT_EQ(std::memcmp(&row.aref_bias, &t.aref, sizeof(float)), 0) << c.name;
    }
}

// Sanity: D and R are reciprocals and were NOT swapped. R must shrink as imp
// grows (harder contact -> smaller regularizer); D must grow.
TEST(SolrefSolimp, DAndRNotSwapped) {
    const float solref[2] = {0.02f, 1.0f};
    const float solimp[5] = {0.9f, 0.95f, 0.001f, 0.5f, 2.0f};
    // shallow (small |pos|, small imp ~ dmin) vs deep (imp -> dmax).
    const CompliantRowTerms shallow =
        ComputeCompliantRowTerms(solref, solimp, -1e-7f, -1e-7f, 0.0f, 1.0f, 0.005f, true);
    const CompliantRowTerms deep =
        ComputeCompliantRowTerms(solref, solimp, -0.002f, -0.002f, 0.0f, 1.0f, 0.005f, true);
    // R = 1/D exactly (reciprocal pair, not swapped).
    EXPECT_NEAR(shallow.R * shallow.D, 1.0f, 1e-4f);
    EXPECT_NEAR(deep.R * deep.D, 1.0f, 1e-4f);
    // harder (deep, higher imp) -> smaller R, larger D.
    EXPECT_GT(deep.imp, shallow.imp);
    EXPECT_LT(deep.R, shallow.R);
    EXPECT_GT(deep.D, shallow.D);
}

// ---------------------------------------------------------------------------
// Regime (2): host-vs-device BYTE-EXACT parity (the OPEN-F integer-power proof).
// ---------------------------------------------------------------------------
TEST(SolrefSolimp, HostDeviceByteIdentical) {
    int dev_count = 0;
    const cudaError_t cnt_err = cudaGetDeviceCount(&dev_count);
    if (cnt_err != cudaSuccess || dev_count == 0) {
        GTEST_SKIP() << "no CUDA device";
    }

    std::vector<KernelInput> inputs(kNumCases);
    std::vector<KernelResult> host_results(kNumCases);
    for (int i = 0; i < kNumCases; ++i) {
        const Case& c = kCases[i];
        KernelInput& q = inputs[i];
        q.solref[0] = c.solref[0];
        q.solref[1] = c.solref[1];
        for (int j = 0; j < 5; ++j) q.solimp[j] = c.solimp[j];
        q.pos_imp = c.pos_imp;
        q.pos_aref = c.pos_aref;
        q.vel = c.vel;
        q.invweight = c.invweight;
        q.dt = c.dt;
        q.refsafe = c.refsafe ? 1 : 0;

        // HOST evaluation -> packed result.
        const CompliantRowTerms t = ComputeCompliantRowTerms(
            q.solref, q.solimp, q.pos_imp, q.pos_aref, q.vel, q.invweight, q.dt,
            q.refsafe != 0);
        host_results[i] = {t.k, t.b, t.imp, t.D, t.R, t.aref};
    }

    KernelInput* d_in = nullptr;
    KernelResult* d_out = nullptr;
    ASSERT_EQ(cudaMalloc(&d_in, sizeof(KernelInput) * kNumCases), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_out, sizeof(KernelResult) * kNumCases), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_in, inputs.data(), sizeof(KernelInput) * kNumCases,
                         cudaMemcpyHostToDevice),
              cudaSuccess);

    const int block = 64;
    const int grid = (kNumCases + block - 1) / block;
    ComputeKernel<<<grid, block>>>(d_in, d_out, kNumCases);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<KernelResult> dev_results(kNumCases);
    ASSERT_EQ(cudaMemcpy(dev_results.data(), d_out, sizeof(KernelResult) * kNumCases,
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    cudaFree(d_in);
    cudaFree(d_out);

    // BYTE-EXACT: every host result bit-pattern == device result bit-pattern.
    for (int i = 0; i < kNumCases; ++i) {
        const int cmp = std::memcmp(&host_results[i], &dev_results[i],
                                    sizeof(KernelResult));
        EXPECT_EQ(cmp, 0) << "host/device byte mismatch in case '" << kCases[i].name
                          << "': host(k=" << host_results[i].k
                          << " aref=" << host_results[i].aref << ") dev(k="
                          << dev_results[i].k << " aref=" << dev_results[i].aref << ")";
    }
}

}  // namespace
