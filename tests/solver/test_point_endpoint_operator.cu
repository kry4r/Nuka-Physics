// Point-mass endpoint operators agree with the dense Jacobian over merged mass degrees of freedom,
// and a warm-started compliant contact block solved through them keeps its cold-start fixed point.

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <type_traits>
#include <utility>
#include <vector>

#include "constraint/coulomb_contact.hpp"
#include "nk/solve/point_endpoint.hpp"
#include "phi/backend_cuda/ops/union_types.cuh"

namespace {

namespace nk = ::nuka::nk;
using ::nuka::math::Vec3;
using ::nuka::phi::nkops::PointMassView;

constexpr uint32_t kParticles = 48u;
constexpr uint32_t kNodes = 32u;
constexpr uint32_t kSides = 6u;
constexpr uint32_t kWarp = 32u;

struct Evaluation {
    float velocity[kSides];
    float velocity_warp[kSides];
    float coupling[kSides][kSides];
    float velocity_after[kSides];
};

// Production reads for every side, then an impulse on side `pushed` through its M^-1 J^T.
__global__ void EvaluateKernel(PointMassView points, const nk::NkRowSide* sides, uint32_t pushed,
                               float impulse, Evaluation* out) {
    const uint32_t lane = threadIdx.x;
    for (uint32_t i = 0u; i < kSides; ++i) {
        const float warp = points.RowVelocityWarp(sides[i], lane);
        if (lane == 0u) {
            out->velocity[i] = points.RowVelocity(sides[i]);
            out->velocity_warp[i] = warp;
            for (uint32_t j = 0u; j < kSides; ++j) out->coupling[i][j] = points.Coupling(sides[i], sides[j]);
        }
    }
    __syncwarp();
    if (lane == 0u) {
        for (uint32_t t = 0u; t < points.Count(sides[pushed]); ++t) {
            const auto term = points.At(sides[pushed], t);
            points.Velocity(term.kind)[term.index] +=
                term.jacobian * (points.InverseMass(term.kind)[term.index] * impulse);
        }
        for (uint32_t i = 0u; i < kSides; ++i) out->velocity_after[i] = points.RowVelocity(sides[i]);
    }
}

// Sequential rows: each side's impulse moves its terms through M^-1 J^T before the next side reads.
__global__ void PushKernel(PointMassView points, const nk::NkRowSide* sides, uint32_t count,
                           const float* impulses) {
    for (uint32_t i = 0u; i < count; ++i)
        for (uint32_t t = 0u; t < points.Count(sides[i]); ++t) {
            const auto term = points.At(sides[i], t);
            points.Velocity(term.kind)[term.index] +=
                term.jacobian * (points.InverseMass(term.kind)[term.index] * impulses[i]);
        }
}

__global__ void MeasureKernel(PointMassView points, const nk::NkRowSide* sides, uint32_t count,
                              float* velocity, float* coupling) {
    for (uint32_t i = 0u; i < count; ++i) {
        velocity[i] = points.RowVelocity(sides[i]);
        for (uint32_t j = 0u; j < count; ++j) coupling[i * count + j] = points.Coupling(sides[i], sides[j]);
    }
}

using Key = std::pair<uint32_t, uint32_t>;
struct DenseVec { double x = 0.0, y = 0.0, z = 0.0; };

struct Measured {
    std::vector<float> velocity, coupling;
    std::vector<Vec3> particle_velocity, grid_velocity;
};

// Smallest eigenvalue of a symmetric matrix by cyclic Jacobi rotations.
double MinEigenvalue(std::vector<double> a, size_t n) {
    for (int sweep = 0; sweep < 64; ++sweep) {
        for (size_t p = 0; p < n; ++p)
            for (size_t q = p + 1; q < n; ++q) {
                if (std::fabs(a[p * n + q]) < 1.0e-300) continue;
                const double theta = 0.5 * std::atan2(2.0 * a[p * n + q], a[q * n + q] - a[p * n + p]);
                const double c = std::cos(theta), s = std::sin(theta);
                for (size_t k = 0; k < n; ++k) {
                    const double kp = a[k * n + p], kq = a[k * n + q];
                    a[k * n + p] = c * kp - s * kq;
                    a[k * n + q] = s * kp + c * kq;
                }
                for (size_t k = 0; k < n; ++k) {
                    const double pk = a[p * n + k], qk = a[q * n + k];
                    a[p * n + k] = c * pk - s * qk;
                    a[q * n + k] = s * pk + c * qk;
                }
            }
    }
    double result = a[0];
    for (size_t i = 1; i < n; ++i) result = std::min(result, a[i * n + i]);
    return result;
}

class PointEndpointOperator : public ::testing::Test {
protected:
    void SetUp() override {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
            GTEST_SKIP() << "no CUDA device";
        for (uint32_t i = 0u; i < kParticles; ++i) {
            particle_inv_mass.push_back(0.5f + 0.03f * float(i % 7u));
            particle_velocity.push_back({0.1f * float(i % 5u) - 0.2f, 0.05f * float(i % 3u), -0.07f * float(i % 4u)});
        }
        for (uint32_t i = 0u; i < kNodes; ++i) {
            grid_inv_mass.push_back(0.8f + 0.05f * float(i % 5u));
            grid_velocity.push_back({-0.03f * float(i % 6u), 0.04f * float(i % 4u) - 0.05f, 0.02f * float(i % 3u)});
        }
        // Two triangles share hub particles 13 and 29; the offset makes every column a full 3x3 map.
        const Vec3 first[3] = {{0.2f, -0.8f, -0.5f}, {0.4f, 0.9f, -0.3f}, {-0.1f, 0.2f, 1.1f}};
        const Vec3 second[3] = {first[1], first[2], {0.9f, 0.6f, 0.4f}};
        AddTriangle({7u, 13u, 29u}, first, {0.2f, 0.3f, 0.5f}, {0.04f, -0.02f, 0.03f});
        AddTriangle({13u, 29u, 41u}, second, {0.25f, 0.45f, 0.3f}, {-0.01f, 0.03f, 0.02f});
        // One material sample's grid stencil, shared by two contacts.
        const uint32_t nodes[4] = {3u, 4u, 9u, 10u};
        const float weights[4] = {0.4f, 0.3f, 0.2f, 0.1f};
        ranges.push_back({uint32_t(terms.size()), 4u});
        for (uint32_t i = 0u; i < 4u; ++i) terms.push_back(nk::WeightedPointEndpointTerm(nk::kNkSideGrid, nodes[i], weights[i]));
        const Vec3 n = Vec3{0.3f, -0.4f, 0.866f};
        const Vec3 n1 = Vec3{0.0f, 0.6f, 0.8f}, n2 = Vec3{0.8f, 0.0f, 0.6f};
        sides[0] = {nk::kNkSidePointEndpoint, 0u, n, {}};
        sides[1] = {nk::kNkSidePointEndpoint, 1u, n * -1.0f, {}};
        sides[2] = {nk::kNkSidePointEndpoint, 2u, n1, {}};
        sides[3] = {nk::kNkSidePointEndpoint, 2u, n2, {}};
        sides[4] = {nk::kNkSideParticle, 13u, n2, {}};
        sides[5] = {nk::kNkSidePointEndpoint, 0u, n * -1.0f, {}};
    }

    void AddTriangle(std::vector<uint32_t> indices, const Vec3* vertices, Vec3 barycentric, Vec3 offset) {
        const Vec3 point = vertices[0] * barycentric.x + vertices[1] * barycentric.y +
                           vertices[2] * barycentric.z + offset;
        nk::PointEndpointTerm built[3];
        ASSERT_TRUE(nk::BuildTrianglePointEndpoint(indices.data(), vertices, barycentric, point, built));
        const uint32_t count = nk::CanonicalizePointEndpointTerms(built, 3u);
        ranges.push_back({uint32_t(terms.size()), count});
        for (uint32_t i = 0u; i < count; ++i) terms.push_back(built[i]);
    }

    // Dense J of one side, accumulated per (kind, index) mass degree of freedom.
    std::map<Key, DenseVec> Dense(const nk::NkRowSide& side) const {
        std::map<Key, DenseVec> result;
        const auto add = [&](uint32_t kind, uint32_t index, Vec3 j) {
            auto& value = result[{kind, index}];
            value.x += j.x; value.y += j.y; value.z += j.z;
        };
        if (side.kind != nk::kNkSidePointEndpoint) {
            add(side.kind, side.index, side.jlin);
            return result;
        }
        const auto range = ranges[side.index];
        for (uint32_t t = 0u; t < range.count; ++t) {
            const auto& term = terms[range.first + t];
            add(term.kind, term.index, term.TransposeMultiply(side.jlin));
        }
        return result;
    }
    double InverseMass(const Key& key) const {
        return key.first == nk::kNkSideGrid ? grid_inv_mass[key.second] : particle_inv_mass[key.second];
    }
    Vec3 Velocity(const Key& key) const {
        return key.first == nk::kNkSideGrid ? grid_velocity[key.second] : particle_velocity[key.second];
    }
    double DenseVelocity(const nk::NkRowSide& side) const {
        double result = 0.0;
        for (const auto& [key, j] : Dense(side)) {
            const Vec3 v = Velocity(key);
            result += j.x * v.x + j.y * v.y + j.z * v.z;
        }
        return result;
    }
    double DenseCoupling(const std::vector<std::pair<nk::NkRowSide, nk::NkRowSide>>& pairs) const {
        double result = 0.0;
        for (const auto& [lhs, rhs] : pairs) {
            const auto a = Dense(lhs), b = Dense(rhs);
            for (const auto& [key, ja] : a) {
                const auto found = b.find(key);
                if (found == b.end()) continue;
                const auto& jb = found->second;
                result += InverseMass(key) * (ja.x * jb.x + ja.y * jb.y + ja.z * jb.z);
            }
        }
        return result;
    }
    // Merged row J = sum of both sides before the mass product, so shared hubs combine signed terms.
    double MergedRowCoupling(const nk::NkRowSide& a, const nk::NkRowSide& b) const {
        auto merged = Dense(a);
        for (const auto& [key, j] : Dense(b)) {
            auto& value = merged[key];
            value.x += j.x; value.y += j.y; value.z += j.z;
        }
        double result = 0.0;
        for (const auto& [key, j] : merged) result += InverseMass(key) * (j.x * j.x + j.y * j.y + j.z * j.z);
        return result;
    }

    Evaluation Run(uint32_t pushed, float impulse) {
        float *d_pim = nullptr, *d_gim = nullptr;
        Vec3 *d_pv = nullptr, *d_gv = nullptr;
        nk::PointEndpointRange* d_ranges = nullptr;
        nk::PointEndpointTerm* d_terms = nullptr;
        nk::NkRowSide* d_sides = nullptr;
        Evaluation* d_out = nullptr;
        const auto upload = [](auto** device, const auto& host) {
            using T = std::remove_reference_t<decltype(host[0])>;
            EXPECT_EQ(cudaMalloc(device, host.size() * sizeof(T)), cudaSuccess);
            EXPECT_EQ(cudaMemcpy(*device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice), cudaSuccess);
        };
        upload(&d_pim, particle_inv_mass);
        upload(&d_gim, grid_inv_mass);
        upload(&d_pv, particle_velocity);
        upload(&d_gv, grid_velocity);
        upload(&d_ranges, ranges);
        upload(&d_terms, terms);
        const std::vector<nk::NkRowSide> host_sides(sides, sides + kSides);
        upload(&d_sides, host_sides);
        EXPECT_EQ(cudaMalloc(&d_out, sizeof(Evaluation)), cudaSuccess);
        PointMassView points;
        points.particle_inv_mass = d_pim;
        points.particle_velocity = d_pv;
        points.grid_inv_mass = d_gim;
        points.grid_velocity = d_gv;
        points.ranges = d_ranges;
        points.terms = d_terms;
        EvaluateKernel<<<1, kWarp>>>(points, d_sides, pushed, impulse, d_out);
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        Evaluation out{};
        EXPECT_EQ(cudaMemcpy(&out, d_out, sizeof(Evaluation), cudaMemcpyDeviceToHost), cudaSuccess);
        for (void* p : {static_cast<void*>(d_pim), static_cast<void*>(d_gim), static_cast<void*>(d_pv),
                        static_cast<void*>(d_gv), static_cast<void*>(d_ranges), static_cast<void*>(d_terms),
                        static_cast<void*>(d_sides), static_cast<void*>(d_out)})
            cudaFree(p);
        return out;
    }

    // Applies `impulses` to `drive` side by side from the initial state, then measures every side.
    Measured Drive(const std::vector<nk::NkRowSide>& drive, const std::vector<float>& impulses) {
        std::vector<void*> owned;
        const auto upload = [&owned](const auto& host) {
            using T = typename std::decay_t<decltype(host)>::value_type;
            T* device = nullptr;
            EXPECT_EQ(cudaMalloc(&device, host.size() * sizeof(T)), cudaSuccess);
            EXPECT_EQ(cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice), cudaSuccess);
            owned.push_back(device);
            return device;
        };
        const auto download = [](const auto* device, size_t count) {
            std::vector<std::remove_const_t<std::remove_pointer_t<decltype(device)>>> host(count);
            EXPECT_EQ(cudaMemcpy(host.data(), device, count * sizeof(host[0]), cudaMemcpyDeviceToHost), cudaSuccess);
            return host;
        };
        PointMassView points;
        points.particle_inv_mass = upload(particle_inv_mass);
        points.particle_velocity = upload(particle_velocity);
        points.grid_inv_mass = upload(grid_inv_mass);
        points.grid_velocity = upload(grid_velocity);
        points.ranges = upload(ranges);
        points.terms = upload(terms);
        const uint32_t count = static_cast<uint32_t>(drive.size());
        const auto* d_sides = upload(drive);
        const auto* d_impulses = upload(impulses);
        float* d_velocity = upload(std::vector<float>(count));
        float* d_coupling = upload(std::vector<float>(size_t{count} * count));
        PushKernel<<<1, 1>>>(points, d_sides, count, d_impulses);
        MeasureKernel<<<1, 1>>>(points, d_sides, count, d_velocity, d_coupling);
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        Measured out;
        out.velocity = download(d_velocity, count);
        out.coupling = download(d_coupling, size_t{count} * count);
        out.particle_velocity = download(points.particle_velocity, kParticles);
        out.grid_velocity = download(points.grid_velocity, kNodes);
        for (void* p : owned) cudaFree(p);
        return out;
    }

    std::vector<float> particle_inv_mass, grid_inv_mass;
    std::vector<Vec3> particle_velocity, grid_velocity;
    std::vector<nk::PointEndpointRange> ranges;
    std::vector<nk::PointEndpointTerm> terms;
    nk::NkRowSide sides[kSides];
};

TEST_F(PointEndpointOperator, GeneralEndpointVirtualWorkMatchesDenseJacobian) {
    const Evaluation out = Run(2u, 0.0f);
    // <A v, n> through the forward columns equals <v, A^T n> as the solver reads it.
    const auto range = ranges[0];
    Vec3 endpoint_velocity{};
    for (uint32_t t = 0u; t < range.count; ++t) {
        const auto& term = terms[range.first + t];
        endpoint_velocity += term.Multiply(particle_velocity[term.index]);
    }
    EXPECT_NEAR(out.velocity[0], endpoint_velocity.Dot(sides[0].jlin), 2.0e-6);
    for (uint32_t i = 0u; i < kSides; ++i) {
        const double reference = DenseVelocity(sides[i]);
        EXPECT_NEAR(out.velocity[i], reference, 2.0e-6 * (1.0 + std::fabs(reference))) << "side " << i;
        EXPECT_NEAR(out.velocity_warp[i], out.velocity[i], 2.0e-6 * (1.0 + std::fabs(reference))) << "side " << i;
        for (uint32_t j = 0u; j < kSides; ++j) {
            const double coupling = DenseCoupling({{sides[i], sides[j]}});
            EXPECT_NEAR(out.coupling[i][j], coupling, 2.0e-6 * (1.0 + std::fabs(coupling))) << i << "," << j;
            EXPECT_EQ(out.coupling[i][j], out.coupling[j][i]) << i << "," << j;
        }
    }
}

TEST_F(PointEndpointOperator, SameHubSidesMergeSignedContributions) {
    const Evaluation out = Run(2u, 0.0f);
    // A row whose two endpoints share hub particles: the four side couplings form the merged J M^-1 J^T.
    const double merged = MergedRowCoupling(sides[0], sides[1]);
    const double assembled = double(out.coupling[0][0]) + out.coupling[0][1] + out.coupling[1][0] + out.coupling[1][1];
    EXPECT_NEAR(assembled, merged, 4.0e-6 * (1.0 + merged));
    EXPECT_GT(std::fabs(out.coupling[0][1]), 1.0e-3);
    EXPECT_GT(std::fabs(double(out.coupling[0][0]) + out.coupling[1][1] - merged), 1.0e-3);
    // Opposite sides on one endpoint cancel exactly in both the effective mass and the relative velocity.
    const double cancelled = double(out.coupling[0][0]) + out.coupling[0][5] + out.coupling[5][0] + out.coupling[5][5];
    EXPECT_NEAR(cancelled, 0.0, 1.0e-6 * out.coupling[0][0]);
    EXPECT_NEAR(double(out.velocity[0]) + out.velocity[5], 0.0, 1.0e-7);
}

TEST_F(PointEndpointOperator, SharedSampleContactsRespondThroughTheirCoupling) {
    const float impulse = 0.37f;
    const Evaluation before = Run(2u, 0.0f);
    const Evaluation after = Run(2u, impulse);
    // Two contacts on one sample couple through its stencil nodes; the coupling is not dropped.
    const double coupling = DenseCoupling({{sides[2], sides[3]}});
    EXPECT_GT(std::fabs(coupling), 1.0e-3);
    EXPECT_NEAR(after.velocity_after[3] - before.velocity_after[3], after.coupling[2][3] * impulse, 2.0e-6);
    EXPECT_NEAR(after.velocity_after[2] - before.velocity_after[2], after.coupling[2][2] * impulse, 2.0e-6);
    // Endpoints with no shared mass degree of freedom do not respond.
    EXPECT_EQ(after.velocity_after[0], before.velocity_after[0]);
    EXPECT_EQ(after.velocity_after[4], before.velocity_after[4]);
}

TEST_F(PointEndpointOperator, SharedSampleGatherAndAggregatedScatterFactorize) {
    // Contacts naming one sample range read v_s = T_s v once and move nodes by H^-1 T_s^T sum_i n_i dl_i.
    ASSERT_EQ(sides[2].index, sides[3].index);
    const std::vector<nk::NkRowSide> contacts = {sides[2], sides[3]};
    const std::vector<float> impulses = {0.37f, -0.21f};
    const Measured before = Drive(contacts, {0.0f, 0.0f});
    const Measured after = Drive(contacts, impulses);
    const auto range = ranges[sides[2].index];
    Vec3 gathered{};
    for (uint32_t t = 0u; t < range.count; ++t) {
        const auto& term = terms[range.first + t];
        gathered += term.Multiply(Velocity({term.kind, term.index}));
    }
    for (uint32_t i = 0u; i < 2u; ++i) {
        const double reference = gathered.Dot(contacts[i].jlin);
        EXPECT_NEAR(before.velocity[i], reference, 2.0e-6 * (1.0 + std::fabs(reference))) << "contact " << i;
    }
    // Aggregating changes only the float association of the sequential scatter.
    const Vec3 aggregated = contacts[0].jlin * impulses[0] + contacts[1].jlin * impulses[1];
    std::vector<bool> touched(kNodes, false);
    for (uint32_t t = 0u; t < range.count; ++t) {
        const auto& term = terms[range.first + t];
        ASSERT_EQ(term.kind, nk::kNkSideGrid);
        touched[term.index] = true;
        const Vec3 expected = grid_velocity[term.index] +
                              term.TransposeMultiply(aggregated) * grid_inv_mass[term.index];
        const Vec3 moved = after.grid_velocity[term.index];
        EXPECT_LE((moved - expected).Length(), 1.0e-6f + 1.0e-5f * expected.Length()) << "node " << term.index;
    }
    for (uint32_t node = 0u; node < kNodes; ++node)
        if (!touched[node]) EXPECT_EQ(after.grid_velocity[node], grid_velocity[node]) << "node " << node;
    // Triangle endpoints on shared hubs carry offset-dependent columns: sharing keys on the range.
    const auto column = [&](uint32_t which, uint32_t particle) {
        for (uint32_t t = 0u; t < ranges[which].count; ++t)
            if (terms[ranges[which].first + t].index == particle) return terms[ranges[which].first + t];
        ADD_FAILURE() << "particle " << particle << " not in range " << which;
        return nk::PointEndpointTerm{};
    };
    const auto first = column(0u, 13u), second = column(1u, 13u);
    float largest = 0.0f;
    for (uint32_t axis = 0u; axis < 3u; ++axis)
        largest = std::max(largest, (first.column[axis] - second.column[axis]).Length());
    EXPECT_GT(largest, 1.0e-3f);
}

TEST_F(PointEndpointOperator, SideCouplingIsPositiveSemidefinite) {
    const std::vector<nk::NkRowSide> all(sides, sides + kSides);
    const Measured out = Drive(all, std::vector<float>(kSides, 0.0f));
    std::vector<double> matrix(kSides * kSides);
    double trace = 0.0;
    for (uint32_t i = 0u; i < kSides; ++i) {
        trace += out.coupling[i * kSides + i];
        for (uint32_t j = 0u; j < kSides; ++j) matrix[i * kSides + j] = out.coupling[i * kSides + j];
    }
    EXPECT_GE(MinEigenvalue(matrix, kSides), -1.0e-6 * trace);
    // p^T W p is |H^-1/2 A^T p|^2 on the merged mass degrees of freedom.
    for (uint32_t trial = 0u; trial < 4u; ++trial) {
        std::vector<double> p(kSides);
        for (uint32_t i = 0u; i < kSides; ++i) p[i] = std::sin(1.7 * i + 0.9 * trial) + 0.25 * trial;
        std::map<Key, DenseVec> transposed;
        for (uint32_t i = 0u; i < kSides; ++i)
            for (const auto& [key, j] : Dense(sides[i])) {
                auto& value = transposed[key];
                value.x += p[i] * j.x; value.y += p[i] * j.y; value.z += p[i] * j.z;
            }
        double dense = 0.0, quadratic = 0.0;
        for (const auto& [key, j] : transposed) dense += InverseMass(key) * (j.x * j.x + j.y * j.y + j.z * j.z);
        for (uint32_t i = 0u; i < kSides; ++i)
            for (uint32_t j = 0u; j < kSides; ++j) quadratic += p[i] * matrix[i * kSides + j] * p[j];
        EXPECT_NEAR(quadratic, dense, 4.0e-6 * (1.0 + dense)) << "trial " << trial;
    }
}

TEST_F(PointEndpointOperator, CompliantWarmStartedContactKeepsTheOriginalFixedPoint) {
    // A sample contact against a static side with R_eff > 0 and a warm impulse already in v_bar.
    const Vec3 n = Vec3{0.2f, -0.3f, 0.93f}.Normalized();
    const Vec3 t1 = n.Cross(Vec3{1.0f, 0.0f, 0.0f}).Normalized();
    const Vec3 t2 = n.Cross(t1);
    const uint32_t sample = sides[2].index;
    const std::vector<nk::NkRowSide> frame = {{nk::kNkSidePointEndpoint, sample, n, {}},
                                              {nk::kNkSidePointEndpoint, sample, t1, {}},
                                              {nk::kNkSidePointEndpoint, sample, t2, {}}};
    const Vec3 regularizer{0.3f, 0.05f, 0.05f}, target{0.8f, 0.1f, -0.05f}, warm{0.6f, 0.05f, -0.02f};
    const float mu1 = 0.5f, mu2 = 0.8f;
    const Measured free = Drive(frame, {0.0f, 0.0f, 0.0f});
    const Measured bar = Drive(frame, {warm.x, warm.y, warm.z});
    const auto& w = bar.coupling;
    const nuka::math::SymmetricMat3 response{w[0] + regularizer.x, w[4] + regularizer.y,
                                             w[8] + regularizer.z, w[1], w[2], w[5]};
    const nuka::math::SymmetricMat3 delassus{w[0], w[4], w[8], w[1], w[2], w[5]};
    const Vec3 v_free{free.velocity[0], free.velocity[1], free.velocity[2]};
    const Vec3 v_bar{bar.velocity[0], bar.velocity[1], bar.velocity[2]};
    EXPECT_LT((v_bar - v_free - delassus.Multiply(warm)).Length(), 2.0e-6f);
    const auto regularized = [&](Vec3 impulse) {
        return Vec3{regularizer.x * impulse.x, regularizer.y * impulse.y, regularizer.z * impulse.z};
    };
    // Gauss-Seidel on the block with the production projection; warm starts apply only increments.
    const auto solve = [&](Vec3 impulse, Vec3 velocity) {
        for (uint32_t k = 0u; k < 4096u; ++k) {
            const Vec3 next = nuka::constraint::ProjectedCoulombStep(
                response, target - velocity - regularized(impulse), impulse, mu1, mu2);
            velocity += delassus.Multiply(next - impulse);
            impulse = next;
        }
        return std::make_pair(impulse, velocity);
    };
    const auto [impulse, velocity] = solve(warm, v_bar);
    ASSERT_GT(impulse.x, 0.1f);
    const auto residual = nuka::constraint::EvaluateCoulombContactResidual(
        response, velocity - target + regularized(impulse), impulse, mu1, mu2);
    EXPECT_TRUE(residual.Within(3.0e-6f, 3.0e-6f, 3.0e-6f))
        << residual.normal_natural_velocity << " " << residual.tangent_natural_velocity;
    const auto [cold, cold_velocity] = solve({}, v_free);
    EXPECT_LT((cold - impulse).Length(), 1.0e-5f);
    EXPECT_LT((cold_velocity - velocity).Length(), 1.0e-5f);
    // The residual detects a dropped R_eff lambda and a total impulse reapplied on top of v_bar.
    EXPECT_FALSE(nuka::constraint::EvaluateCoulombContactResidual(
        response, velocity - target, impulse, mu1, mu2).Within(3.0e-6f, 3.0e-6f, 3.0e-6f));
    const Vec3 reapplied = v_bar + delassus.Multiply(impulse);
    EXPECT_FALSE(nuka::constraint::EvaluateCoulombContactResidual(
        response, reapplied - target + regularized(impulse), impulse, mu1, mu2)
        .Within(3.0e-6f, 3.0e-6f, 3.0e-6f));
    // Moving the frozen state by the solved impulse through production scatter matches the operator.
    const Measured moved = Drive(frame, {impulse.x, impulse.y, impulse.z});
    for (uint32_t axis = 0u; axis < 3u; ++axis) {
        const float expected = axis == 0u ? velocity.x : axis == 1u ? velocity.y : velocity.z;
        EXPECT_NEAR(moved.velocity[axis], expected, 4.0e-6f * (1.0f + std::fabs(expected))) << "axis " << axis;
    }
}

}  // namespace
