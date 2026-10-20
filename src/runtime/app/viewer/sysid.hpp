#pragma once
// System identification: an episode replayed under candidate model values, and the CMA-ES search
// that picks the candidates.

#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace nuka::runtime::app::viewer {

// What the identification replays: from the first step's state, each later step's drive targets on
// the episode's joints, scored against the joint positions recorded after that step.
struct SysIdEpisode {
    // A floating root: its link, its articulation, and its pose then body velocity at the first step.
    struct Root {
        uint32_t link = 0u;
        uint32_t articulation = 0u;
        float values[13] = {};
    };
    uint32_t env = 0u;
    std::vector<uint32_t> links;  // the joints' links, the columns below
    size_t steps = 0u;
    std::vector<float> q, qd, action;  // per step and joint
    std::vector<Root> roots;
};

// One replay's mean squared joint error [rad² or m²]; infinite when the replay blew up.
struct SysIdResult {
    uint64_t request = 0u;
    double loss = 0.0;
};

// A searched quantity: registry parameter `param` on `links`, set to the value, or for a scale to
// each link's nominal times the value. The search sees [min, max] as [0, 1].
struct SysIdDimension {
    std::string key;    // what the request named it
    std::string field;  // the scene field an apply edits
    uint32_t param = 0u;
    bool scale = false;
    std::vector<std::string> joints;
    std::vector<uint32_t> links;
    std::vector<double> nominal;  // per link
    double min = 0.0;
    double max = 1.0;
    double Value(double x) const { return min + x * (max - min); }
};

// The registry writes a point of the unit cube stands for, as (param, element, value) triples.
std::vector<float> SysIdWrites(const std::vector<SysIdDimension>& dims, const std::vector<double>& x);

// (mu/mu_w, lambda)-CMA-ES on the unit cube; the samples it hands out are clipped to the cube.
class Cmaes {
public:
    Cmaes() = default;
    Cmaes(std::vector<double> mean, double sigma, uint32_t lambda, uint64_t seed);
    std::vector<std::vector<double>> Ask();
    // `points` as Ask gave them, each with its loss.
    void Tell(const std::vector<std::vector<double>>& points, const std::vector<double>& losses);
    // The spread along every axis has shrunk below 1e-4 of the ranges.
    bool Converged() const;

private:
    void Decompose();

    size_t n_ = 0u;
    uint32_t lambda_ = 0u;
    uint32_t mu_ = 0u;
    std::vector<double> weights_;
    double mueff_ = 0.0, cc_ = 0.0, cs_ = 0.0, c1_ = 0.0, cmu_ = 0.0, damps_ = 0.0, chin_ = 0.0;
    std::vector<double> mean_, pc_, ps_;
    std::vector<double> c_, b_;  // the covariance and its eigenvectors as columns, n×n row-major
    std::vector<double> d_;      // the square roots of its eigenvalues
    double sigma_ = 0.0;
    uint64_t evaluations_ = 0u;
    std::mt19937_64 rng_;
};

}  // namespace nuka::runtime::app::viewer
