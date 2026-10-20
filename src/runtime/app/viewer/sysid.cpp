#include "runtime/app/viewer/sysid.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>

namespace nuka::runtime::app::viewer {

std::vector<float> SysIdWrites(const std::vector<SysIdDimension>& dims, const std::vector<double>& x) {
    std::vector<float> out;
    for (size_t i = 0; i < dims.size(); ++i) {
        const SysIdDimension& d = dims[i];
        const double v = d.Value(x[i]);
        for (size_t j = 0; j < d.links.size(); ++j)
            out.insert(out.end(), {static_cast<float>(d.param), static_cast<float>(d.links[j]),
                                   static_cast<float>(d.scale ? d.nominal[j] * v : v)});
    }
    return out;
}

// Hansen's defaults for the learning rates and the step-size damping.
Cmaes::Cmaes(std::vector<double> mean, double sigma, uint32_t lambda, uint64_t seed)
    : n_(mean.size()), lambda_(lambda), mu_(lambda / 2u), mean_(std::move(mean)), sigma_(sigma), rng_(seed) {
    const double n = static_cast<double>(n_);
    for (uint32_t i = 0; i < mu_; ++i) weights_.push_back(std::log(mu_ + 0.5) - std::log(i + 1.0));
    const double sum = std::accumulate(weights_.begin(), weights_.end(), 0.0);
    double squares = 0.0;
    for (double& w : weights_) {
        w /= sum;
        squares += w * w;
    }
    mueff_ = 1.0 / squares;
    cc_ = (4.0 + mueff_ / n) / (n + 4.0 + 2.0 * mueff_ / n);
    cs_ = (mueff_ + 2.0) / (n + mueff_ + 5.0);
    c1_ = 2.0 / ((n + 1.3) * (n + 1.3) + mueff_);
    cmu_ = std::min(1.0 - c1_, 2.0 * (mueff_ - 2.0 + 1.0 / mueff_) / ((n + 2.0) * (n + 2.0) + mueff_));
    damps_ = 1.0 + 2.0 * std::max(0.0, std::sqrt((mueff_ - 1.0) / (n + 1.0)) - 1.0) + cs_;
    chin_ = std::sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n));
    pc_.assign(n_, 0.0);
    ps_.assign(n_, 0.0);
    c_.assign(n_ * n_, 0.0);
    for (size_t i = 0; i < n_; ++i) c_[i * n_ + i] = 1.0;
    b_ = c_;
    d_.assign(n_, 1.0);
}

std::vector<std::vector<double>> Cmaes::Ask() {
    std::normal_distribution<double> normal;
    std::vector<std::vector<double>> points(lambda_, std::vector<double>(n_));
    std::vector<double> z(n_);
    for (std::vector<double>& x : points) {
        for (size_t i = 0; i < n_; ++i) z[i] = d_[i] * normal(rng_);
        for (size_t r = 0; r < n_; ++r) {
            double y = 0.0;
            for (size_t k = 0; k < n_; ++k) y += b_[r * n_ + k] * z[k];
            x[r] = std::clamp(mean_[r] + sigma_ * y, 0.0, 1.0);
        }
    }
    return points;
}

void Cmaes::Tell(const std::vector<std::vector<double>>& points, const std::vector<double>& losses) {
    std::vector<size_t> order(points.size());
    std::iota(order.begin(), order.end(), size_t{0});
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return losses[a] < losses[b]; });
    const std::vector<double> old = mean_;
    std::fill(mean_.begin(), mean_.end(), 0.0);
    for (uint32_t i = 0; i < mu_; ++i)
        for (size_t r = 0; r < n_; ++r) mean_[r] += weights_[i] * points[order[i]][r];
    evaluations_ += lambda_;
    std::vector<double> step(n_), whitened(n_, 0.0), rotated(n_, 0.0);
    for (size_t r = 0; r < n_; ++r) step[r] = (mean_[r] - old[r]) / sigma_;
    // C^(-1/2) step = B diag(1/d) B^T step
    for (size_t k = 0; k < n_; ++k) {
        for (size_t r = 0; r < n_; ++r) rotated[k] += b_[r * n_ + k] * step[r];
        rotated[k] /= d_[k];
    }
    for (size_t r = 0; r < n_; ++r)
        for (size_t k = 0; k < n_; ++k) whitened[r] += b_[r * n_ + k] * rotated[k];
    const double ps_gain = std::sqrt(cs_ * (2.0 - cs_) * mueff_);
    double norm = 0.0;
    for (size_t r = 0; r < n_; ++r) {
        ps_[r] = (1.0 - cs_) * ps_[r] + ps_gain * whitened[r];
        norm += ps_[r] * ps_[r];
    }
    norm = std::sqrt(norm);
    const double generations = static_cast<double>(evaluations_) / static_cast<double>(lambda_);
    const double fade = std::sqrt(1.0 - std::pow(1.0 - cs_, 2.0 * generations));
    const bool hsig = norm / fade / chin_ < 1.4 + 2.0 / (static_cast<double>(n_) + 1.0);
    const double pc_gain = std::sqrt(cc_ * (2.0 - cc_) * mueff_);
    for (size_t r = 0; r < n_; ++r) pc_[r] = (1.0 - cc_) * pc_[r] + (hsig ? pc_gain * step[r] : 0.0);
    const double keep = 1.0 - c1_ - cmu_ + (hsig ? 0.0 : c1_ * cc_ * (2.0 - cc_));
    for (size_t r = 0; r < n_; ++r)
        for (size_t k = 0; k < n_; ++k) {
            double rank_mu = 0.0;
            for (uint32_t i = 0; i < mu_; ++i)
                rank_mu += weights_[i] * (points[order[i]][r] - old[r]) * (points[order[i]][k] - old[k]);
            c_[r * n_ + k] = keep * c_[r * n_ + k] + c1_ * pc_[r] * pc_[k] + cmu_ * rank_mu / (sigma_ * sigma_);
        }
    sigma_ *= std::exp(cs_ / damps_ * (norm / chin_ - 1.0));
    Decompose();
}

bool Cmaes::Converged() const { return sigma_ * *std::max_element(d_.begin(), d_.end()) < 1e-4; }

// Cyclic Jacobi rotations: C = B diag(d²) B^T.
void Cmaes::Decompose() {
    std::vector<double> a = c_;
    b_.assign(n_ * n_, 0.0);
    for (size_t i = 0; i < n_; ++i) b_[i * n_ + i] = 1.0;
    for (int sweep = 0; sweep < 64; ++sweep) {
        double off = 0.0;
        for (size_t p = 0; p < n_; ++p)
            for (size_t q = p + 1u; q < n_; ++q) off += a[p * n_ + q] * a[p * n_ + q];
        if (off < 1e-30) break;
        for (size_t p = 0; p < n_; ++p)
            for (size_t q = p + 1u; q < n_; ++q) {
                const double apq = a[p * n_ + q];
                if (apq == 0.0) continue;
                const double theta = (a[q * n_ + q] - a[p * n_ + p]) / (2.0 * apq);
                const double t = (theta >= 0.0 ? 1.0 : -1.0) / (std::abs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0), s = t * c;
                for (size_t k = 0; k < n_; ++k) {
                    const double akp = a[k * n_ + p], akq = a[k * n_ + q];
                    a[k * n_ + p] = c * akp - s * akq;
                    a[k * n_ + q] = s * akp + c * akq;
                }
                for (size_t k = 0; k < n_; ++k) {
                    const double apk = a[p * n_ + k], aqk = a[q * n_ + k];
                    a[p * n_ + k] = c * apk - s * aqk;
                    a[q * n_ + k] = s * apk + c * aqk;
                }
                for (size_t k = 0; k < n_; ++k) {
                    const double bkp = b_[k * n_ + p], bkq = b_[k * n_ + q];
                    b_[k * n_ + p] = c * bkp - s * bkq;
                    b_[k * n_ + q] = s * bkp + c * bkq;
                }
            }
    }
    for (size_t i = 0; i < n_; ++i) d_[i] = std::sqrt(std::max(a[i * n_ + i], 1e-20));
}

}  // namespace nuka::runtime::app::viewer
