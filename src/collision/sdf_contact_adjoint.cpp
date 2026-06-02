// ---------------------------------------------------------------------------
// nuka::collision::sdf_contact_adjoint -- envelope-theorem contact-geometry
// adjoint of the Newton-summed-SDF contact (v0.7 p08-C, Channel 2). Host fp32.
// See sdf_contact_adjoint.hpp for the derivation (envelope theorem; valley-proof;
// no step_curvature; point gradient deliberately absent = Channel 3).
// ---------------------------------------------------------------------------

#include "collision/sdf_contact_adjoint.hpp"

#include <cmath>

namespace nuka::collision {

namespace sdf = nuka::runtime::sdf;
using nuka::math::Vec3;

namespace {

// Gather the 8-corner trilinear stencil at local point p (mirrors
// sparse_sdf_query.cuh::sparse_sdf_sample EXACTLY: same floor, same fractional,
// same corner order). Returns false if any corner is out of band.
struct Stencil {
    bool in_band = false;
    float inv_h = 0.0f;
    float fx = 0.0f, fy = 0.0f, fz = 0.0f;
    float val[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    Vec3 grad[8];  // per-corner cooked gradients
    // weight per corner (canonical (di,dj,dk) order)
    float w[8] = {0, 0, 0, 0, 0, 0, 0, 0};
};

inline int CornerIndex(int di, int dj, int dk) { return (di << 2) | (dj << 1) | dk; }

Stencil GatherStencil(const sdf::SparseSdfDevice& s, Vec3 p) {
    Stencil out;
    if (s.cell_count == 0u || s.voxel_size <= 0.0f) return out;
    const float inv_h = 1.0f / s.voxel_size;
    out.inv_h = inv_h;
    const float gx = (p.x - s.origin.x) * inv_h;
    const float gy = (p.y - s.origin.y) * inv_h;
    const float gz = (p.z - s.origin.z) * inv_h;
    const int32_t i0 = sdf::SdfFloorInt(gx);
    const int32_t j0 = sdf::SdfFloorInt(gy);
    const int32_t k0 = sdf::SdfFloorInt(gz);
    out.fx = gx - static_cast<float>(i0);
    out.fy = gy - static_cast<float>(j0);
    out.fz = gz - static_cast<float>(k0);
    for (int di = 0; di < 2; ++di)
        for (int dj = 0; dj < 2; ++dj)
            for (int dk = 0; dk < 2; ++dk) {
                const int32_t ci = i0 + di, cj = j0 + dj, ck = k0 + dk;
                if (ci < 0 || cj < 0 || ck < 0) return out;  // not in band
                const uint64_t key = sdf::PackSdfCellKey(static_cast<uint32_t>(ci),
                                                         static_cast<uint32_t>(cj),
                                                         static_cast<uint32_t>(ck));
                const uint32_t idx = sdf::FindSdfCell(s, key);
                if (idx >= s.cell_count) return out;
                const int c = CornerIndex(di, dj, dk);
                out.val[c] = s.cell_values[idx];
                out.grad[c] = (s.cell_gradients != nullptr) ? s.cell_gradients[idx]
                                                            : Vec3{0.0f, 0.0f, 0.0f};
            }
    const float wx[2] = {1.0f - out.fx, out.fx};
    const float wy[2] = {1.0f - out.fy, out.fy};
    const float wz[2] = {1.0f - out.fz, out.fz};
    for (int di = 0; di < 2; ++di)
        for (int dj = 0; dj < 2; ++dj)
            for (int dk = 0; dk < 2; ++dk)
                out.w[CornerIndex(di, dj, dk)] = wx[di] * wy[dj] * wz[dk];
    out.in_band = true;
    return out;
}

// Rotate a LOCAL direction to world using the frame's columns (matches
// SdfBodyFrame::DirLocalToWorld).
inline Vec3 DirL2W(const SdfBodyFrame& f, Vec3 l) {
    return {f.rot_col0.x * l.x + f.rot_col1.x * l.y + f.rot_col2.x * l.z,
            f.rot_col0.y * l.x + f.rot_col1.y * l.y + f.rot_col2.y * l.z,
            f.rot_col0.z * l.x + f.rot_col1.z * l.y + f.rot_col2.z * l.z};
}

// R H R^T given the frame columns (world basis) and a local 3x3 H. Implemented as
// world rows R * (H * R^T). R has columns rot_col0/1/2; R^T row r = component r of
// each column. We build R (3x3, row-major) then compute R*H*R^T.
Mat3 RotateMat(const SdfBodyFrame& f, const Mat3& h_local) {
    Mat3 R;  // columns are body axes in world
    R.at(0, 0) = f.rot_col0.x; R.at(0, 1) = f.rot_col1.x; R.at(0, 2) = f.rot_col2.x;
    R.at(1, 0) = f.rot_col0.y; R.at(1, 1) = f.rot_col1.y; R.at(1, 2) = f.rot_col2.y;
    R.at(2, 0) = f.rot_col0.z; R.at(2, 1) = f.rot_col1.z; R.at(2, 2) = f.rot_col2.z;
    // T = H * R^T  (R^T[a][b] = R[b][a])
    Mat3 T;
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) {
            float acc = 0.0f;
            for (int k = 0; k < 3; ++k) acc += h_local.at(a, k) * R.at(b, k);
            T.at(a, b) = acc;
        }
    // out = R * T
    Mat3 out;
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) {
            float acc = 0.0f;
            for (int k = 0; k < 3; ++k) acc += R.at(a, k) * T.at(k, b);
            out.at(a, b) = acc;
        }
    return out;
}

}  // namespace

float sample_sdf_with_hessian(const sdf::SparseSdfDevice& s, Vec3 p_local, Vec3& out_grad_local,
                              Mat3& out_hessian_local) {
    out_grad_local = Vec3{0.0f, 0.0f, 0.0f};
    out_hessian_local = Mat3::Zero();
    const Stencil st = GatherStencil(s, p_local);
    if (!st.in_band) return sdf::SparseSdfDevice::kOutsideBand;

    // phi + interpolated gradient (fixed-order, == sparse_sdf_sample).
    float phi = 0.0f;
    Vec3 g{0.0f, 0.0f, 0.0f};
    for (int c = 0; c < 8; ++c) {
        phi += st.w[c] * st.val[c];
        g.x += st.w[c] * st.grad[c].x;
        g.y += st.w[c] * st.grad[c].y;
        g.z += st.w[c] * st.grad[c].z;
    }
    out_grad_local = g;

    // H_local[a][b] = d g[a]/d p_b = sum_c (dw_c/dp_b) * grad_c[a].
    // dw_c/dp_x = (di? +inv_h : -inv_h) * wy[dj] * wz[dk], etc.
    const float wx[2] = {1.0f - st.fx, st.fx};
    const float wy[2] = {1.0f - st.fy, st.fy};
    const float wz[2] = {1.0f - st.fz, st.fz};
    Mat3 H = Mat3::Zero();
    for (int di = 0; di < 2; ++di)
        for (int dj = 0; dj < 2; ++dj)
            for (int dk = 0; dk < 2; ++dk) {
                const int c = CornerIndex(di, dj, dk);
                const float dwx = (di ? st.inv_h : -st.inv_h) * wy[dj] * wz[dk];
                const float dwy = wx[di] * (dj ? st.inv_h : -st.inv_h) * wz[dk];
                const float dwz = wx[di] * wy[dj] * (dk ? st.inv_h : -st.inv_h);
                // contribution to column b (p_b derivative) of each grad component a
                H.at(0, 0) += dwx * st.grad[c].x; H.at(0, 1) += dwy * st.grad[c].x; H.at(0, 2) += dwz * st.grad[c].x;
                H.at(1, 0) += dwx * st.grad[c].y; H.at(1, 1) += dwy * st.grad[c].y; H.at(1, 2) += dwz * st.grad[c].y;
                H.at(2, 0) += dwx * st.grad[c].z; H.at(2, 1) += dwy * st.grad[c].z; H.at(2, 2) += dwz * st.grad[c].z;
            }
    out_hessian_local = H;
    return phi;
}

TrilinearWeights trilinear_weights(const sdf::SparseSdfDevice& s, Vec3 p_local) {
    TrilinearWeights out;
    const Stencil st = GatherStencil(s, p_local);
    out.in_band = st.in_band;
    if (!st.in_band) return out;
    for (int c = 0; c < 8; ++c) out.w[c] = st.w[c];
    return out;
}

DepthCellValueGrad depth_grad_wrt_cell_values(const sdf::SparseSdfDevice& sdf_a,
                                              const SdfBodyFrame& frame_a,
                                              const sdf::SparseSdfDevice& sdf_b,
                                              const SdfBodyFrame& frame_b,
                                              const SdfContactResult& contact) {
    DepthCellValueGrad out;
    if (!contact.valid || !contact.penetrating) return out;
    const TrilinearWeights wa = trilinear_weights(sdf_a, frame_a.WorldToLocal(contact.point));
    const TrilinearWeights wb = trilinear_weights(sdf_b, frame_b.WorldToLocal(contact.point));
    if (!wa.in_band || !wb.in_band) return out;
    // depth = -(phi_a+phi_b); phi = sum_c w_c value_c => d phi/d value_c = w_c.
    for (int c = 0; c < 8; ++c) {
        out.d_depth_d_cellvalue_a[c] = -wa.w[c];
        out.d_depth_d_cellvalue_b[c] = -wb.w[c];
    }
    out.valid = true;
    return out;
}

DepthPoseGrad depth_grad_wrt_translation(const SdfContactResult& contact) {
    DepthPoseGrad out;
    if (!contact.valid || !contact.penetrating) return out;
    // partial phi_a / partial t_a = -grad_a_world  => partial depth / partial t_a = +grad_a_world.
    out.d_depth_d_translation_a = contact.grad_a_world;
    out.d_depth_d_translation_b = contact.grad_b_world;
    out.valid = true;
    return out;
}

NormalNormalizeJacobian normal_normalize_jacobian(const SdfContactResult& contact) {
    NormalNormalizeJacobian out;
    if (!contact.valid) return out;
    const Vec3 d = contact.grad_b_world - contact.grad_a_world;
    const float len = std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
    if (len < 1.0e-6f) return out;  // degenerate (forward fell back); not differentiable here
    const float inv = 1.0f / len;
    const Vec3 n = d * inv;
    // J_norm = (I - n n^T)/len
    Mat3 j;
    j.at(0, 0) = (1.0f - n.x * n.x) * inv; j.at(0, 1) = (-n.x * n.y) * inv; j.at(0, 2) = (-n.x * n.z) * inv;
    j.at(1, 0) = (-n.y * n.x) * inv;       j.at(1, 1) = (1.0f - n.y * n.y) * inv; j.at(1, 2) = (-n.y * n.z) * inv;
    j.at(2, 0) = (-n.z * n.x) * inv;       j.at(2, 1) = (-n.z * n.y) * inv;       j.at(2, 2) = (1.0f - n.z * n.z) * inv;
    out.j_norm = j;
    out.inv_len = inv;
    out.valid = true;
    return out;
}

NormalCellGradGrad normal_grad_wrt_cell_gradient_b(const sdf::SparseSdfDevice& sdf_b,
                                                   const SdfBodyFrame& frame_b,
                                                   const SdfContactResult& contact) {
    NormalCellGradGrad out;
    const NormalNormalizeJacobian nj = normal_normalize_jacobian(contact);
    if (!nj.valid) return out;
    const auto wb = trilinear_weights(sdf_b, frame_b.WorldToLocal(contact.point));
    if (!wb.in_band) return out;
    // d = grad_b_world - grad_a_world; grad_b_world = R_b * sum_c w_c cell_grad_b[c].
    // partial grad_b_world / partial cell_grad_b[c] (local axis e) = R_b (w_c e),
    // so partial d / partial cell_grad_b[c] along local axis e = R_b * (w_c e),
    // and dn = J_norm * that. The cooked cell-gradients are LOCAL-frame, so the
    // "axis" here is the LOCAL gradient axis rotated to world by R_b.
    for (int c = 0; c < 8; ++c)
        for (int e = 0; e < 3; ++e) {
            Vec3 local_dir{e == 0 ? 1.0f : 0.0f, e == 1 ? 1.0f : 0.0f, e == 2 ? 1.0f : 0.0f};
            Vec3 dgrad_world = DirL2W(frame_b, local_dir * wb.w[c]);  // R_b (w_c e)
            out.dn_b[c][e] = nj.j_norm.Mul(dgrad_world);              // J_norm * (+R_b w_c e)
        }
    out.valid = true;
    return out;
}

NormalPoseFrozenTerm normal_pose_frozen_term(const sdf::SparseSdfDevice& sdf_a,
                                             const SdfBodyFrame& frame_a,
                                             const sdf::SparseSdfDevice& sdf_b,
                                             const SdfBodyFrame& frame_b,
                                             const SdfContactResult& contact) {
    NormalPoseFrozenTerm out;
    const NormalNormalizeJacobian nj = normal_normalize_jacobian(contact);
    if (!nj.valid) return out;

    // Field Hessians at p* (local).
    Vec3 ga_l, gb_l;
    Mat3 ha_l, hb_l;
    const float pa = sample_sdf_with_hessian(sdf_a, frame_a.WorldToLocal(contact.point), ga_l, ha_l);
    const float pb = sample_sdf_with_hessian(sdf_b, frame_b.WorldToLocal(contact.point), gb_l, hb_l);
    if (pa >= sdf::SparseSdfDevice::kOutsideBand * 0.5f ||
        pb >= sdf::SparseSdfDevice::kOutsideBand * 0.5f)
        return out;

    // d(grad_a_world)/d t_a = R_a H_a_local R_a^T * (dx/dt where dx = -R_a^T dt -> the
    //   R_a H R_a^T already folds the two R's; the -R_a^T is the dx, R_a maps back).
    //   grad_a_world = R_a g_local(R_a^T(p - t_a)); d/d t_a = R_a H_a (-R_a^T) = -R_a H_a R_a^T.
    // d = grad_b_world - grad_a_world, so:
    //   partial d / partial t_a = -(-R_a H_a R_a^T) = +R_a H_a R_a^T
    //   partial d / partial t_b =  (-R_b H_b R_b^T) = -R_b H_b R_b^T
    const Mat3 RaHaRaT = RotateMat(frame_a, ha_l);
    const Mat3 RbHbRbT = RotateMat(frame_b, hb_l);
    // dn = J_norm * (partial d / partial t)
    auto compose = [&](const Mat3& dd, float sign) {
        Mat3 res;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) {
                float acc = 0.0f;
                for (int k = 0; k < 3; ++k) acc += nj.j_norm.at(r, k) * (sign * dd.at(k, c));
                res.at(r, c) = acc;
            }
        return res;
    };
    out.frozen_d_normal_d_translation_a = compose(RaHaRaT, +1.0f);
    out.frozen_d_normal_d_translation_b = compose(RbHbRbT, -1.0f);
    out.valid = true;
    return out;
}

}  // namespace nuka::collision
