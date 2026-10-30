// One thread owns one (scenario, branch) result: no search and no shared writes.
#include <cuda_runtime_api.h>
#include <cmath>
#include <cstdint>

extern "C" __global__ void branch_two_port_kernel(
    const double* voltage_re, const double* voltage_im,
    const int32_t* from, const int32_t* to,
    const double* y_ff_re, const double* y_ff_im,
    const double* y_tf_re, const double* y_tf_im,
    const double* y_ft_re, const double* y_ft_im,
    const double* y_tt_re, const double* y_tt_im,
    const double* base_i_from, const double* base_i_to,
    const double* inv_line_rating, const double* transformer_loading_scale,
    const uint8_t* in_service, const uint8_t* kind,
    const int32_t* outages, double* output,
    unsigned long long batch, unsigned long long branch_count,
    unsigned long long nbus, double sbase) {
    const auto tid = blockIdx.x * static_cast<unsigned long long>(blockDim.x) + threadIdx.x;
    const auto stride = blockDim.x * static_cast<unsigned long long>(gridDim.x);
    const auto total = batch * branch_count;
    for (auto item = tid; item < total; item += stride) {
        const auto branch = item % branch_count;
        const auto scenario = item / branch_count;
        double* result = output + item * 14;
        if (!in_service[branch] || outages[scenario] == static_cast<int32_t>(branch)) {
            for (int field = 0; field < 14; ++field) result[field] = 0.0;
            continue;
        }
        const auto vf_offset = scenario * nbus + from[branch];
        const auto vt_offset = scenario * nbus + to[branch];
        const double fr = voltage_re[vf_offset], fi = voltage_im[vf_offset];
        const double tr = voltage_re[vt_offset], ti = voltage_im[vt_offset];

        // Match BranchCatalog's column-major [ff, tf, ft, tt] two-port formula.
        const double i0r = (y_ff_re[branch] * fr - y_ff_im[branch] * fi)
                         + (y_ft_re[branch] * tr - y_ft_im[branch] * ti);
        const double i0i = (y_ff_re[branch] * fi + y_ff_im[branch] * fr)
                         + (y_ft_re[branch] * ti + y_ft_im[branch] * tr);
        const double i1r = (y_tf_re[branch] * fr - y_tf_im[branch] * fi)
                         + (y_tt_re[branch] * tr - y_tt_im[branch] * ti);
        const double i1i = (y_tf_re[branch] * fi + y_tf_im[branch] * fr)
                         + (y_tt_re[branch] * ti + y_tt_im[branch] * tr);
        const double pf = (fr * i0r + fi * i0i) * sbase;
        const double qf = (fi * i0r - fr * i0i) * sbase;
        const double pt = (tr * i1r + ti * i1i) * sbase;
        const double qt = (ti * i1r - tr * i1i) * sbase;
        const double i0n = std::sqrt(i0r * i0r + i0i * i0i);
        const double i1n = std::sqrt(i1r * i1r + i1i * i1i);
        const double current_from = i0n * base_i_from[branch];
        const double current_to = i1n * base_i_to[branch];
        const double current = std::fmax(current_from, current_to);
        const double loading = kind[branch] == 0
            ? current * inv_line_rating[branch]
            : std::fmax(i0n, i1n) * transformer_loading_scale[branch];
        constexpr double rad_to_degree = 57.2957795130823208768;
        result[0] = pf;
        result[1] = qf;
        result[2] = pt;
        result[3] = qt;
        result[4] = pf + pt;
        result[5] = qf + qt;
        result[6] = current_from;
        result[7] = current_to;
        result[8] = current;
        result[9] = std::sqrt(fr * fr + fi * fi);
        result[10] = std::sqrt(tr * tr + ti * ti);
        result[11] = std::atan2(fi, fr) * rad_to_degree;
        result[12] = std::atan2(ti, tr) * rad_to_degree;
        result[13] = loading;
    }
}
