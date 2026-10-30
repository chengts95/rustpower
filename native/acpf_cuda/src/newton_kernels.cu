// 真正的批量 Newton：每个场景独立更新、独立收敛，已收敛 slot 不再工作。

extern "C" __global__ void prepare_newton_voltage_kernel(
    const double* base_v_re, const double* base_v_im,
    double* v_re, double* v_im, double* inv_vmag,
    unsigned long long batch, unsigned long long nbus) {
    const unsigned long long tid = blockIdx.x * static_cast<unsigned long long>(blockDim.x) + threadIdx.x;
    const unsigned long long stride = blockDim.x * static_cast<unsigned long long>(gridDim.x);
    for (unsigned long long p = tid; p < batch * nbus; p += stride) {
        const unsigned long long scenario = p / nbus;
        const double scale = 1.0 + 0.0001 * (scenario % 17);
        const double re = base_v_re[p - scenario * nbus] * scale;
        const double im = base_v_im[p - scenario * nbus] * scale;
        v_re[p] = re;
        v_im[p] = im;
        inv_vmag[p] = rsqrt(re * re + im * im);
    }
}

extern "C" __global__ void newton_power_mismatch_kernel(
    const int* y_col_ptr, const int* y_row_indices, const int* mirror,
    const double* y_re, const double* y_im,
    const double* v_re, const double* v_im,
    const double* sbus_re, const double* sbus_im,
    double* scalc_re, double* scalc_im, double* mismatch, double* rhs,
    const int* mask, unsigned long long batch, unsigned long long nbus, unsigned long long n_active, unsigned long long npq, unsigned long long n) {
    constexpr int warp_size = 32;
    const unsigned long long warp = (blockIdx.x * static_cast<unsigned long long>(blockDim.x) + threadIdx.x) / warp_size;
    const int lane = threadIdx.x & (warp_size - 1);
    if (warp >= batch * nbus) return;
    const unsigned long long scenario = warp / nbus;
    if (mask[scenario] == 0) return;
    const unsigned long long k = warp - scenario * nbus;
    const unsigned long long vector_base = scenario * nbus;
    const unsigned long long ybase = scenario * static_cast<unsigned long long>(y_col_ptr[nbus]);
    double current_re = 0.0;
    double current_im = 0.0;
    for (int p = y_col_ptr[k] + lane; p < y_col_ptr[k + 1]; p += warp_size) {
        const unsigned long long i = static_cast<unsigned long long>(y_row_indices[p]);
        const int q = mirror[p];
        const double vr = v_re[vector_base + i];
        const double vi = v_im[vector_base + i];
        current_re += y_re[ybase + q] * vr - y_im[ybase + q] * vi;
        current_im += y_re[ybase + q] * vi + y_im[ybase + q] * vr;
    }
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        current_re += __shfl_down_sync(0xffffffffu, current_re, offset);
        current_im += __shfl_down_sync(0xffffffffu, current_im, offset);
    }
    if (lane != 0) return;
    const double vr = v_re[vector_base + k];
    const double vi = v_im[vector_base + k];
    const double sr = vr * current_re + vi * current_im;
    const double si = vi * current_re - vr * current_im;
    scalc_re[warp] = sr;
    scalc_im[warp] = si;
    if (k < n_active) {
        const double fp = sr - sbus_re[k];
        mismatch[scenario * n + k] = fp;
        rhs[scenario * n + k] = -fp;
        if (k < npq) {
            const double fq = si - sbus_im[k];
            mismatch[scenario * n + n_active + k] = fq;
            rhs[scenario * n + n_active + k] = -fq;
        }
    }
}

extern "C" __global__ void newton_convergence_kernel(
    const double* mismatch, int* mask, double* norms,
    unsigned long long batch, unsigned long long n, double tolerance) {
    const unsigned long long scenario = blockIdx.x;
    if (scenario >= batch || mask[scenario] <= 0) return;
    double local = 0.0;
    for (unsigned long long row = threadIdx.x; row < n; row += blockDim.x) {
        const double value = fabs(mismatch[scenario * n + row]);
        local = isfinite(value) ? fmax(local, value) : 1e300;
    }
    __shared__ double reduction[256];
    reduction[threadIdx.x] = local;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) reduction[threadIdx.x] = fmax(reduction[threadIdx.x], reduction[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        norms[scenario] = reduction[0];
        if (reduction[0] <= tolerance) mask[scenario] = 0;
    }
}

extern "C" __global__ void newton_jacobian_csr_column_kernel(
    const int* y_col_ptr, const int* y_row_indices, const int* mirror,
    const double* y_re, const double* y_im,
    const double* v_re, const double* v_im, const double* inv_vmag,
    const double* scalc_re, const double* scalc_im,
    const int* pq_ends, const int* active_ends, const int* diag_ptrs, const int* j_col_ptr,
    double* values, const int* mask,
    unsigned long long batch, unsigned long long n_active, unsigned long long npq, unsigned long long nnz, unsigned long long active_ynnz, unsigned long long nbus) {
    const unsigned long long tid = blockIdx.x * static_cast<unsigned long long>(blockDim.x) + threadIdx.x;
    const unsigned long long stride = blockDim.x * static_cast<unsigned long long>(gridDim.x);
    for (unsigned long long task = tid; task < batch * n_active; task += stride) {
        const unsigned long long scenario = task / n_active;
        if (mask[scenario] == 0) continue;
        const unsigned long long k = task - scenario * n_active;
        const int start = y_col_ptr[k];
        const int pq = pq_ends[k];
        const int active = active_ends[k];
        const int diagonal = diag_ptrs[k] - start;
        const unsigned long long vbase = scenario * nbus;
        const unsigned long long jbase = scenario * nnz;
        const unsigned long long first = static_cast<unsigned long long>(j_col_ptr[k]);
        const bool is_pq = k < npq;
        const unsigned long long second = is_pq ? static_cast<unsigned long long>(j_col_ptr[n_active + k]) : 0;
        const double ek = v_re[vbase + k];
        const double fk = -v_im[vbase + k];
        const double inv_k = inv_vmag[vbase + k];
        const double sr = scalc_re[vbase + k];
        const double si = scalc_im[vbase + k];
        for (int t = 0; t < active; ++t) {
            const int p = start + t;
            const unsigned long long i = static_cast<unsigned long long>(y_row_indices[p]);
            const int q = mirror[p];
            const double vi_re = v_re[vbase + i];
            const double fi = -v_im[vbase + i];
            const double va_re = y_re[q] * ek - y_im[q] * fk;
            const double va_im = y_re[q] * fk + y_im[q] * ek;
            const double j11 = fi * va_re - vi_re * va_im;
            const double j21 = -(vi_re * va_re + fi * va_im);
            const double inv = inv_vmag[vbase + i];
            double a = j11;
            double b = -j21 * inv;
            double c = j21;
            double d = j11 * inv;
            if (t == diagonal) {
                a -= si;
                if (is_pq) {
                    b += sr * inv_k;
                    c += sr;
                    d += si * inv_k;
                }
            }
            values[jbase + first + t] = a;
            if (t < pq) values[jbase + first + active + t] = b;
            if (is_pq) {
                values[jbase + second + t] = c;
                if (t < pq) values[jbase + second + active + t] = d;
            }
        }
    }
}

extern "C" __global__ void newton_update_voltage_kernel(
    double* v_re, double* v_im, double* inv_vmag, const double* dx, int* mask,
    unsigned long long batch, unsigned long long nbus, unsigned long long n_active, unsigned long long npq, unsigned long long n) {
    const unsigned long long tid = blockIdx.x * static_cast<unsigned long long>(blockDim.x) + threadIdx.x;
    const unsigned long long stride = blockDim.x * static_cast<unsigned long long>(gridDim.x);
    for (unsigned long long task = tid; task < batch * n_active; task += stride) {
        const unsigned long long scenario = task / n_active;
        if (mask[scenario] == 0) continue;
        const unsigned long long k = task - scenario * n_active;
        const unsigned long long vindex = scenario * nbus + k;
        const double old_re = v_re[vindex];
        const double old_im = v_im[vindex];
        const double angle = atan2(old_im, old_re) + dx[scenario * n + k];
        double magnitude = sqrt(old_re * old_re + old_im * old_im);
        if (k < npq) magnitude += dx[scenario * n + n_active + k];
        double sin_angle, cos_angle;
        sincos(angle, &sin_angle, &cos_angle);
        const double re = magnitude * cos_angle;
        const double im = magnitude * sin_angle;
        if (!(magnitude > 0.0) || !isfinite(re) || !isfinite(im)) {
            mask[scenario] = -1;
            continue;
        }
        v_re[vindex] = re;
        v_im[vindex] = im;
        inv_vmag[vindex] = 1.0 / magnitude;
    }
}

// FP32 只提供牛顿方向；电压、状态和后续残差始终保留 FP64。
extern "C" __global__ void newton_update_voltage_fp32_kernel(
    double* v_re, double* v_im, double* inv_vmag, const float* dx, int* mask,
    unsigned long long batch, unsigned long long nbus, unsigned long long n_active, unsigned long long npq, unsigned long long n) {
    const unsigned long long tid = blockIdx.x * static_cast<unsigned long long>(blockDim.x) + threadIdx.x;
    const unsigned long long stride = blockDim.x * static_cast<unsigned long long>(gridDim.x);
    for (unsigned long long task = tid; task < batch * n_active; task += stride) {
        const unsigned long long scenario = task / n_active;
        if (mask[scenario] == 0) continue;
        const unsigned long long k = task - scenario * n_active;
        const unsigned long long vindex = scenario * nbus + k;
        const double old_re = v_re[vindex];
        const double old_im = v_im[vindex];
        const double angle = atan2(old_im, old_re) + static_cast<double>(dx[scenario * n + k]);
        double magnitude = sqrt(old_re * old_re + old_im * old_im);
        if (k < npq) magnitude += static_cast<double>(dx[scenario * n + n_active + k]);
        double sin_angle, cos_angle;
        sincos(angle, &sin_angle, &cos_angle);
        const double re = magnitude * cos_angle;
        const double im = magnitude * sin_angle;
        if (!(magnitude > 0.0) || !isfinite(re) || !isfinite(im)) {
            mask[scenario] = -1;
            continue;
        }
        v_re[vindex] = re;
        v_im[vindex] = im;
        inv_vmag[vindex] = 1.0 / magnitude;
    }
}


// 当前默认：逐项并行在本矩阵上优于逐列版；列版保留用于后续 warp 执行组实验。
extern "C" __global__ void newton_jacobian_csr_kernel(
    const int* y_col_ptr, const int* y_row_indices, const int* mirror,
    const double* y_re, const double* y_im,
    const double* v_re, const double* v_im, const double* inv_vmag,
    const double* scalc_re, const double* scalc_im,
    const int* pq_ends, const int* active_ends, const int* diag_ptrs, const int* j_col_ptr,
    double* values, const int* mask,
    unsigned long long batch, unsigned long long n_active, unsigned long long npq, unsigned long long nnz, unsigned long long active_ynnz, unsigned long long nbus) {
    const unsigned long long tid = blockIdx.x * static_cast<unsigned long long>(blockDim.x) + threadIdx.x;
    const unsigned long long stride = blockDim.x * static_cast<unsigned long long>(gridDim.x);
    for (unsigned long long task = tid; task < batch * active_ynnz; task += stride) {
        const unsigned long long scenario = task / active_ynnz;
        if (mask[scenario] == 0) continue;
        const int p = static_cast<int>(task - scenario * active_ynnz);
        unsigned long long low = 0;
        unsigned long long high = n_active;
        while (low + 1 < high) {
            const unsigned long long middle = low + (high - low) / 2;
            if (y_col_ptr[middle] <= p) low = middle;
            else high = middle;
        }
        const unsigned long long k = low;
        const int start = y_col_ptr[k];
        const int t = p - start;
        const int pq = pq_ends[k];
        const int active = active_ends[k];
        if (t >= active) continue;
        const unsigned long long i = static_cast<unsigned long long>(y_row_indices[p]);
        const int q = mirror[p];
        const unsigned long long vbase = scenario * nbus;
        const double ek = v_re[vbase + k];
        const double fk = -v_im[vbase + k];
        const double vi_re = v_re[vbase + i];
        const double fi = -v_im[vbase + i];
        const double va_re = y_re[q] * ek - y_im[q] * fk;
        const double va_im = y_re[q] * fk + y_im[q] * ek;
        const double j11 = fi * va_re - vi_re * va_im;
        const double j21 = -(vi_re * va_re + fi * va_im);
        const double inv = inv_vmag[vbase + i];
        const unsigned long long jbase = scenario * nnz;
        const unsigned long long first = static_cast<unsigned long long>(j_col_ptr[k]);
        double a = j11;
        double b = -j21 * inv;
        double c = 0.0;
        double d = 0.0;
        const bool is_pq = k < npq;
        unsigned long long second = 0;
        if (is_pq) {
            second = static_cast<unsigned long long>(j_col_ptr[n_active + k]);
            c = j21;
            d = j11 * inv;
        }
        if (p == diag_ptrs[k]) {
            const double sr = scalc_re[vbase + k];
            const double si = scalc_im[vbase + k];
            const double inv_k = inv_vmag[vbase + k];
            a -= si;
            if (is_pq) {
                b += sr * inv_k;
                c += sr;
                d += si * inv_k;
            }
        }
        values[jbase + first + t] = a;
        if (t < pq) values[jbase + first + active + t] = b;
        if (is_pq) {
            values[jbase + second + t] = c;
            if (t < pq) values[jbase + second + active + t] = d;
        }
    }
}

// 爱莉希雅的小地图：一次连续读取换到列号，之后所有 CSR 偏移都直接计算。
extern "C" __global__ void newton_jacobian_csr_mapped_kernel(
    const int* y_col_ptr, const int* y_row_indices, const int* mirror,
    const double* y_re, const double* y_im,
    const double* v_re, const double* v_im, const double* inv_vmag,
    const double* scalc_re, const double* scalc_im,
    const int* pq_ends, const int* active_ends, const int* diag_ptrs, const int* j_col_ptr,
    double* values, const int* mask, const int* column_of_nnz,
    unsigned long long batch, unsigned long long n_active, unsigned long long npq, unsigned long long nnz, unsigned long long active_ynnz, unsigned long long nbus) {
    const unsigned long long tid = blockIdx.x * static_cast<unsigned long long>(blockDim.x) + threadIdx.x;
    const unsigned long long stride = blockDim.x * static_cast<unsigned long long>(gridDim.x);
    for (unsigned long long task = tid; task < batch * active_ynnz; task += stride) {
        const unsigned long long scenario = task / active_ynnz;
        if (mask[scenario] == 0) continue;
        const int p = static_cast<int>(task - scenario * active_ynnz);
        const unsigned long long k = static_cast<unsigned long long>(column_of_nnz[p]);
        const int t = p - y_col_ptr[k];
        const int pq = pq_ends[k];
        const int active = active_ends[k];
        if (t >= active) continue;
        const unsigned long long i = static_cast<unsigned long long>(y_row_indices[p]);
        const int q = mirror[p];
        const unsigned long long vbase = scenario * nbus;
        const unsigned long long ybase = scenario * static_cast<unsigned long long>(y_col_ptr[nbus]);
        const double ek = v_re[vbase + k];
        const double fk = -v_im[vbase + k];
        const double vi_re = v_re[vbase + i];
        const double fi = -v_im[vbase + i];
        const double va_re = y_re[ybase + q] * ek - y_im[ybase + q] * fk;
        const double va_im = y_re[ybase + q] * fk + y_im[ybase + q] * ek;
        const double j11 = fi * va_re - vi_re * va_im;
        const double j21 = -(vi_re * va_re + fi * va_im);
        const double inv = inv_vmag[vbase + i];
        const unsigned long long jbase = scenario * nnz;
        const unsigned long long first = static_cast<unsigned long long>(j_col_ptr[k]);
        double a = j11;
        double b = -j21 * inv;
        double c = 0.0;
        double d = 0.0;
        const bool is_pq = k < npq;
        unsigned long long second = 0;
        if (is_pq) {
            second = static_cast<unsigned long long>(j_col_ptr[n_active + k]);
            c = j21;
            d = j11 * inv;
        }
        if (p == diag_ptrs[k]) {
            const double sr = scalc_re[vbase + k];
            const double si = scalc_im[vbase + k];
            const double inv_k = inv_vmag[vbase + k];
            a -= si;
            if (is_pq) {
                b += sr * inv_k;
                c += sr;
                d += si * inv_k;
            }
        }
        values[jbase + first + t] = a;
        if (t < pq) values[jbase + first + active + t] = b;
        if (is_pq) {
            values[jbase + second + t] = c;
            if (t < pq) values[jbase + second + active + t] = d;
        }
    }
}

// 爱莉希雅的协作列分组：已知列与 CSR 行段，不再需要二分查找。
struct YbusView {
    const int* col_ptr;
    const int* row_indices;
    const int* mirror;
    const double* re;
    const double* im;
};

struct NewtonStateView {
    const double* v_re;
    const double* v_im;
    const double* inv_vmag;
    const double* scalc_re;
    const double* scalc_im;
};

struct JacobianView {
    const int* pq_ends;
    const int* active_ends;
    const int* diag_ptrs;
    const int* row_ptr;
    double* values;
};

struct JacobianArgs {
    YbusView ybus;
    NewtonStateView state;
    JacobianView jacobian;
    const int* mask;
    int n_active;
    int npq;
    int nbus;
    int batch;
    unsigned long long nnz;
};

struct EdgeDerivative {
    double j11;
    double j21;
};

template<bool NEED_J21>
__device__ __forceinline__ EdgeDerivative evaluate_edge(
    double ek, double fk, double vi_re, double fi, double yr, double yi) {
    const double va_re = yr * ek - yi * fk;
    const double va_im = yr * fk + yi * ek;
    EdgeDerivative result;
    result.j11 = fi * va_re - vi_re * va_im;
    if constexpr (NEED_J21) result.j21 = -(vi_re * va_re + fi * va_im);
    else result.j21 = 0.0;
    return result;
}

template<bool PQ_COLUMN, bool PQ_ROW>
__device__ __forceinline__ void write_edge(
    const JacobianArgs& args, unsigned long long vbase, int p, int t,
    double ek, double fk, double* a, double* b, double* c, double* d) {
    const auto& y = args.ybus;
    const auto& s = args.state;
    const int i = y.row_indices[p];
    const int q = y.mirror[p];
    const double vi_re = s.v_re[vbase + i];
    const double fi = -s.v_im[vbase + i];
    const auto edge = evaluate_edge<PQ_COLUMN || PQ_ROW>(
        ek, fk, vi_re, fi, y.re[q], y.im[q]);
    a[t] = edge.j11;
    if constexpr (PQ_ROW) {
        const double inv = s.inv_vmag[vbase + i];
        b[t] = -edge.j21 * inv;
        if constexpr (PQ_COLUMN) {
            c[t] = edge.j21;
            d[t] = edge.j11 * inv;
        }
    } else if constexpr (PQ_COLUMN) {
        c[t] = edge.j21;
    }
}

template<int LANES, bool PQ_COLUMN>
__device__ __forceinline__ void fill_column_group(
    const JacobianArgs& args, int scenario, int k, int lane) {
    static_assert(LANES == 2 || LANES == 4 || LANES == 8);
    const auto& y = args.ybus;
    const auto& s = args.state;
    const auto& j = args.jacobian;
    const int start = y.col_ptr[k];
    const int pq = j.pq_ends[k];
    const int active = j.active_ends[k];
    const unsigned long long vbase = static_cast<unsigned long long>(scenario) * args.nbus;
    const unsigned long long jbase = static_cast<unsigned long long>(scenario) * args.nnz;
    double* const a = j.values + jbase + j.row_ptr[k];
    double* const b = a + active;
    double* c = nullptr;
    double* d = nullptr;
    if constexpr (PQ_COLUMN) {
        c = j.values + jbase + j.row_ptr[args.n_active + k];
        d = c + active;
    }
    const double ek = s.v_re[vbase + k];
    const double fk = -s.v_im[vbase + k];
    for (int t = lane; t < pq; t += LANES) {
        write_edge<PQ_COLUMN, true>(args, vbase, start + t, t, ek, fk, a, b, c, d);
    }
    for (int t = pq + lane; t < active; t += LANES) {
        write_edge<PQ_COLUMN, false>(args, vbase, start + t, t, ek, fk, a, b, c, d);
    }
    const int diag = j.diag_ptrs[k] - start;
    const int diagonal_lane = PQ_COLUMN ? (diag & (LANES - 1)) : ((diag - pq) & (LANES - 1));
    if (lane == diagonal_lane) {
        const double sr = s.scalc_re[vbase + k];
        const double si = s.scalc_im[vbase + k];
        a[diag] -= si;
        if constexpr (PQ_COLUMN) {
            const double inv_k = s.inv_vmag[vbase + k];
            b[diag] += sr * inv_k;
            c[diag] += sr;
            d[diag] += si * inv_k;
        }
    }
}

template<int LANES>
__device__ __forceinline__ void run_jacobian(const JacobianArgs& args) {
    const int local_column = threadIdx.x / LANES;
    const int lane = threadIdx.x % LANES;
    const int columns_per_block = blockDim.x / LANES;
    const int k = blockIdx.x * columns_per_block + local_column;
    const int scenario = blockIdx.y;
    if (k >= args.n_active || scenario >= args.batch || args.mask[scenario] == 0) return;
    if (k < args.npq) fill_column_group<LANES, true>(args, scenario, k, lane);
    else fill_column_group<LANES, false>(args, scenario, k, lane);
}

extern "C" __global__ void newton_jacobian_csr_l2(JacobianArgs args) {
    run_jacobian<2>(args);
}

extern "C" __global__ void newton_jacobian_csr_l4(JacobianArgs args) {
    run_jacobian<4>(args);
}

extern "C" __global__ void newton_jacobian_csr_l8(JacobianArgs args) {
    run_jacobian<8>(args);
}
