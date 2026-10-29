// 爱莉希雅的 CUDA 小厨房：每个 slot 都是独立系统，绝不让场景之间串味。

extern "C" __global__ void prepare_voltage_kernel(
    const double* base_v_re,
    const double* base_v_im,
    double* v_re,
    double* v_im,
    double* inv_vmag,
    double* expected_x,
    unsigned long long batch,
    unsigned long long nbus,
    unsigned long long n) {
    const unsigned long long tid = blockIdx.x * static_cast<unsigned long long>(blockDim.x) + threadIdx.x;
    const unsigned long long stride = blockDim.x * static_cast<unsigned long long>(gridDim.x);
    const unsigned long long voltage_count = batch * nbus;
    for (unsigned long long p = tid; p < voltage_count; p += stride) {
        const unsigned long long scenario = p / nbus;
        const double scale = 1.0 + 0.0001 * (scenario % 17);
        const double re = base_v_re[p - scenario * nbus] * scale;
        const double im = base_v_im[p - scenario * nbus] * scale;
        v_re[p] = re;
        v_im[p] = im;
        inv_vmag[p] = rsqrt(re * re + im * im);
    }
    const unsigned long long vector_count = batch * n;
    for (unsigned long long p = tid; p < vector_count; p += stride) {
        const unsigned long long scenario = p / n;
        const unsigned long long row = p - scenario * n;
        expected_x[p] = 0.5 + 0.01 * ((row + 3 * scenario) % 11);
    }
}

// 借 CSC 的镜像位置遍历 Y 的一行，一次得到 S_calc 和 AC mismatch。
extern "C" __global__ void batch_power_mismatch_kernel(
    const int* y_col_ptr,
    const int* y_row_indices,
    const int* mirror,
    const double* y_re,
    const double* y_im,
    const double* v_re,
    const double* v_im,
    const double* sbus_re,
    const double* sbus_im,
    double* scalc_re,
    double* scalc_im,
    double* mismatch,
    unsigned long long batch,
    unsigned long long nbus,
    unsigned long long n_active,
    unsigned long long npq,
    unsigned long long n) {
    constexpr int warp_size = 32;
    const unsigned long long warp = (blockIdx.x * static_cast<unsigned long long>(blockDim.x) + threadIdx.x) / warp_size;
    const int lane = threadIdx.x & (warp_size - 1);
    if (warp >= batch * nbus) return;
    const unsigned long long scenario = warp / nbus;
    const unsigned long long k = warp - scenario * nbus;
    const unsigned long long vector_base = scenario * nbus;
    double current_re = 0.0;
    double current_im = 0.0;
    for (int p = y_col_ptr[k] + lane; p < y_col_ptr[k + 1]; p += warp_size) {
        const unsigned long long i = static_cast<unsigned long long>(y_row_indices[p]);
        const int q = mirror[p];
        const double vr = v_re[vector_base + i];
        const double vi = v_im[vector_base + i];
        current_re += y_re[q] * vr - y_im[q] * vi;
        current_im += y_re[q] * vi + y_im[q] * vr;
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
        mismatch[scenario * n + k] = sr - sbus_re[k];
        if (k < npq) mismatch[scenario * n + n_active + k] = si - sbus_im[k];
    }
}

// JacobianOperator::fill::<true, false> 的逐列并行版。
// 列结构完全确定：一个 thread 顺序扫一列，不做二分查找，也没有对角线原子操作。
extern "C" __global__ void batch_jacobian_csr_column_kernel(
    const int* y_col_ptr,
    const int* y_row_indices,
    const int* mirror,
    const double* y_re,
    const double* y_im,
    const double* v_re,
    const double* v_im,
    const double* inv_vmag,
    const double* scalc_re,
    const double* scalc_im,
    const int* pq_ends,
    const int* active_ends,
    const int* diag_ptrs,
    const int* j_col_ptr,
    double* values,
    unsigned long long batch,
    unsigned long long n_active,
    unsigned long long npq,
    unsigned long long nnz,
    unsigned long long active_ynnz,
    unsigned long long nbus) {
    const unsigned long long tid = blockIdx.x * static_cast<unsigned long long>(blockDim.x) + threadIdx.x;
    const unsigned long long stride = blockDim.x * static_cast<unsigned long long>(gridDim.x);
    for (unsigned long long task = tid; task < batch * n_active; task += stride) {
        const unsigned long long scenario = task / n_active;
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

// 一个 warp 负责一个场景中的一行。B 个 RHS 同时生成，没有跨场景归约。
extern "C" __global__ void batch_csr_matvec_kernel(
    const int* row_ptr,
    const int* col_indices,
    const double* values,
    const double* x,
    double* y,
    unsigned long long batch,
    unsigned long long n,
    unsigned long long nnz) {
    constexpr int warp_size = 32;
    const unsigned long long warp = (blockIdx.x * static_cast<unsigned long long>(blockDim.x) + threadIdx.x) / warp_size;
    const int lane = threadIdx.x & (warp_size - 1);
    if (warp >= batch * n) return;
    const unsigned long long scenario = warp / n;
    const unsigned long long row = warp - scenario * n;
    const unsigned long long value_base = scenario * nnz;
    const unsigned long long vector_base = scenario * n;
    double sum = 0.0;
    for (int p = row_ptr[row] + lane; p < row_ptr[row + 1]; p += warp_size) {
        sum += values[value_base + p] * x[vector_base + col_indices[p]];
    }
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if (lane == 0) y[warp] = sum;
}


// 当前默认：逐项并行在本矩阵上优于逐列版，保留直接索引以避免重复计算列内共享量。
extern "C" __global__ void batch_jacobian_csr_kernel(
    const int* y_col_ptr, const int* y_row_indices, const int* mirror,
    const double* y_re, const double* y_im,
    const double* v_re, const double* v_im, const double* inv_vmag,
    const double* scalc_re, const double* scalc_im,
    const int* pq_ends, const int* active_ends, const int* diag_ptrs, const int* j_col_ptr,
    double* values, unsigned long long batch, unsigned long long n_active, unsigned long long npq, unsigned long long nnz,
    unsigned long long active_ynnz, unsigned long long nbus) {
    const unsigned long long tid = blockIdx.x * static_cast<unsigned long long>(blockDim.x) + threadIdx.x;
    const unsigned long long stride = blockDim.x * static_cast<unsigned long long>(gridDim.x);
    for (unsigned long long task = tid; task < batch * active_ynnz; task += stride) {
        const unsigned long long scenario = task / active_ynnz;
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


extern "C" __global__ void batch_cast_fp64_to_fp32_kernel(
    const double* source, float* destination, unsigned long long count) {
    const unsigned long long tid = blockIdx.x * static_cast<unsigned long long>(blockDim.x) + threadIdx.x;
    const unsigned long long stride = blockDim.x * static_cast<unsigned long long>(gridDim.x);
    for (unsigned long long p = tid; p < count; p += stride) destination[p] = static_cast<float>(source[p]);
}

extern "C" __global__ void batch_csr_matvec_fp32_kernel(
    const int* row_ptr, const int* col_indices, const float* values,
    const float* x, float* y, unsigned long long batch, unsigned long long n, unsigned long long nnz) {
    constexpr int warp_size = 32;
    const unsigned long long warp = (blockIdx.x * static_cast<unsigned long long>(blockDim.x) + threadIdx.x) / warp_size;
    const int lane = threadIdx.x & (warp_size - 1);
    if (warp >= batch * n) return;
    const unsigned long long scenario = warp / n;
    const unsigned long long row = warp - scenario * n;
    const unsigned long long value_base = scenario * nnz;
    const unsigned long long vector_base = scenario * n;
    float sum = 0.0f;
    for (int p = row_ptr[row] + lane; p < row_ptr[row + 1]; p += warp_size) {
        sum = fmaf(values[value_base + p], x[vector_base + col_indices[p]], sum);
    }
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if (lane == 0) y[warp] = sum;
}
