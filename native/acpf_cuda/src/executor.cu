// RustPower production-bound C ABI: direct CUDA/cuDSS calls, no PTX or NVRTC.
#include "../include/acpf_cuda.h"

#include <cuda_runtime_api.h>
#include <cudss.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

extern "C" __global__ void newton_power_mismatch_kernel(
    const int*, const int*, const int*, const double*, const double*,
    const double*, const double*, const double*, const double*,
    double*, double*, double*, double*, const int*,
    unsigned long long, unsigned long long, unsigned long long,
    unsigned long long, unsigned long long);
extern "C" __global__ void newton_convergence_kernel(
    const double*, int*, double*, unsigned long long, unsigned long long, double);
extern "C" __global__ void newton_jacobian_csr_mapped_kernel(
    const int*, const int*, const int*, const double*, const double*,
    const double*, const double*, const double*, const double*, const double*,
    const int*, const int*, const int*, const int*, double*, const int*, const int*,
    unsigned long long, unsigned long long, unsigned long long,
    unsigned long long, unsigned long long, unsigned long long);
extern "C" __global__ void newton_update_voltage_kernel(
    double*, double*, double*, const double*, int*,
    unsigned long long, unsigned long long, unsigned long long,
    unsigned long long, unsigned long long);
extern "C" __global__ void branch_two_port_kernel(
    const double*, const double*, const int32_t*, const int32_t*,
    const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*,
    const uint8_t*, const uint8_t*, const int32_t*, double*,
    unsigned long long, unsigned long long, unsigned long long, double);

namespace {

thread_local std::string creation_error;

struct Failure : std::runtime_error {
    int code;
    Failure(int code_, const std::string& message) : std::runtime_error(message), code(code_) {}
};

void require(bool condition, const char* message) {
    if (!condition) throw Failure(RP_CUDA_INVALID_ARGUMENT, message);
}

void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw Failure(RP_CUDA_CUDA_ERROR, std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

void cudss_check(cudssStatus_t status, const char* operation) {
    if (status != CUDSS_STATUS_SUCCESS) {
        throw Failure(RP_CUDA_CUDSS_ERROR,
                      std::string(operation) + ": cuDSS status " + std::to_string(status));
    }
}

unsigned grid_for(std::size_t threads, unsigned block) {
    const auto blocks = (threads + block - 1) / block;
    require(blocks > 0 && blocks <= std::numeric_limits<unsigned>::max(), "CUDA grid is too large");
    return static_cast<unsigned>(blocks);
}

struct DeviceBuffer {
    void* pointer{};
    std::size_t bytes{};

    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t bytes_) : bytes(bytes_) {
        require(bytes > 0, "zero-sized device allocation");
        cuda_check(cudaMalloc(&pointer, bytes), "cudaMalloc");
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept
        : pointer(std::exchange(other.pointer, nullptr)), bytes(std::exchange(other.bytes, 0)) {}
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            cudaFree(pointer);
            pointer = std::exchange(other.pointer, nullptr);
            bytes = std::exchange(other.bytes, 0);
        }
        return *this;
    }
    ~DeviceBuffer() { cudaFree(pointer); }

    void upload(const void* source, std::size_t count) {
        require(source && count <= bytes, "invalid device upload");
        cuda_check(cudaMemcpy(pointer, source, count, cudaMemcpyHostToDevice), "cudaMemcpy H2D");
    }
    void download(void* destination, std::size_t count) const {
        require(destination && count <= bytes, "invalid device download");
        cuda_check(cudaMemcpy(destination, pointer, count, cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
    }
};

// Every scenario owns its numerical Ybus values; patterns remain shared.
__global__ void executor_prepare_voltage_kernel(
    const double* base_re, const double* base_im,
    double* voltage_re, double* voltage_im, double* inverse_magnitude,
    const double* base_y_re, const double* base_y_im, double* y_re, double* y_im,
    unsigned long long batch, unsigned long long nbus, unsigned long long ynnz) {
    const auto tid = blockIdx.x * static_cast<unsigned long long>(blockDim.x) + threadIdx.x;
    const auto stride = blockDim.x * static_cast<unsigned long long>(gridDim.x);
    for (auto p = tid; p < batch * nbus; p += stride) {
        const auto local = p % nbus;
        const double re = base_re[local];
        const double im = base_im[local];
        voltage_re[p] = re;
        voltage_im[p] = im;
        inverse_magnitude[p] = rsqrt(re * re + im * im);
    }
    for (auto p = tid; p < batch * ynnz; p += stride) {
        const auto local = p % ynnz;
        y_re[p] = base_y_re[local];
        y_im[p] = base_y_im[local];
    }
}

__global__ void executor_apply_outages_kernel(
    double* y_re, double* y_im, const int32_t* counts, const int32_t* slots,
    const double* patch_re, const double* patch_im,
    unsigned long long batch, unsigned long long ynnz) {
    const auto tid = blockIdx.x * static_cast<unsigned long long>(blockDim.x) + threadIdx.x;
    const auto stride = blockDim.x * static_cast<unsigned long long>(gridDim.x);
    for (auto p = tid; p < batch * 4; p += stride) {
        const auto scenario = p / 4;
        const auto local = p % 4;
        if (local >= static_cast<unsigned long long>(counts[scenario])) continue;
        const auto slot = static_cast<unsigned long long>(slots[p]);
        y_re[scenario * ynnz + slot] = patch_re[p];
        y_im[scenario * ynnz + slot] = patch_im[p];
    }
}

} // namespace

struct RpCudaBranchCatalog {
    enum Column { FROM, TO, Y_FF_RE, Y_FF_IM, Y_TF_RE, Y_TF_IM, Y_FT_RE, Y_FT_IM,
                  Y_TT_RE, Y_TT_IM, BASE_FROM, BASE_TO, INV_RATING, TRAFO_SCALE,
                  IN_SERVICE, KIND, COLUMN_COUNT };
    int nbus{}, branch_count{};
    double sbase{};
    DeviceBuffer columns[COLUMN_COUNT];

    RpCudaBranchCatalog(const RpCudaBranchCatalogV1& input)
        : nbus(input.nbus), branch_count(input.branch_count), sbase(input.sbase) {
        require(nbus > 0 && branch_count > 0 && std::isfinite(sbase) && sbase > 0,
                "invalid branch catalog dimensions");
        const void* host[COLUMN_COUNT] = {
            input.from, input.to, input.y_ff_re, input.y_ff_im, input.y_tf_re, input.y_tf_im,
            input.y_ft_re, input.y_ft_im, input.y_tt_re, input.y_tt_im, input.base_i_from_ka,
            input.base_i_to_ka, input.inv_line_rating, input.transformer_loading_scale,
            input.in_service, input.kind,
        };
        const auto count = static_cast<std::size_t>(branch_count);
        for (int column = 0; column < COLUMN_COUNT; ++column) {
            require(host[column] != nullptr, "null branch catalog array");
            const auto width = column <= TO ? sizeof(int32_t)
                : column >= IN_SERVICE ? sizeof(uint8_t) : sizeof(double);
            columns[column] = DeviceBuffer(count * width);
            columns[column].upload(host[column], columns[column].bytes);
        }
    }
};

struct RpCudaExecutor {
    int nbus{}, npv{}, npq{}, n_active{}, n{}, batch{}, solver_batch{};
    std::size_t nnz{}, ynnz{}, active_ynnz{};
    std::string last_error;

    DeviceBuffer j_row_ptr, j_col_indices;
    DeviceBuffer y_col_ptr, y_row_indices, mirror, base_y_re, base_y_im, y_re, y_im;
    DeviceBuffer base_v_re, base_v_im, sbus_re, sbus_im;
    DeviceBuffer pq_ends, active_ends, diag_ptrs, column_of_nnz;
    DeviceBuffer values, v_re, v_im, inv_vmag, scalc_re, scalc_im;
    DeviceBuffer mismatch, rhs, solution, mask, norms;
    cudaStream_t stream{};

    cudssHandle_t handle{};
    cudssConfig_t config{};
    cudssData_t data{};
    cudssMatrix_t matrix{}, rhs_matrix{}, solution_matrix{};

    RpCudaExecutor(const RpCudaBaselineV1& base, int batch_)
        : nbus(base.nbus), npv(base.npv), npq(base.npq), n_active(base.npv + base.npq),
          n(base.npv + 2 * base.npq), batch(batch_), solver_batch(std::max(batch_, 2)),
          nnz(base.j_nnz), ynnz(base.y_nnz),
          j_row_ptr((static_cast<std::size_t>(n) + 1) * sizeof(int)),
          j_col_indices(nnz * sizeof(int)),
          y_col_ptr((static_cast<std::size_t>(nbus) + 1) * sizeof(int)),
          y_row_indices(ynnz * sizeof(int)), mirror(ynnz * sizeof(int)),
          base_y_re(ynnz * sizeof(double)), base_y_im(ynnz * sizeof(double)),
          base_v_re(static_cast<std::size_t>(nbus) * sizeof(double)),
          base_v_im(static_cast<std::size_t>(nbus) * sizeof(double)),
          sbus_re(static_cast<std::size_t>(nbus) * sizeof(double)),
          sbus_im(static_cast<std::size_t>(nbus) * sizeof(double)),
          pq_ends(static_cast<std::size_t>(n_active) * sizeof(int)),
          active_ends(static_cast<std::size_t>(n_active) * sizeof(int)),
          diag_ptrs(static_cast<std::size_t>(n_active) * sizeof(int)) {
        require(nbus > 0 && npv >= 0 && npq > 0 && n_active > 0 && n_active < nbus,
                "invalid bus partition");
        require(batch > 0 && nnz > 0 && ynnz > 0, "invalid batch or sparse dimensions");
        require(base.j_row_ptr[n] == base.j_nnz && base.y_col_ptr[nbus] == base.y_nnz,
                "sparse offsets do not match dimensions");
        active_ynnz = static_cast<std::size_t>(base.y_col_ptr[n_active]);
        require(active_ynnz > 0 && active_ynnz <= ynnz, "invalid active Ybus extent");
        // cuDSS 1.0 produces a non-finite solution for a one-system solve on this
        // path, with and without UBATCH_SIZE. Pad B=1 with an identical shadow case.
        const auto b = static_cast<std::size_t>(solver_batch);
        require(b <= std::numeric_limits<std::size_t>::max() / nnz &&
                b <= std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(n) &&
                b <= std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(nbus) &&
                b <= std::numeric_limits<std::size_t>::max() / ynnz,
                "batch allocation size overflow");

        column_of_nnz = DeviceBuffer(active_ynnz * sizeof(int));
        y_re = DeviceBuffer(b * ynnz * sizeof(double));
        y_im = DeviceBuffer(b * ynnz * sizeof(double));
        values = DeviceBuffer(b * nnz * sizeof(double));
        v_re = DeviceBuffer(b * nbus * sizeof(double));
        v_im = DeviceBuffer(b * nbus * sizeof(double));
        inv_vmag = DeviceBuffer(b * nbus * sizeof(double));
        scalc_re = DeviceBuffer(b * nbus * sizeof(double));
        scalc_im = DeviceBuffer(b * nbus * sizeof(double));
        mismatch = DeviceBuffer(b * n * sizeof(double));
        rhs = DeviceBuffer(b * n * sizeof(double));
        solution = DeviceBuffer(b * n * sizeof(double));
        mask = DeviceBuffer(b * sizeof(int));
        norms = DeviceBuffer(b * sizeof(double));

        j_row_ptr.upload(base.j_row_ptr, j_row_ptr.bytes);
        j_col_indices.upload(base.j_col_indices, j_col_indices.bytes);
        y_col_ptr.upload(base.y_col_ptr, y_col_ptr.bytes);
        y_row_indices.upload(base.y_row_indices, y_row_indices.bytes);
        mirror.upload(base.mirror, mirror.bytes);
        base_y_re.upload(base.y_re, base_y_re.bytes);
        base_y_im.upload(base.y_im, base_y_im.bytes);
        base_v_re.upload(base.initial_v_re, base_v_re.bytes);
        base_v_im.upload(base.initial_v_im, base_v_im.bytes);
        sbus_re.upload(base.sbus_re, sbus_re.bytes);
        sbus_im.upload(base.sbus_im, sbus_im.bytes);
        pq_ends.upload(base.pq_ends, pq_ends.bytes);
        active_ends.upload(base.active_ends, active_ends.bytes);
        diag_ptrs.upload(base.diag_ptrs, diag_ptrs.bytes);

        std::vector<int> columns(active_ynnz);
        for (int k = 0; k < n_active; ++k) {
            for (int p = base.y_col_ptr[k]; p < base.y_col_ptr[k + 1]; ++p) {
                columns[static_cast<std::size_t>(p)] = k;
            }
        }
        column_of_nnz.upload(columns.data(), column_of_nnz.bytes);
        std::vector<int> ones(b, 1);
        mask.upload(ones.data(), mask.bytes);
        cuda_check(cudaMemset(solution.pointer, 0, solution.bytes), "clear solution");
        cuda_check(cudaStreamCreate(&stream), "cudaStreamCreate");
    }

    void initialize_solver() {
        cudss_check(cudssCreate(&handle), "cudssCreate");
        cudss_check(cudssConfigCreate(&config), "cudssConfigCreate");
        cudss_check(cudssDataCreate(handle, &data), "cudssDataCreate");
        cudss_check(cudssSetStream(handle, stream), "cudssSetStream");
        cudss_check(cudssConfigSet(config, CUDSS_CONFIG_UBATCH_SIZE,
                                   &solver_batch, sizeof(solver_batch)),
                    "set uniform batch size");
        cudss_check(cudssDataSet(handle, data, CUDSS_DATA_UBATCH_MASK, mask.pointer, mask.bytes),
                    "set initial batch mask");
        cudss_check(cudssMatrixCreateCsr(
                        &matrix, n, n, nnz, j_row_ptr.pointer, nullptr, j_col_indices.pointer,
                        values.pointer, CUDA_R_32I, CUDA_R_32I, CUDA_R_64F,
                        CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO),
                    "create batch Jacobian");
        cudss_check(cudssMatrixCreateDn(&rhs_matrix, n, 1, n, rhs.pointer, CUDA_R_64F,
                                       CUDSS_LAYOUT_COL_MAJOR),
                    "create batch RHS");
        cudss_check(cudssMatrixCreateDn(&solution_matrix, n, 1, n, solution.pointer, CUDA_R_64F,
                                       CUDSS_LAYOUT_COL_MAJOR),
                    "create batch solution");
        cudss_check(cudssExecute(handle, CUDSS_PHASE_ANALYSIS, config, data,
                                 matrix, solution_matrix, rhs_matrix),
                    "cuDSS batch analysis");
        cuda_check(cudaStreamSynchronize(stream), "synchronize batch analysis");
    }

    ~RpCudaExecutor() {
        if (matrix) cudssMatrixDestroy(matrix);
        if (rhs_matrix) cudssMatrixDestroy(rhs_matrix);
        if (solution_matrix) cudssMatrixDestroy(solution_matrix);
        if (data) cudssDataDestroy(handle, data);
        if (config) cudssConfigDestroy(config);
        if (handle) cudssDestroy(handle);
        if (stream) cudaStreamDestroy(stream);
    }

    void solve(const RpCudaSolveOptionsV1& options, const RpCudaOutageBatchV1* outages,
               int* case_status, int* iterations, double* residuals,
               double* voltage_re, double* voltage_im) {
        const auto b = static_cast<std::size_t>(solver_batch);
        std::vector<int> host_mask(b, 1), host_iterations(b, 0), host_status(b, RP_CUDA_CASE_CONVERGED);
        std::vector<double> host_norms(b, std::numeric_limits<double>::infinity());
        mask.upload(host_mask.data(), mask.bytes);

        executor_prepare_voltage_kernel<<<65535, 256, 0, stream>>>(
            static_cast<const double*>(base_v_re.pointer), static_cast<const double*>(base_v_im.pointer),
            static_cast<double*>(v_re.pointer), static_cast<double*>(v_im.pointer),
            static_cast<double*>(inv_vmag.pointer),
            static_cast<const double*>(base_y_re.pointer), static_cast<const double*>(base_y_im.pointer),
            static_cast<double*>(y_re.pointer), static_cast<double*>(y_im.pointer), b, nbus, ynnz);
        cuda_check(cudaGetLastError(), "launch exact voltage and Ybus preparation");
        if (outages) {
            std::vector<int32_t> counts(b, 0), slots(b * 4, 0);
            std::vector<double> patch_re(b * 4, 0.0), patch_im(b * 4, 0.0);
            std::copy_n(outages->patch_counts, batch, counts.begin());
            std::copy_n(outages->slots, static_cast<std::size_t>(batch) * 4, slots.begin());
            std::copy_n(outages->y_re, static_cast<std::size_t>(batch) * 4, patch_re.begin());
            std::copy_n(outages->y_im, static_cast<std::size_t>(batch) * 4, patch_im.begin());
            for (int scenario = 0; scenario < batch; ++scenario) {
                require(counts[scenario] >= 0 && counts[scenario] <= 4, "invalid outage patch count");
                for (int k = 0; k < counts[scenario]; ++k) {
                    const auto slot = slots[static_cast<std::size_t>(scenario) * 4 + k];
                    require(slot >= 0 && static_cast<std::size_t>(slot) < ynnz,
                            "outage Ybus slot is out of bounds");
                }
            }
            DeviceBuffer d_counts(b * sizeof(int32_t)), d_slots(b * 4 * sizeof(int32_t));
            DeviceBuffer d_re(b * 4 * sizeof(double)), d_im(b * 4 * sizeof(double));
            d_counts.upload(counts.data(), d_counts.bytes);
            d_slots.upload(slots.data(), d_slots.bytes);
            d_re.upload(patch_re.data(), d_re.bytes);
            d_im.upload(patch_im.data(), d_im.bytes);
            executor_apply_outages_kernel<<<grid_for(b * 4, 256), 256, 0, stream>>>(
                static_cast<double*>(y_re.pointer), static_cast<double*>(y_im.pointer),
                static_cast<const int32_t*>(d_counts.pointer), static_cast<const int32_t*>(d_slots.pointer),
                static_cast<const double*>(d_re.pointer), static_cast<const double*>(d_im.pointer), b, ynnz);
            cuda_check(cudaGetLastError(), "apply scenario Ybus outage patches");
        }

        bool failed_mask = false;
        for (int iteration = 0; iteration <= options.max_iterations; ++iteration) {
            newton_power_mismatch_kernel<<<grid_for(b * nbus * 32, 256), 256, 0, stream>>>(
                static_cast<const int*>(y_col_ptr.pointer), static_cast<const int*>(y_row_indices.pointer),
                static_cast<const int*>(mirror.pointer), static_cast<const double*>(y_re.pointer),
                static_cast<const double*>(y_im.pointer), static_cast<const double*>(v_re.pointer),
                static_cast<const double*>(v_im.pointer), static_cast<const double*>(sbus_re.pointer),
                static_cast<const double*>(sbus_im.pointer), static_cast<double*>(scalc_re.pointer),
                static_cast<double*>(scalc_im.pointer), static_cast<double*>(mismatch.pointer),
                static_cast<double*>(rhs.pointer), static_cast<const int*>(mask.pointer),
                b, nbus, n_active, npq, n);
            newton_convergence_kernel<<<solver_batch, 256, 0, stream>>>(
                static_cast<const double*>(mismatch.pointer), static_cast<int*>(mask.pointer),
                static_cast<double*>(norms.pointer), b, n, options.tolerance);
            mask.download(host_mask.data(), mask.bytes);
            norms.download(host_norms.data(), norms.bytes);

            int active_count = 0;
            for (int scenario = 0; scenario < solver_batch; ++scenario) {
                auto& state = host_mask[scenario];
                if (state < 0 || !std::isfinite(host_norms[scenario])) {
                    state = 0;
                    host_status[scenario] = RP_CUDA_CASE_NON_FINITE;
                    failed_mask = true;
                } else if (state == 1) {
                    ++active_count;
                    ++host_iterations[scenario];
                }
            }
            if (failed_mask) {
                mask.upload(host_mask.data(), mask.bytes);
                failed_mask = false;
            }
            if (active_count == 0) break;
            if (iteration == options.max_iterations) {
                for (int scenario = 0; scenario < solver_batch; ++scenario) {
                    if (host_mask[scenario] == 1) {
                        host_status[scenario] = RP_CUDA_CASE_MAX_ITERATIONS;
                        --host_iterations[scenario];
                    }
                }
                break;
            }

            newton_jacobian_csr_mapped_kernel<<<65535, 256, 0, stream>>>(
                static_cast<const int*>(y_col_ptr.pointer), static_cast<const int*>(y_row_indices.pointer),
                static_cast<const int*>(mirror.pointer), static_cast<const double*>(y_re.pointer),
                static_cast<const double*>(y_im.pointer), static_cast<const double*>(v_re.pointer),
                static_cast<const double*>(v_im.pointer), static_cast<const double*>(inv_vmag.pointer),
                static_cast<const double*>(scalc_re.pointer), static_cast<const double*>(scalc_im.pointer),
                static_cast<const int*>(pq_ends.pointer), static_cast<const int*>(active_ends.pointer),
                static_cast<const int*>(diag_ptrs.pointer), static_cast<const int*>(j_row_ptr.pointer),
                static_cast<double*>(values.pointer), static_cast<const int*>(mask.pointer),
                static_cast<const int*>(column_of_nnz.pointer),
                b, n_active, npq, nnz, active_ynnz, nbus);
            cudss_check(cudssDataSet(handle, data, CUDSS_DATA_UBATCH_MASK, mask.pointer, mask.bytes),
                        "refresh active batch mask");
            const int phase = iteration == 0 ? CUDSS_PHASE_FACTORIZATION : CUDSS_PHASE_REFACTORIZATION;
            cudss_check(cudssExecute(handle, phase, config, data, matrix, solution_matrix, rhs_matrix),
                        "cuDSS Newton factorization");
            cudss_check(cudssExecute(handle, CUDSS_PHASE_SOLVE, config, data,
                                     matrix, solution_matrix, rhs_matrix),
                        "cuDSS Newton solve");
            newton_update_voltage_kernel<<<65535, 256, 0, stream>>>(
                static_cast<double*>(v_re.pointer), static_cast<double*>(v_im.pointer),
                static_cast<double*>(inv_vmag.pointer), static_cast<const double*>(solution.pointer),
                static_cast<int*>(mask.pointer), b, nbus, n_active, npq, n);
            cuda_check(cudaGetLastError(), "launch Newton update");
        }
        cuda_check(cudaStreamSynchronize(stream), "synchronize Newton batch");
        std::copy_n(host_status.begin(), batch, case_status);
        std::copy_n(host_iterations.begin(), batch, iterations);
        std::copy_n(host_norms.begin(), batch, residuals);
        if (solver_batch == batch) {
            v_re.download(voltage_re, v_re.bytes);
            v_im.download(voltage_im, v_im.bytes);
        } else {
            std::vector<double> padded_re(b * nbus), padded_im(b * nbus);
            v_re.download(padded_re.data(), v_re.bytes);
            v_im.download(padded_im.data(), v_im.bytes);
            std::copy_n(padded_re.begin(), batch * nbus, voltage_re);
            std::copy_n(padded_im.begin(), batch * nbus, voltage_im);
        }
    }

    void postprocess_branches(const RpCudaBranchCatalog& catalog, const int32_t* outage_indices,
                              double* output, RpCudaPostprocessStatsV1* stats) {
        require(catalog.nbus == nbus, "branch catalog bus count differs from executor");
        const auto b = static_cast<std::size_t>(solver_batch);
        const auto branches = static_cast<std::size_t>(catalog.branch_count);
        require(branches <= std::numeric_limits<std::size_t>::max() / b / 14,
                "branch output allocation overflow");
        std::vector<int32_t> host_outages(b, -1);
        std::copy_n(outage_indices, batch, host_outages.begin());
        for (int scenario = 0; scenario < batch; ++scenario) {
            require(host_outages[scenario] >= -1 && host_outages[scenario] < catalog.branch_count,
                    "branch outage index is out of bounds");
        }
        DeviceBuffer outages(b * sizeof(int32_t));
        DeviceBuffer flows(b * branches * 14 * sizeof(double));
        outages.upload(host_outages.data(), outages.bytes);

        cudaEvent_t events[4]{};
        for (auto& event : events) cuda_check(cudaEventCreate(&event), "create branch timing event");
        cuda_check(cudaEventRecord(events[0], stream), "record branch kernel start");
        branch_two_port_kernel<<<grid_for(b * branches, 256), 256, 0, stream>>>(
            static_cast<const double*>(v_re.pointer), static_cast<const double*>(v_im.pointer),
            static_cast<const int32_t*>(catalog.columns[RpCudaBranchCatalog::FROM].pointer),
            static_cast<const int32_t*>(catalog.columns[RpCudaBranchCatalog::TO].pointer),
            static_cast<const double*>(catalog.columns[RpCudaBranchCatalog::Y_FF_RE].pointer),
            static_cast<const double*>(catalog.columns[RpCudaBranchCatalog::Y_FF_IM].pointer),
            static_cast<const double*>(catalog.columns[RpCudaBranchCatalog::Y_TF_RE].pointer),
            static_cast<const double*>(catalog.columns[RpCudaBranchCatalog::Y_TF_IM].pointer),
            static_cast<const double*>(catalog.columns[RpCudaBranchCatalog::Y_FT_RE].pointer),
            static_cast<const double*>(catalog.columns[RpCudaBranchCatalog::Y_FT_IM].pointer),
            static_cast<const double*>(catalog.columns[RpCudaBranchCatalog::Y_TT_RE].pointer),
            static_cast<const double*>(catalog.columns[RpCudaBranchCatalog::Y_TT_IM].pointer),
            static_cast<const double*>(catalog.columns[RpCudaBranchCatalog::BASE_FROM].pointer),
            static_cast<const double*>(catalog.columns[RpCudaBranchCatalog::BASE_TO].pointer),
            static_cast<const double*>(catalog.columns[RpCudaBranchCatalog::INV_RATING].pointer),
            static_cast<const double*>(catalog.columns[RpCudaBranchCatalog::TRAFO_SCALE].pointer),
            static_cast<const uint8_t*>(catalog.columns[RpCudaBranchCatalog::IN_SERVICE].pointer),
            static_cast<const uint8_t*>(catalog.columns[RpCudaBranchCatalog::KIND].pointer),
            static_cast<const int32_t*>(outages.pointer), static_cast<double*>(flows.pointer),
            b, branches, static_cast<unsigned long long>(nbus), catalog.sbase);
        cuda_check(cudaGetLastError(), "launch two-port branch postprocessing");
        cuda_check(cudaEventRecord(events[1], stream), "record branch kernel end");
        cuda_check(cudaEventRecord(events[2], stream), "record branch download start");
        cuda_check(cudaMemcpyAsync(output, flows.pointer, static_cast<std::size_t>(batch) * branches * 14 * sizeof(double),
                                   cudaMemcpyDeviceToHost, stream), "download branch results");
        cuda_check(cudaEventRecord(events[3], stream), "record branch download end");
        cuda_check(cudaStreamSynchronize(stream), "synchronize branch postprocessing");
        if (stats) {
            float kernel_ms = 0.0F, download_ms = 0.0F;
            cuda_check(cudaEventElapsedTime(&kernel_ms, events[0], events[1]), "measure branch kernel");
            cuda_check(cudaEventElapsedTime(&download_ms, events[2], events[3]), "measure branch download");
            stats->kernel_milliseconds = kernel_ms;
            stats->download_milliseconds = download_ms;
        }
        for (auto event : events) cuda_check(cudaEventDestroy(event), "destroy branch timing event");
    }
};

extern "C" int32_t rp_cuda_probe() {
    try {
        cuda_check(cudaSetDevice(0), "cudaSetDevice");
        cuda_check(cudaFree(nullptr), "initialize CUDA context");
        creation_error.clear();
        return RP_CUDA_SUCCESS;
    } catch (const Failure& error) {
        creation_error = error.what();
        return error.code;
    } catch (const std::exception& error) {
        creation_error = error.what();
        return RP_CUDA_INTERNAL_ERROR;
    } catch (...) {
        creation_error = "unknown C++ exception";
        return RP_CUDA_INTERNAL_ERROR;
    }
}

extern "C" int32_t rp_cuda_create(
    const RpCudaBaselineV1* baseline, int32_t batch, RpCudaExecutor** out_executor) {
    if (out_executor) *out_executor = nullptr;
    try {
        require(baseline && out_executor, "null create argument");
        require(baseline->struct_size == sizeof(RpCudaBaselineV1) &&
                baseline->abi_version == RP_CUDA_ABI_VERSION,
                "incompatible baseline ABI");
        const void* pointers[] = {
            baseline->j_row_ptr, baseline->j_col_indices, baseline->y_col_ptr,
            baseline->y_row_indices, baseline->mirror, baseline->pq_ends,
            baseline->active_ends, baseline->diag_ptrs, baseline->y_re, baseline->y_im,
            baseline->sbus_re, baseline->sbus_im, baseline->initial_v_re, baseline->initial_v_im,
        };
        for (const void* pointer : pointers) require(pointer != nullptr, "null baseline array");
        require(baseline->nbus > 0 && baseline->npv >= 0 && baseline->npq > 0 && batch > 0 &&
                baseline->j_nnz > 0 && baseline->y_nnz > 0,
                "invalid baseline or batch dimensions");
        const std::int64_t active = static_cast<std::int64_t>(baseline->npv) + baseline->npq;
        require(active > 0 && active < baseline->nbus &&
                active + baseline->npq <= std::numeric_limits<int>::max(),
                "invalid bus partition");
        cuda_check(cudaSetDevice(0), "cudaSetDevice");
        cuda_check(cudaFree(nullptr), "initialize CUDA context");
        auto executor = std::make_unique<RpCudaExecutor>(*baseline, batch);
        executor->initialize_solver();
        *out_executor = executor.release();
        creation_error.clear();
        return RP_CUDA_SUCCESS;
    } catch (const Failure& error) {
        creation_error = error.what();
        return error.code;
    } catch (const std::exception& error) {
        creation_error = error.what();
        return RP_CUDA_INTERNAL_ERROR;
    } catch (...) {
        creation_error = "unknown C++ exception";
        return RP_CUDA_INTERNAL_ERROR;
    }
}

extern "C" int32_t rp_cuda_solve_base_batch(
    RpCudaExecutor* executor, int32_t batch, const RpCudaSolveOptionsV1* options,
    int32_t* case_status, int32_t* iterations, double* residuals,
    double* voltage_re, double* voltage_im) {
    if (!executor) return RP_CUDA_INVALID_ARGUMENT;
    try {
        require(options && case_status && iterations && residuals && voltage_re && voltage_im,
                "null solve argument");
        require(options->struct_size == sizeof(RpCudaSolveOptionsV1), "incompatible solve options ABI");
        require(batch == executor->batch, "solve batch must equal executor batch capacity");
        require(options->max_iterations >= 0 && std::isfinite(options->tolerance) && options->tolerance > 0,
                "invalid Newton options");
        executor->solve(*options, nullptr, case_status, iterations, residuals, voltage_re, voltage_im);
        executor->last_error.clear();
        return RP_CUDA_SUCCESS;
    } catch (const Failure& error) {
        executor->last_error = error.what();
        return error.code;
    } catch (const std::exception& error) {
        executor->last_error = error.what();
        return RP_CUDA_INTERNAL_ERROR;
    } catch (...) {
        executor->last_error = "unknown C++ exception";
        return RP_CUDA_INTERNAL_ERROR;
    }
}

extern "C" int32_t rp_cuda_solve_outage_batch(
    RpCudaExecutor* executor, int32_t batch, const RpCudaOutageBatchV1* outages,
    const RpCudaSolveOptionsV1* options, int32_t* case_status, int32_t* iterations,
    double* residuals, double* voltage_re, double* voltage_im) {
    if (!executor) return RP_CUDA_INVALID_ARGUMENT;
    try {
        require(outages && options && case_status && iterations && residuals && voltage_re && voltage_im,
                "null outage solve argument");
        require(outages->struct_size == sizeof(RpCudaOutageBatchV1) &&
                options->struct_size == sizeof(RpCudaSolveOptionsV1), "incompatible outage solve ABI");
        require(outages->patch_counts && outages->slots && outages->y_re && outages->y_im,
                "null outage patch array");
        require(batch == executor->batch, "solve batch must equal executor batch capacity");
        require(options->max_iterations >= 0 && std::isfinite(options->tolerance) && options->tolerance > 0,
                "invalid Newton options");
        executor->solve(*options, outages, case_status, iterations, residuals, voltage_re, voltage_im);
        executor->last_error.clear();
        return RP_CUDA_SUCCESS;
    } catch (const Failure& error) {
        executor->last_error = error.what();
        return error.code;
    } catch (const std::exception& error) {
        executor->last_error = error.what();
        return RP_CUDA_INTERNAL_ERROR;
    } catch (...) {
        executor->last_error = "unknown C++ exception";
        return RP_CUDA_INTERNAL_ERROR;
    }
}

extern "C" int32_t rp_cuda_branch_catalog_create(
    const RpCudaBranchCatalogV1* catalog, RpCudaBranchCatalog** out_catalog) {
    if (out_catalog) *out_catalog = nullptr;
    try {
        require(catalog && out_catalog, "null branch catalog create argument");
        require(catalog->struct_size == sizeof(RpCudaBranchCatalogV1),
                "incompatible branch catalog ABI");
        cuda_check(cudaSetDevice(0), "cudaSetDevice for branch catalog");
        cuda_check(cudaFree(nullptr), "initialize branch catalog CUDA context");
        auto value = std::make_unique<RpCudaBranchCatalog>(*catalog);
        *out_catalog = value.release();
        creation_error.clear();
        return RP_CUDA_SUCCESS;
    } catch (const Failure& error) {
        creation_error = error.what();
        return error.code;
    } catch (const std::exception& error) {
        creation_error = error.what();
        return RP_CUDA_INTERNAL_ERROR;
    } catch (...) {
        creation_error = "unknown C++ exception";
        return RP_CUDA_INTERNAL_ERROR;
    }
}

extern "C" void rp_cuda_branch_catalog_destroy(RpCudaBranchCatalog* catalog) {
    delete catalog;
}

extern "C" int32_t rp_cuda_postprocess_branches(
    RpCudaExecutor* executor, const RpCudaBranchCatalog* catalog,
    const int32_t* outage_indices, double* output, RpCudaPostprocessStatsV1* stats) {
    if (!executor) return RP_CUDA_INVALID_ARGUMENT;
    try {
        require(catalog && outage_indices && output, "null branch postprocessing argument");
        require(!stats || stats->struct_size == sizeof(RpCudaPostprocessStatsV1),
                "incompatible branch postprocessing stats ABI");
        executor->postprocess_branches(*catalog, outage_indices, output, stats);
        executor->last_error.clear();
        return RP_CUDA_SUCCESS;
    } catch (const Failure& error) {
        executor->last_error = error.what();
        return error.code;
    } catch (const std::exception& error) {
        executor->last_error = error.what();
        return RP_CUDA_INTERNAL_ERROR;
    } catch (...) {
        executor->last_error = "unknown C++ exception";
        return RP_CUDA_INTERNAL_ERROR;
    }
}

extern "C" const char* rp_cuda_last_error(const RpCudaExecutor* executor) {
    return executor ? executor->last_error.c_str() : creation_error.c_str();
}

extern "C" void rp_cuda_destroy(RpCudaExecutor* executor) {
    delete executor;
}
