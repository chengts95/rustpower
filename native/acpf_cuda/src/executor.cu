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

// Production starts every scenario from precisely the same Rust-supplied state.
__global__ void executor_prepare_voltage_kernel(
    const double* base_re, const double* base_im,
    double* voltage_re, double* voltage_im, double* inverse_magnitude,
    unsigned long long batch, unsigned long long nbus) {
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
}

} // namespace

struct RpCudaExecutor {
    int nbus{}, npv{}, npq{}, n_active{}, n{}, batch{};
    std::size_t nnz{}, ynnz{}, active_ynnz{};
    std::string last_error;

    DeviceBuffer j_row_ptr, j_col_indices;
    DeviceBuffer y_col_ptr, y_row_indices, mirror, y_re, y_im;
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
          n(base.npv + 2 * base.npq), batch(batch_), nnz(base.j_nnz), ynnz(base.y_nnz),
          j_row_ptr((static_cast<std::size_t>(n) + 1) * sizeof(int)),
          j_col_indices(nnz * sizeof(int)),
          y_col_ptr((static_cast<std::size_t>(nbus) + 1) * sizeof(int)),
          y_row_indices(ynnz * sizeof(int)), mirror(ynnz * sizeof(int)),
          y_re(ynnz * sizeof(double)), y_im(ynnz * sizeof(double)),
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
        const auto b = static_cast<std::size_t>(batch);
        require(b <= std::numeric_limits<std::size_t>::max() / nnz &&
                b <= std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(n) &&
                b <= std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(nbus),
                "batch allocation size overflow");

        column_of_nnz = DeviceBuffer(active_ynnz * sizeof(int));
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
        y_re.upload(base.y_re, y_re.bytes);
        y_im.upload(base.y_im, y_im.bytes);
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
        cudss_check(cudssConfigSet(config, CUDSS_CONFIG_UBATCH_SIZE, &batch, sizeof(batch)),
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

    void solve(const RpCudaSolveOptionsV1& options, int* case_status,
               int* iterations, double* residuals, double* voltage_re, double* voltage_im) {
        const auto b = static_cast<std::size_t>(batch);
        std::vector<int> host_mask(b, 1), host_iterations(b, 0), host_status(b, RP_CUDA_CASE_CONVERGED);
        std::vector<double> host_norms(b, std::numeric_limits<double>::infinity());
        mask.upload(host_mask.data(), mask.bytes);

        executor_prepare_voltage_kernel<<<65535, 256, 0, stream>>>(
            static_cast<const double*>(base_v_re.pointer), static_cast<const double*>(base_v_im.pointer),
            static_cast<double*>(v_re.pointer), static_cast<double*>(v_im.pointer),
            static_cast<double*>(inv_vmag.pointer), b, nbus);
        cuda_check(cudaGetLastError(), "launch exact voltage preparation");

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
            newton_convergence_kernel<<<batch, 256, 0, stream>>>(
                static_cast<const double*>(mismatch.pointer), static_cast<int*>(mask.pointer),
                static_cast<double*>(norms.pointer), b, n, options.tolerance);
            mask.download(host_mask.data(), mask.bytes);
            norms.download(host_norms.data(), norms.bytes);

            int active_count = 0;
            for (int scenario = 0; scenario < batch; ++scenario) {
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
                for (int scenario = 0; scenario < batch; ++scenario) {
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
        std::copy(host_status.begin(), host_status.end(), case_status);
        std::copy(host_iterations.begin(), host_iterations.end(), iterations);
        std::copy(host_norms.begin(), host_norms.end(), residuals);
        v_re.download(voltage_re, v_re.bytes);
        v_im.download(voltage_im, v_im.bytes);
    }
};

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
        executor->solve(*options, case_status, iterations, residuals, voltage_re, voltage_im);
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
