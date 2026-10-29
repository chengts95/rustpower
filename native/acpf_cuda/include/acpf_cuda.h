#ifndef RUSTPOWER_ACPF_CUDA_H
#define RUSTPOWER_ACPF_CUDA_H

#include <stdint.h>

#ifdef _WIN32
#define RP_CUDA_API __declspec(dllexport)
#else
#define RP_CUDA_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define RP_CUDA_ABI_VERSION 1u
#define RP_CUDA_SUCCESS 0
#define RP_CUDA_INVALID_ARGUMENT 1
#define RP_CUDA_CUDA_ERROR 2
#define RP_CUDA_CUDSS_ERROR 3
#define RP_CUDA_INTERNAL_ERROR 4

#define RP_CUDA_CASE_CONVERGED 0
#define RP_CUDA_CASE_MAX_ITERATIONS 1
#define RP_CUDA_CASE_NON_FINITE 2

typedef struct RpCudaExecutor RpCudaExecutor;

typedef struct RpCudaBaselineV1 {
    uint32_t struct_size;
    uint32_t abi_version;
    int32_t nbus;
    int32_t npv;
    int32_t npq;
    int32_t j_nnz;
    int32_t y_nnz;
    const int32_t* j_row_ptr;
    const int32_t* j_col_indices;
    const int32_t* y_col_ptr;
    const int32_t* y_row_indices;
    const int32_t* mirror;
    const int32_t* pq_ends;
    const int32_t* active_ends;
    const int32_t* diag_ptrs;
    const double* y_re;
    const double* y_im;
    const double* sbus_re;
    const double* sbus_im;
    const double* initial_v_re;
    const double* initial_v_im;
} RpCudaBaselineV1;

typedef struct RpCudaSolveOptionsV1 {
    uint32_t struct_size;
    int32_t max_iterations;
    double tolerance;
} RpCudaSolveOptionsV1;

RP_CUDA_API int32_t rp_cuda_create(
    const RpCudaBaselineV1* baseline,
    int32_t batch,
    RpCudaExecutor** out_executor);

/* This first ABI deliberately uses one fixed uniform batch per executor. */
RP_CUDA_API int32_t rp_cuda_solve_base_batch(
    RpCudaExecutor* executor,
    int32_t batch,
    const RpCudaSolveOptionsV1* options,
    int32_t* case_status,
    int32_t* iterations,
    double* residuals,
    double* voltage_re,
    double* voltage_im);

/* Passing null returns the last creation error on this thread. */
RP_CUDA_API const char* rp_cuda_last_error(const RpCudaExecutor* executor);
RP_CUDA_API void rp_cuda_destroy(RpCudaExecutor* executor);

#ifdef __cplusplus
}
#endif
#endif
