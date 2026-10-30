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
typedef struct RpCudaBranchCatalog RpCudaBranchCatalog;

/* Immutable two-port constants in the shared solver's bus ordering. */
typedef struct RpCudaBranchCatalogV1 {
    uint32_t struct_size;
    int32_t nbus;
    int32_t branch_count;
    const int32_t* from;
    const int32_t* to;
    const double* y_ff_re;
    const double* y_ff_im;
    const double* y_tf_re;
    const double* y_tf_im;
    const double* y_ft_re;
    const double* y_ft_im;
    const double* y_tt_re;
    const double* y_tt_im;
    const double* base_i_from_ka;
    const double* base_i_to_ka;
    const double* inv_line_rating;
    const double* transformer_loading_scale;
    const uint8_t* in_service;
    const uint8_t* kind; /* 0 = line, 1 = transformer */
    double sbase;
} RpCudaBranchCatalogV1;

typedef struct RpCudaPostprocessStatsV1 {
    uint32_t struct_size;
    double kernel_milliseconds;
    double download_milliseconds;
} RpCudaPostprocessStatsV1;

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

/* Four absolute Ybus value slots per scenario, compiled by OutageTask. */
typedef struct RpCudaOutageBatchV1 {
    uint32_t struct_size;
    const int32_t* patch_counts;
    const int32_t* slots;
    const double* y_re;
    const double* y_im;
} RpCudaOutageBatchV1;

/* Initialize device zero without allocating an executor. Safe to call during
 * backend selection; diagnostics are returned by rp_cuda_last_error(NULL). */
RP_CUDA_API int32_t rp_cuda_probe(void);

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

RP_CUDA_API int32_t rp_cuda_solve_outage_batch(
    RpCudaExecutor* executor,
    int32_t batch,
    const RpCudaOutageBatchV1* outages,
    const RpCudaSolveOptionsV1* options,
    int32_t* case_status,
    int32_t* iterations,
    double* residuals,
    double* voltage_re,
    double* voltage_im);

RP_CUDA_API int32_t rp_cuda_branch_catalog_create(
    const RpCudaBranchCatalogV1* catalog,
    RpCudaBranchCatalog** out_catalog);
RP_CUDA_API void rp_cuda_branch_catalog_destroy(RpCudaBranchCatalog* catalog);

/* Uses the executor's last device-side voltage. One outage index per scenario;
 * a negative index means no outage. Output is scenario-major, 14 doubles/branch. */
RP_CUDA_API int32_t rp_cuda_postprocess_branches(
    RpCudaExecutor* executor,
    const RpCudaBranchCatalog* catalog,
    const int32_t* outage_indices,
    double* output,
    RpCudaPostprocessStatsV1* stats);

/* Passing null returns the last creation error on this thread. */
RP_CUDA_API const char* rp_cuda_last_error(const RpCudaExecutor* executor);
RP_CUDA_API void rp_cuda_destroy(RpCudaExecutor* executor);

#ifdef __cplusplus
}
#endif
#endif
