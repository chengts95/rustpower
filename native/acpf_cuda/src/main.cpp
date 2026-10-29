// RustPower experimental batched ACPF foundation.
// C++ owns the executor; NVCC builds the native CUDA kernels offline.
#define NOMINMAX
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cudss.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <windows.h>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename T>
T bind_symbol(HMODULE library, const char* name) {
    const auto symbol = GetProcAddress(library, name);
    if (!symbol) throw std::runtime_error(std::string("missing DLL symbol: ") + name);
    return reinterpret_cast<T>(symbol);
}

struct Api {
    HMODULE cudart{};
    HMODULE cuda{};
    HMODULE cudss{};
    std::vector<DLL_DIRECTORY_COOKIE> directories;

    decltype(&::cudaSetDevice) cudaSetDevice_fn{};
    decltype(&::cudaGetDeviceProperties) cudaGetDeviceProperties_fn{};
    decltype(&::cudaMemGetInfo) cudaMemGetInfo_fn{};
    cudaError_t (*cudaMalloc_fn)(void**, std::size_t){};
    decltype(&::cudaFree) cudaFree_fn{};
    decltype(&::cudaMemcpy) cudaMemcpy_fn{};
    decltype(&::cudaMemset) cudaMemset_fn{};
    decltype(&::cudaStreamCreate) cudaStreamCreate_fn{};
    decltype(&::cudaStreamDestroy) cudaStreamDestroy_fn{};
    decltype(&::cudaStreamSynchronize) cudaStreamSynchronize_fn{};
    decltype(&::cudaGetErrorString) cudaGetErrorString_fn{};

    decltype(&::cuInit) cuInit_fn{};
    decltype(&::cuModuleLoadData) cuModuleLoadData_fn{};
    decltype(&::cuModuleUnload) cuModuleUnload_fn{};
    decltype(&::cuModuleGetFunction) cuModuleGetFunction_fn{};
    decltype(&::cuLaunchKernel) cuLaunchKernel_fn{};
    decltype(&::cuGetErrorName) cuGetErrorName_fn{};

    decltype(&::cudssCreate) cudssCreate_fn{};
    decltype(&::cudssDestroy) cudssDestroy_fn{};
    decltype(&::cudssConfigCreate) cudssConfigCreate_fn{};
    decltype(&::cudssConfigDestroy) cudssConfigDestroy_fn{};
    decltype(&::cudssDataCreate) cudssDataCreate_fn{};
    decltype(&::cudssDataDestroy) cudssDataDestroy_fn{};
    decltype(&::cudssSetStream) cudssSetStream_fn{};
    decltype(&::cudssSetJitCacheDir) cudssSetJitCacheDir_fn{};
    decltype(&::cudssConfigSet) cudssConfigSet_fn{};
    decltype(&::cudssDataSet) cudssDataSet_fn{};
    decltype(&::cudssDataGet) cudssDataGet_fn{};
    decltype(&::cudssMatrixCreateCsr) cudssMatrixCreateCsr_fn{};
    decltype(&::cudssMatrixCreateDn) cudssMatrixCreateDn_fn{};
    decltype(&::cudssMatrixDestroy) cudssMatrixDestroy_fn{};
    decltype(&::cudssExecute) cudssExecute_fn{};
    decltype(&::cudssGetProperty) cudssGetProperty_fn{};

    explicit Api(const fs::path& root) {
        SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        const fs::path packages = root / "gpu-benchmarks/rtx-2070-super/python-packages/nvidia";
        // cuDSS may resolve its own JIT dependencies from this directory; we do not call NVRTC.
        for (const char* relative : {"cu12/bin", "cuda_runtime/bin", "cublas/bin", "cuda_nvrtc/bin"}) {
            const auto path = (packages / relative).wstring();
            const auto cookie = AddDllDirectory(path.c_str());
            if (!cookie) {
                throw std::runtime_error("AddDllDirectory failed for " + (packages / relative).string());
            }
            directories.push_back(cookie);
        }
        auto load = [&](const fs::path& path) {
            HMODULE module = LoadLibraryExW(
                path.wstring().c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
            if (!module) throw std::runtime_error("LoadLibrary failed for " + path.string());
            return module;
        };
        cudart = load(packages / "cuda_runtime/bin/cudart64_12.dll");
        cuda = LoadLibraryW(L"nvcuda.dll");
        if (!cuda) throw std::runtime_error("LoadLibrary failed for nvcuda.dll");
        cudss = load(packages / "cu12/bin/cudss64_1.dll");

#define RP_BIND(library, field, symbol) field = bind_symbol<decltype(field)>(library, symbol)
        RP_BIND(cudart, cudaSetDevice_fn, "cudaSetDevice");
        RP_BIND(cudart, cudaGetDeviceProperties_fn, "cudaGetDeviceProperties");
        RP_BIND(cudart, cudaMemGetInfo_fn, "cudaMemGetInfo");
        RP_BIND(cudart, cudaMalloc_fn, "cudaMalloc");
        RP_BIND(cudart, cudaFree_fn, "cudaFree");
        RP_BIND(cudart, cudaMemcpy_fn, "cudaMemcpy");
        RP_BIND(cudart, cudaMemset_fn, "cudaMemset");
        RP_BIND(cudart, cudaStreamCreate_fn, "cudaStreamCreate");
        RP_BIND(cudart, cudaStreamDestroy_fn, "cudaStreamDestroy");
        RP_BIND(cudart, cudaStreamSynchronize_fn, "cudaStreamSynchronize");
        RP_BIND(cudart, cudaGetErrorString_fn, "cudaGetErrorString");
        RP_BIND(cuda, cuInit_fn, "cuInit");
        RP_BIND(cuda, cuModuleLoadData_fn, "cuModuleLoadData");
        RP_BIND(cuda, cuModuleUnload_fn, "cuModuleUnload");
        RP_BIND(cuda, cuModuleGetFunction_fn, "cuModuleGetFunction");
        RP_BIND(cuda, cuLaunchKernel_fn, "cuLaunchKernel");
        RP_BIND(cuda, cuGetErrorName_fn, "cuGetErrorName");
        RP_BIND(cudss, cudssCreate_fn, "cudssCreate");
        RP_BIND(cudss, cudssDestroy_fn, "cudssDestroy");
        RP_BIND(cudss, cudssConfigCreate_fn, "cudssConfigCreate");
        RP_BIND(cudss, cudssConfigDestroy_fn, "cudssConfigDestroy");
        RP_BIND(cudss, cudssDataCreate_fn, "cudssDataCreate");
        RP_BIND(cudss, cudssDataDestroy_fn, "cudssDataDestroy");
        RP_BIND(cudss, cudssSetStream_fn, "cudssSetStream");
        RP_BIND(cudss, cudssSetJitCacheDir_fn, "cudssSetJitCacheDir");
        RP_BIND(cudss, cudssConfigSet_fn, "cudssConfigSet");
        RP_BIND(cudss, cudssDataSet_fn, "cudssDataSet");
        RP_BIND(cudss, cudssDataGet_fn, "cudssDataGet");
        RP_BIND(cudss, cudssMatrixCreateCsr_fn, "cudssMatrixCreateCsr");
        RP_BIND(cudss, cudssMatrixCreateDn_fn, "cudssMatrixCreateDn");
        RP_BIND(cudss, cudssMatrixDestroy_fn, "cudssMatrixDestroy");
        RP_BIND(cudss, cudssExecute_fn, "cudssExecute");
        RP_BIND(cudss, cudssGetProperty_fn, "cudssGetProperty");
#undef RP_BIND
    }
};

void cuda_check(Api& api, cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + api.cudaGetErrorString_fn(status));
    }
}

void cu_check(Api& api, CUresult status, const char* operation) {
    if (status != CUDA_SUCCESS) {
        const char* name = nullptr;
        api.cuGetErrorName_fn(status, &name);
        throw std::runtime_error(std::string(operation) + ": " + (name ? name : std::to_string(status)));
    }
}

void cudss_check(cudssStatus_t status, const char* operation) {
    if (status != CUDSS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation) + ": cuDSS status " + std::to_string(status));
    }
}

struct DeviceBuffer {
    Api* api{};
    void* pointer{};
    std::size_t bytes{};

    DeviceBuffer() = default;
    DeviceBuffer(Api& api_, std::size_t bytes_) : api(&api_), bytes(bytes_) {
        require(bytes > 0, "refusing a zero-sized device allocation");
        cuda_check(*api, api->cudaMalloc_fn(&pointer, bytes), "cudaMalloc");
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept { swap(other); }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            reset();
            swap(other);
        }
        return *this;
    }
    ~DeviceBuffer() { reset(); }
    void swap(DeviceBuffer& other) noexcept {
        std::swap(api, other.api);
        std::swap(pointer, other.pointer);
        std::swap(bytes, other.bytes);
    }
    void reset() {
        if (pointer && api) api->cudaFree_fn(pointer);
        pointer = nullptr;
        bytes = 0;
    }
    void upload(const void* source, std::size_t count) {
        require(count <= bytes, "device upload exceeds allocation");
        cuda_check(*api, api->cudaMemcpy_fn(pointer, source, count, cudaMemcpyHostToDevice), "cudaMemcpy H2D");
    }
    void download(void* destination, std::size_t count) const {
        require(count <= bytes, "device download exceeds allocation");
        cuda_check(*api, api->cudaMemcpy_fn(destination, pointer, count, cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
    }
};

struct Stream {
    Api* api{};
    cudaStream_t value{};
    explicit Stream(Api& api_) : api(&api_) {
        cuda_check(*api, api->cudaStreamCreate_fn(&value), "cudaStreamCreate");
    }
    ~Stream() {
        if (value) api->cudaStreamDestroy_fn(value);
    }
    void synchronize() { cuda_check(*api, api->cudaStreamSynchronize_fn(value), "cudaStreamSynchronize"); }
};

struct CudaKernels {
    Api* api{};
    CUmodule module{};
    CUfunction prepare{};
    CUfunction power_mismatch{};
    CUfunction jacobian{};
    CUfunction matvec{};
    CUfunction matvec_fp32{};
    CUfunction cast_fp32{};

    CudaKernels(Api& api_, const fs::path& ptx_path) : api(&api_) {
        std::ifstream input(ptx_path, std::ios::binary);
        require(input.good(), "cannot open offline PTX: " + ptx_path.string());
        const std::string ptx((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
        cu_check(*api, api->cuModuleLoadData_fn(&module, ptx.data()), "cuModuleLoadData");
        cu_check(*api, api->cuModuleGetFunction_fn(&prepare, module, "prepare_voltage_kernel"), "get prepare kernel");
        cu_check(*api, api->cuModuleGetFunction_fn(&power_mismatch, module, "batch_power_mismatch_kernel"),
                 "get power mismatch kernel");
        cu_check(*api, api->cuModuleGetFunction_fn(&jacobian, module, "batch_jacobian_csr_kernel"),
                 "get Jacobian kernel");
        cu_check(*api, api->cuModuleGetFunction_fn(&matvec, module, "batch_csr_matvec_kernel"), "get matvec kernel");
        cu_check(*api, api->cuModuleGetFunction_fn(&matvec_fp32, module, "batch_csr_matvec_fp32_kernel"),
                 "get FP32 matvec kernel");
        cu_check(*api, api->cuModuleGetFunction_fn(&cast_fp32, module, "batch_cast_fp64_to_fp32_kernel"),
                 "get FP32 cast kernel");
    }
    ~CudaKernels() {
        if (module) api->cuModuleUnload_fn(module);
    }
};

struct GroupedYbusView {
    const int* col_ptr{};
    const int* row_indices{};
    const int* mirror{};
    const double* re{};
    const double* im{};
};

struct GroupedNewtonStateView {
    const double* v_re{};
    const double* v_im{};
    const double* inv_vmag{};
    const double* scalc_re{};
    const double* scalc_im{};
};

struct GroupedJacobianView {
    const int* pq_ends{};
    const int* active_ends{};
    const int* diag_ptrs{};
    const int* row_ptr{};
    double* values{};
};

// 爱莉希雅的小约定：这里的布局必须与 newton_kernels.cu 完全一致。
struct GroupedJacobianArgs {
    GroupedYbusView ybus{};
    GroupedNewtonStateView state{};
    GroupedJacobianView jacobian{};
    const int* mask{};
    int n_active{};
    int npq{};
    int nbus{};
    int batch{};
    unsigned long long nnz{};
};

struct NewtonKernels {
    Api* api{};
    CUmodule module{};
    CUfunction prepare{}, power_mismatch{}, convergence{}, jacobian{}, jacobian_mapped{}, update{}, update_fp32{};
    CUfunction jacobian_l2{}, jacobian_l4{}, jacobian_l8{};

    NewtonKernels(Api& api_, const fs::path& ptx_path) : api(&api_) {
        std::ifstream input(ptx_path, std::ios::binary);
        require(input.good(), "cannot open offline PTX: " + ptx_path.string());
        const std::string ptx((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
        cu_check(*api, api->cuModuleLoadData_fn(&module, ptx.data()), "cuModuleLoadData");
        cu_check(*api, api->cuModuleGetFunction_fn(&prepare, module, "prepare_newton_voltage_kernel"), "get Newton prepare");
        cu_check(*api, api->cuModuleGetFunction_fn(&power_mismatch, module, "newton_power_mismatch_kernel"), "get Newton power");
        cu_check(*api, api->cuModuleGetFunction_fn(&convergence, module, "newton_convergence_kernel"), "get convergence");
        cu_check(*api, api->cuModuleGetFunction_fn(&jacobian, module, "newton_jacobian_csr_kernel"), "get Newton Jacobian");
        cu_check(*api, api->cuModuleGetFunction_fn(&jacobian_mapped, module, "newton_jacobian_csr_mapped_kernel"),
                 "get mapped Newton Jacobian");
        cu_check(*api, api->cuModuleGetFunction_fn(&jacobian_l2, module, "newton_jacobian_csr_l2"), "get L2 Jacobian");
        cu_check(*api, api->cuModuleGetFunction_fn(&jacobian_l4, module, "newton_jacobian_csr_l4"), "get L4 Jacobian");
        cu_check(*api, api->cuModuleGetFunction_fn(&jacobian_l8, module, "newton_jacobian_csr_l8"), "get L8 Jacobian");
        cu_check(*api, api->cuModuleGetFunction_fn(&update, module, "newton_update_voltage_kernel"), "get voltage update");
        cu_check(*api, api->cuModuleGetFunction_fn(&update_fp32, module, "newton_update_voltage_fp32_kernel"),
                 "get mixed-precision voltage update");
    }
    ~NewtonKernels() {
        if (module) api->cuModuleUnload_fn(module);
    }
};

unsigned grid_for(std::size_t threads, unsigned block) {
    const auto blocks = (threads + block - 1) / block;
    require(blocks > 0 && blocks <= std::numeric_limits<unsigned>::max(), "CUDA grid is too large");
    return static_cast<unsigned>(blocks);
}

template <typename F>
double timed(Stream& stream, F&& action) {
    stream.synchronize();
    const auto start = Clock::now();
    action();
    stream.synchronize();
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

double median(std::vector<double> values) {
    require(!values.empty(), "median of an empty sample");
    std::sort(values.begin(), values.end());
    const auto middle = values.size() / 2;
    return values.size() % 2 ? values[middle] : 0.5 * (values[middle - 1] + values[middle]);
}

struct Fixture {
    std::uint32_t n{}, nnz{}, nbus{}, ynnz{}, npv{}, npq{}, active_ynnz{};
    std::vector<int> row_ptr, col_indices, y_col_ptr, y_row_indices, mirror;
    std::vector<int> pq_ends, active_ends, diag_ptrs;
    std::vector<double> values, rhs, y_re, y_im, v_re, v_im;
    std::vector<double> sbus_re, sbus_im, scalc_re, scalc_im;
    std::vector<double> converged_v_re, converged_v_im;
    int converged_iterations{};
    double converged_norm{};
};

template <typename T>
void read_vector(std::ifstream& input, std::vector<T>& values, std::size_t count) {
    values.resize(count);
    input.read(reinterpret_cast<char*>(values.data()), count * sizeof(T));
}

Fixture read_fixture(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    require(input.good(), "cannot open fixture: " + path.string());
    char magic[8]{};
    Fixture f;
    input.read(magic, 8);
    input.read(reinterpret_cast<char*>(&f.n), sizeof(f.n));
    input.read(reinterpret_cast<char*>(&f.nnz), sizeof(f.nnz));
    input.read(reinterpret_cast<char*>(&f.nbus), sizeof(f.nbus));
    input.read(reinterpret_cast<char*>(&f.ynnz), sizeof(f.ynnz));
    input.read(reinterpret_cast<char*>(&f.npv), sizeof(f.npv));
    input.read(reinterpret_cast<char*>(&f.npq), sizeof(f.npq));
    input.read(reinterpret_cast<char*>(&f.active_ynnz), sizeof(f.active_ynnz));
    require(input.good() && std::string(magic, 8) == std::string("RPACPF2\0", 8), "invalid fixture header");
    const std::size_t active = f.npv + f.npq;
    read_vector(input, f.row_ptr, static_cast<std::size_t>(f.n) + 1);
    read_vector(input, f.col_indices, f.nnz);
    read_vector(input, f.values, f.nnz);
    read_vector(input, f.rhs, f.n);
    read_vector(input, f.y_col_ptr, static_cast<std::size_t>(f.nbus) + 1);
    read_vector(input, f.y_row_indices, f.ynnz);
    read_vector(input, f.mirror, f.ynnz);
    read_vector(input, f.y_re, f.ynnz);
    read_vector(input, f.y_im, f.ynnz);
    read_vector(input, f.v_re, f.nbus);
    read_vector(input, f.v_im, f.nbus);
    read_vector(input, f.sbus_re, f.nbus);
    read_vector(input, f.sbus_im, f.nbus);
    read_vector(input, f.scalc_re, f.nbus);
    read_vector(input, f.scalc_im, f.nbus);
    read_vector(input, f.pq_ends, active);
    read_vector(input, f.active_ends, active);
    read_vector(input, f.diag_ptrs, active);
    read_vector(input, f.converged_v_re, f.nbus);
    read_vector(input, f.converged_v_im, f.nbus);
    input.read(reinterpret_cast<char*>(&f.converged_iterations), sizeof(f.converged_iterations));
    input.read(reinterpret_cast<char*>(&f.converged_norm), sizeof(f.converged_norm));
    require(input.good(), "truncated fixture");
    require(f.row_ptr.front() == 0 && f.row_ptr.back() == static_cast<int>(f.nnz), "invalid CSR offsets");
    require(f.y_col_ptr.front() == 0 && f.y_col_ptr.back() == static_cast<int>(f.ynnz), "invalid Ybus offsets");
    require(f.n == active + f.npq && f.y_col_ptr[active] == static_cast<int>(f.active_ynnz),
            "inconsistent AC dimensions");
    return f;
}

std::size_t free_memory(Api& api) {
    std::size_t free_bytes{};
    std::size_t total_bytes{};
    cuda_check(api, api.cudaMemGetInfo_fn(&free_bytes, &total_bytes), "cudaMemGetInfo");
    return free_bytes;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const fs::path root = std::getenv("RUSTPOWER_ROOT")
                                  ? fs::path(std::getenv("RUSTPOWER_ROOT"))
                                  : fs::current_path();
        const fs::path fixture_path = argc > 1 ? fs::path(argv[1]) : root / "target/acpf-cuda/pegase9241-jacobian.bin";
        const std::size_t batch = argc > 2 ? std::stoull(argv[2]) : 2;
        const int repeats = argc > 3 ? std::stoi(argv[3]) : 1;
        const bool mixed_newton_mode = argc > 4 && std::string(argv[4]) == "newton32";
        const bool newton_mode = mixed_newton_mode || (argc > 4 && std::string(argv[4]) == "newton");
        const bool fp32_mode = argc > 4 && std::string(argv[4]) == "fp32";
        const bool fp32_solver = mixed_newton_mode || fp32_mode;
        const bool mapped_jacobian = argc > 5 && std::string(argv[5]) == "map";
        const int jacobian_lanes = argc > 5 && !mapped_jacobian ? std::stoi(argv[5]) : 1;
        require(argc <= 6 && (argc <= 4 || newton_mode || fp32_mode),
                "usage: acpf_batch_demo [fixture] [batch] [repeats] [newton|newton32|fp32] [map|2|4|8]");
        require(batch > 0 && repeats > 0, "batch and repeats must be positive");
        require(!newton_mode || repeats == 1, "full Newton currently runs one repeat per process");
        require((!mapped_jacobian || newton_mode) &&
                (jacobian_lanes == 1 || ((jacobian_lanes == 2 || jacobian_lanes == 4 || jacobian_lanes == 8) && newton_mode)),
                "Jacobian option requires Newton mode and one of map, 2, 4, 8");

        Api api(root);
        cuda_check(api, api.cudaSetDevice_fn(0), "cudaSetDevice");
        cuda_check(api, api.cudaFree_fn(nullptr), "CUDA context initialization");
        cu_check(api, api.cuInit_fn(0), "cuInit");
        cudaDeviceProp properties{};
        cuda_check(api, api.cudaGetDeviceProperties_fn(&properties, 0), "cudaGetDeviceProperties");
        const auto free_before = free_memory(api);
        std::cout << "GPU: " << properties.name << "\nBatch: " << batch << ", precision: "
                  << (mixed_newton_mode ? "FP32 solve + FP64 residual" : (fp32_mode ? "FP32" : "FP64"))
                  << "\n" << std::flush;

        const auto fixture = read_fixture(fixture_path);
        const std::size_t n = fixture.n;
        const std::size_t nnz = fixture.nnz;
        const std::size_t nbus = fixture.nbus;
        const std::size_t ynnz = fixture.ynnz;
        const std::size_t n_active = fixture.npv + fixture.npq;
        const std::size_t npq = fixture.npq;
        const std::size_t active_ynnz = fixture.active_ynnz;
        require(n > 0 && nnz > 0, "empty fixture");
        require(batch <= std::numeric_limits<std::size_t>::max() / nnz, "batch nnz overflow");
        require(batch <= std::numeric_limits<std::size_t>::max() / n, "batch vector overflow");
        require(batch <= std::numeric_limits<std::size_t>::max() / nbus, "batch voltage overflow");

        DeviceBuffer d_row_ptr(api, (n + 1) * sizeof(int));
        DeviceBuffer d_col_indices(api, nnz * sizeof(int));
        DeviceBuffer d_y_col_ptr(api, (nbus + 1) * sizeof(int));
        DeviceBuffer d_y_row_indices(api, ynnz * sizeof(int));
        DeviceBuffer d_mirror(api, ynnz * sizeof(int));
        DeviceBuffer d_y_re(api, ynnz * sizeof(double));
        DeviceBuffer d_y_im(api, ynnz * sizeof(double));
        DeviceBuffer d_base_v_re(api, nbus * sizeof(double));
        DeviceBuffer d_base_v_im(api, nbus * sizeof(double));
        DeviceBuffer d_sbus_re(api, nbus * sizeof(double));
        DeviceBuffer d_sbus_im(api, nbus * sizeof(double));
        DeviceBuffer d_pq_ends(api, n_active * sizeof(int));
        DeviceBuffer d_active_ends(api, n_active * sizeof(int));
        DeviceBuffer d_diag_ptrs(api, n_active * sizeof(int));
        DeviceBuffer d_column_of_nnz(api, active_ynnz * sizeof(int));
        DeviceBuffer d_values(api, batch * nnz * sizeof(double));
        DeviceBuffer d_v_re(api, batch * nbus * sizeof(double));
        DeviceBuffer d_v_im(api, batch * nbus * sizeof(double));
        DeviceBuffer d_inv_vmag(api, batch * nbus * sizeof(double));
        DeviceBuffer d_scalc_re(api, batch * nbus * sizeof(double));
        DeviceBuffer d_scalc_im(api, batch * nbus * sizeof(double));
        DeviceBuffer d_mismatch(api, batch * n * sizeof(double));
        DeviceBuffer d_expected(api, batch * n * sizeof(double));
        DeviceBuffer d_rhs(api, batch * n * sizeof(double));
        DeviceBuffer d_solution(api, batch * n * sizeof(double));
        DeviceBuffer d_mask(api, batch * sizeof(int));
        DeviceBuffer d_norms(api, batch * sizeof(double));
        d_row_ptr.upload(fixture.row_ptr.data(), (n + 1) * sizeof(int));
        d_col_indices.upload(fixture.col_indices.data(), nnz * sizeof(int));
        d_y_col_ptr.upload(fixture.y_col_ptr.data(), (nbus + 1) * sizeof(int));
        d_y_row_indices.upload(fixture.y_row_indices.data(), ynnz * sizeof(int));
        d_mirror.upload(fixture.mirror.data(), ynnz * sizeof(int));
        d_y_re.upload(fixture.y_re.data(), ynnz * sizeof(double));
        d_y_im.upload(fixture.y_im.data(), ynnz * sizeof(double));
        d_base_v_re.upload(fixture.v_re.data(), nbus * sizeof(double));
        d_base_v_im.upload(fixture.v_im.data(), nbus * sizeof(double));
        d_sbus_re.upload(fixture.sbus_re.data(), nbus * sizeof(double));
        d_sbus_im.upload(fixture.sbus_im.data(), nbus * sizeof(double));
        d_pq_ends.upload(fixture.pq_ends.data(), n_active * sizeof(int));
        d_active_ends.upload(fixture.active_ends.data(), n_active * sizeof(int));
        d_diag_ptrs.upload(fixture.diag_ptrs.data(), n_active * sizeof(int));
        std::vector<int> column_of_nnz(active_ynnz);
        for (std::size_t k = 0; k < n_active; ++k) {
            for (int p = fixture.y_col_ptr[k]; p < fixture.y_col_ptr[k + 1]; ++p) {
                column_of_nnz[static_cast<std::size_t>(p)] = static_cast<int>(k);
            }
        }
        d_column_of_nnz.upload(column_of_nnz.data(), active_ynnz * sizeof(int));
        const std::vector<int> ones(batch, 1);
        d_mask.upload(ones.data(), batch * sizeof(int));
        cuda_check(api, api.cudaMemset_fn(d_solution.pointer, 0, d_solution.bytes), "clear solution");

        Stream stream(api);
        CudaKernels kernels(api, root / "target/acpf-cuda/kernels.ptx");
        NewtonKernels newton_kernels(api, root / "target/acpf-cuda/newton_kernels.ptx");
        void* base_v_re = d_base_v_re.pointer;
        void* base_v_im = d_base_v_im.pointer;
        void* v_re = d_v_re.pointer;
        void* v_im = d_v_im.pointer;
        void* inv_vmag = d_inv_vmag.pointer;
        void* expected = d_expected.pointer;
        void* values = d_values.pointer;
        std::size_t batch_arg = batch;
        std::size_t n_arg = n;
        std::size_t nbus_arg = nbus;
        std::size_t n_active_arg = n_active;
        std::size_t npq_arg = npq;
        std::size_t nnz_arg = nnz;
        std::size_t active_ynnz_arg = active_ynnz;
        void* prepare_args[] = {&base_v_re, &base_v_im, &v_re, &v_im, &inv_vmag, &expected,
                                &batch_arg, &nbus_arg, &n_arg};
        const auto prepare_ms = timed(stream, [&] {
            cu_check(api, api.cuLaunchKernel_fn(kernels.prepare, 65535, 1, 1, 256, 1, 1, 0,
                                                reinterpret_cast<CUstream>(stream.value), prepare_args, nullptr),
                     "launch prepare_voltage_kernel");
        });

        void* y_col_ptr = d_y_col_ptr.pointer;
        void* y_row_indices = d_y_row_indices.pointer;
        void* mirror = d_mirror.pointer;
        void* y_re = d_y_re.pointer;
        void* y_im = d_y_im.pointer;
        void* sbus_re = d_sbus_re.pointer;
        void* sbus_im = d_sbus_im.pointer;
        void* scalc_re = d_scalc_re.pointer;
        void* scalc_im = d_scalc_im.pointer;
        void* mismatch = d_mismatch.pointer;
        void* power_args[] = {&y_col_ptr, &y_row_indices, &mirror, &y_re, &y_im, &v_re, &v_im,
                              &sbus_re, &sbus_im, &scalc_re, &scalc_im, &mismatch, &batch_arg,
                              &nbus_arg, &n_active_arg, &npq_arg, &n_arg};
        const auto power_ms = timed(stream, [&] {
            cu_check(api, api.cuLaunchKernel_fn(kernels.power_mismatch, grid_for(batch * nbus * 32, 256), 1, 1,
                                                256, 1, 1, 0, reinterpret_cast<CUstream>(stream.value), power_args, nullptr),
                     "launch batch_power_mismatch_kernel");
        });

        void* pq_ends = d_pq_ends.pointer;
        void* active_ends = d_active_ends.pointer;
        void* diag_ptrs = d_diag_ptrs.pointer;
        void* row_ptr = d_row_ptr.pointer;
        void* jacobian_args[] = {&y_col_ptr, &y_row_indices, &mirror, &y_re, &y_im, &v_re, &v_im,
                                 &inv_vmag, &scalc_re, &scalc_im, &pq_ends, &active_ends, &diag_ptrs,
                                 &row_ptr, &values, &batch_arg, &n_active_arg, &npq_arg, &nnz_arg,
                                 &active_ynnz_arg, &nbus_arg};
        const auto jacobian_ms = timed(stream, [&] {
            cu_check(api, api.cuLaunchKernel_fn(kernels.jacobian, 65535, 1, 1, 256, 1, 1, 0,
                                                reinterpret_cast<CUstream>(stream.value), jacobian_args, nullptr),
                     "launch batch_jacobian_csr_kernel");
        });

        void* col_indices = d_col_indices.pointer;
        void* rhs = d_rhs.pointer;
        void* matvec_args[] = {&row_ptr, &col_indices, &values, &expected, &rhs, &batch_arg, &n_arg, &nnz_arg};
        const auto matvec_ms = timed(stream, [&] {
            cu_check(api, api.cuLaunchKernel_fn(kernels.matvec, grid_for(batch * n * 32, 256), 1, 1, 256, 1, 1, 0,
                                                reinterpret_cast<CUstream>(stream.value), matvec_args, nullptr),
                     "launch batch_csr_matvec_kernel");
        });

        auto validate_base = [](const std::vector<double>& actual, const std::vector<double>& reference,
                                const char* label) {
            double max_error = 0.0;
            double max_reference = 0.0;
            for (std::size_t i = 0; i < reference.size(); ++i) {
                max_error = std::max(max_error, std::abs(actual[i] - reference[i]));
                max_reference = std::max(max_reference, std::abs(reference[i]));
            }
            require(max_error <= 1e-9 * std::max(1.0, max_reference),
                    std::string(label) + " mismatch: max error " + std::to_string(max_error));
            return max_error;
        };
        std::vector<double> base_jacobian(nnz), base_mismatch(n), base_scalc_re(nbus), base_scalc_im(nbus);
        d_values.download(base_jacobian.data(), nnz * sizeof(double));
        d_mismatch.download(base_mismatch.data(), n * sizeof(double));
        d_scalc_re.download(base_scalc_re.data(), nbus * sizeof(double));
        d_scalc_im.download(base_scalc_im.data(), nbus * sizeof(double));
        const auto jacobian_error = validate_base(base_jacobian, fixture.values, "Jacobian");
        const auto mismatch_error = validate_base(base_mismatch, fixture.rhs, "mismatch");
        const auto scalc_re_error = validate_base(base_scalc_re, fixture.scalc_re, "S_calc real");
        const auto scalc_im_error = validate_base(base_scalc_im, fixture.scalc_im, "S_calc imaginary");

        std::unique_ptr<DeviceBuffer> d_values32, d_expected32, d_rhs32, d_solution32;
        double fp32_cast_ms = 0.0;
        double fp32_matvec_ms = 0.0;
        void* solver_values = d_values.pointer;
        void* solver_rhs = d_rhs.pointer;
        void* solver_solution = d_solution.pointer;
        cudaDataType solver_type = CUDA_R_64F;
        if (fp32_solver) {
            solver_type = CUDA_R_32F;
            d_values32 = std::make_unique<DeviceBuffer>(api, batch * nnz * sizeof(float));
            d_expected32 = std::make_unique<DeviceBuffer>(api, batch * n * sizeof(float));
            d_rhs32 = std::make_unique<DeviceBuffer>(api, batch * n * sizeof(float));
            d_solution32 = std::make_unique<DeviceBuffer>(api, batch * n * sizeof(float));
            void* values32 = d_values32->pointer;
            void* expected32 = d_expected32->pointer;
            void* rhs32 = d_rhs32->pointer;
            std::size_t value_count = batch * nnz;
            std::size_t vector_count = batch * n;
            void* cast_values_args[] = {&values, &values32, &value_count};
            fp32_cast_ms += timed(stream, [&] {
                cu_check(api, api.cuLaunchKernel_fn(kernels.cast_fp32, 65535, 1, 1, 256, 1, 1, 0,
                                                    reinterpret_cast<CUstream>(stream.value), cast_values_args, nullptr),
                         "cast Jacobian to FP32");
            });
            void* cast_expected_args[] = {&expected, &expected32, &vector_count};
            fp32_cast_ms += timed(stream, [&] {
                cu_check(api, api.cuLaunchKernel_fn(kernels.cast_fp32, 65535, 1, 1, 256, 1, 1, 0,
                                                    reinterpret_cast<CUstream>(stream.value), cast_expected_args, nullptr),
                         "cast expected solution to FP32");
            });
            void* matvec32_args[] = {&row_ptr, &col_indices, &values32, &expected32, &rhs32,
                                     &batch_arg, &n_arg, &nnz_arg};
            fp32_matvec_ms = timed(stream, [&] {
                cu_check(api, api.cuLaunchKernel_fn(kernels.matvec_fp32, grid_for(batch * n * 32, 256), 1, 1, 256, 1, 1,
                                                    0, reinterpret_cast<CUstream>(stream.value), matvec32_args, nullptr),
                         "launch FP32 batch CSR matvec");
            });
            solver_values = values32;
            solver_rhs = rhs32;
            solver_solution = d_solution32->pointer;
        }

        cudssHandle_t handle{};
        cudssConfig_t config{};
        cudssData_t data{};
        cudssMatrix_t matrix{};
        cudssMatrix_t solution{};
        cudssMatrix_t rhs_matrix{};
        cudss_check(api.cudssCreate_fn(&handle), "cudssCreate");
        cudss_check(api.cudssConfigCreate_fn(&config), "cudssConfigCreate");
        cudss_check(api.cudssDataCreate_fn(handle, &data), "cudssDataCreate");
        cudss_check(api.cudssSetStream_fn(handle, stream.value), "cudssSetStream");
        const auto jit_dir = root / "target/acpf-cuda/cudss-jit-cache";
        fs::create_directories(jit_dir);
        cudss_check(api.cudssSetJitCacheDir_fn(handle, jit_dir.string().c_str()), "cudssSetJitCacheDir");
        const int batch_config = static_cast<int>(batch);
        require(batch_config > 0 && static_cast<std::size_t>(batch_config) == batch, "batch exceeds cuDSS int range");
        cudss_check(api.cudssConfigSet_fn(config, CUDSS_CONFIG_UBATCH_SIZE, &batch_config, sizeof(batch_config)),
                    "set CUDSS_CONFIG_UBATCH_SIZE");
        cudss_check(api.cudssDataSet_fn(handle, data, CUDSS_DATA_UBATCH_MASK, d_mask.pointer, d_mask.bytes),
                    "set CUDSS_DATA_UBATCH_MASK");
        cudss_check(api.cudssMatrixCreateCsr_fn(&matrix, n, n, nnz, d_row_ptr.pointer, nullptr,
                                                d_col_indices.pointer, solver_values, CUDA_R_32I, CUDA_R_32I,
                                                solver_type, CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO),
                    "cudssMatrixCreateCsr");
        cudss_check(api.cudssMatrixCreateDn_fn(&rhs_matrix, n, 1, n, solver_rhs, solver_type,
                                               CUDSS_LAYOUT_COL_MAJOR),
                    "create batched RHS");
        cudss_check(api.cudssMatrixCreateDn_fn(&solution, n, 1, n, solver_solution, solver_type,
                                               CUDSS_LAYOUT_COL_MAJOR),
                    "create batched solution");

        const auto analysis_ms = timed(stream, [&] {
            cudss_check(api.cudssExecute_fn(handle, CUDSS_PHASE_ANALYSIS, config, data, matrix, solution, rhs_matrix),
                        "cuDSS batch analysis");
        });
        std::vector<double> factor_ms, solve_ms;
        std::vector<double> newton_power_ms, newton_jacobian_ms, newton_convergence_ms, newton_update_ms, newton_cast_ms;
        std::vector<int> newton_iterations(batch, 0), host_mask = ones;
        std::vector<double> host_norms(batch, std::numeric_limits<double>::infinity());
        bool newton_converged = !newton_mode;
        double newton_wall_ms = 0.0;
        const double tolerance = 1e-8;
        constexpr int max_newton_iterations = 30;
        if (newton_mode) {
            const auto newton_start = Clock::now();
            void* mask = d_mask.pointer;
            void* norms = d_norms.pointer;
            void* rhs_ptr = d_rhs.pointer;
            void* dx = mixed_newton_mode ? d_solution32->pointer : d_solution.pointer;
            void* values32 = mixed_newton_mode ? d_values32->pointer : nullptr;
            void* rhs32 = mixed_newton_mode ? d_rhs32->pointer : nullptr;
            const std::size_t value_count = batch * nnz;
            const std::size_t vector_count = batch * n;
            void* cast_values_args[] = {&values, &values32, const_cast<std::size_t*>(&value_count)};
            void* cast_rhs_args[] = {&rhs_ptr, &rhs32, const_cast<std::size_t*>(&vector_count)};
            d_mask.upload(ones.data(), batch * sizeof(int));
            void* newton_prepare_args[] = {&base_v_re, &base_v_im, &v_re, &v_im, &inv_vmag, &batch_arg, &nbus_arg};
            cu_check(api, api.cuLaunchKernel_fn(newton_kernels.prepare, 65535, 1, 1, 256, 1, 1, 0,
                                                reinterpret_cast<CUstream>(stream.value), newton_prepare_args, nullptr),
                     "launch Newton voltage preparation");
            void* power_args_newton[] = {&y_col_ptr, &y_row_indices, &mirror, &y_re, &y_im, &v_re, &v_im,
                                         &sbus_re, &sbus_im, &scalc_re, &scalc_im, &mismatch, &rhs_ptr, &mask,
                                         &batch_arg, &nbus_arg, &n_active_arg, &npq_arg, &n_arg};
            void* convergence_args[] = {&mismatch, &mask, &norms, &batch_arg, &n_arg, const_cast<double*>(&tolerance)};
            void* jacobian_args_newton[] = {&y_col_ptr, &y_row_indices, &mirror, &y_re, &y_im, &v_re, &v_im,
                                            &inv_vmag, &scalc_re, &scalc_im, &pq_ends, &active_ends, &diag_ptrs,
                                            &row_ptr, &values, &mask, &batch_arg, &n_active_arg, &npq_arg, &nnz_arg,
                                            &active_ynnz_arg, &nbus_arg};
            void* column_of_nnz_ptr = d_column_of_nnz.pointer;
            void* jacobian_args_mapped[] = {&y_col_ptr, &y_row_indices, &mirror, &y_re, &y_im, &v_re, &v_im,
                                            &inv_vmag, &scalc_re, &scalc_im, &pq_ends, &active_ends, &diag_ptrs,
                                            &row_ptr, &values, &mask, &column_of_nnz_ptr, &batch_arg, &n_active_arg,
                                            &npq_arg, &nnz_arg, &active_ynnz_arg, &nbus_arg};
            GroupedJacobianArgs grouped_args{};
            grouped_args.ybus = {static_cast<const int*>(d_y_col_ptr.pointer),
                                 static_cast<const int*>(d_y_row_indices.pointer),
                                 static_cast<const int*>(d_mirror.pointer),
                                 static_cast<const double*>(d_y_re.pointer),
                                 static_cast<const double*>(d_y_im.pointer)};
            grouped_args.state = {static_cast<const double*>(d_v_re.pointer),
                                  static_cast<const double*>(d_v_im.pointer),
                                  static_cast<const double*>(d_inv_vmag.pointer),
                                  static_cast<const double*>(d_scalc_re.pointer),
                                  static_cast<const double*>(d_scalc_im.pointer)};
            grouped_args.jacobian = {static_cast<const int*>(d_pq_ends.pointer),
                                     static_cast<const int*>(d_active_ends.pointer),
                                     static_cast<const int*>(d_diag_ptrs.pointer),
                                     static_cast<const int*>(d_row_ptr.pointer),
                                     static_cast<double*>(d_values.pointer)};
            grouped_args.mask = static_cast<const int*>(d_mask.pointer);
            require(n_active <= static_cast<std::size_t>(std::numeric_limits<int>::max()) &&
                    npq <= static_cast<std::size_t>(std::numeric_limits<int>::max()) &&
                    nbus <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
                    "grouped Jacobian dimensions exceed int range");
            grouped_args.n_active = static_cast<int>(n_active);
            grouped_args.npq = static_cast<int>(npq);
            grouped_args.nbus = static_cast<int>(nbus);
            grouped_args.batch = batch_config;
            grouped_args.nnz = nnz;
            void* grouped_launch_args[] = {&grouped_args};
            CUfunction grouped_jacobian = newton_kernels.jacobian_l2;
            if (jacobian_lanes == 4) grouped_jacobian = newton_kernels.jacobian_l4;
            if (jacobian_lanes == 8) grouped_jacobian = newton_kernels.jacobian_l8;
            require(jacobian_lanes == 1 || batch <= 65535, "grouped Jacobian batch exceeds CUDA grid Y limit");
            void* update_args[] = {&v_re, &v_im, &inv_vmag, &dx, &mask,
                                   &batch_arg, &nbus_arg, &n_active_arg, &npq_arg, &n_arg};
            for (int iteration = 0; iteration <= max_newton_iterations; ++iteration) {
                newton_power_ms.push_back(timed(stream, [&] {
                    cu_check(api, api.cuLaunchKernel_fn(newton_kernels.power_mismatch, grid_for(batch * nbus * 32, 256),
                                                        1, 1, 256, 1, 1, 0, reinterpret_cast<CUstream>(stream.value),
                                                        power_args_newton, nullptr), "launch Newton power/mismatch");
                }));
                newton_convergence_ms.push_back(timed(stream, [&] {
                    cu_check(api, api.cuLaunchKernel_fn(newton_kernels.convergence, grid_for(batch, 1), 1, 1, 256, 1, 1,
                                                        0, reinterpret_cast<CUstream>(stream.value), convergence_args, nullptr),
                             "launch Newton convergence");
                }));
                d_mask.download(host_mask.data(), batch * sizeof(int));
                d_norms.download(host_norms.data(), batch * sizeof(double));
                std::size_t active_count = 0;
                double active_max_norm = 0.0;
                for (std::size_t scenario = 0; scenario < batch; ++scenario) {
                    require(host_mask[scenario] >= 0, "non-finite Newton voltage update");
                    if (host_mask[scenario] == 1) {
                        ++active_count;
                        active_max_norm = std::max(active_max_norm, host_norms[scenario]);
                    }
                }
                std::cout << "Newton evaluation " << iteration << ": active " << active_count
                          << ", max mismatch " << active_max_norm << "\n" << std::flush;
                if (active_count == 0) {
                    newton_converged = true;
                    break;
                }
                require(iteration < max_newton_iterations, "Newton did not converge within 30 updates");
                for (std::size_t scenario = 0; scenario < batch; ++scenario) {
                    if (host_mask[scenario] == 1) ++newton_iterations[scenario];
                }
                newton_jacobian_ms.push_back(timed(stream, [&] {
                    if (mapped_jacobian) {
                        cu_check(api, api.cuLaunchKernel_fn(newton_kernels.jacobian_mapped, 65535, 1, 1, 256, 1, 1, 0,
                                                            reinterpret_cast<CUstream>(stream.value), jacobian_args_mapped, nullptr),
                                 "launch Newton mapped Jacobian");
                    } else if (jacobian_lanes == 1) {
                        cu_check(api, api.cuLaunchKernel_fn(newton_kernels.jacobian, 65535, 1, 1, 256, 1, 1, 0,
                                                            reinterpret_cast<CUstream>(stream.value), jacobian_args_newton, nullptr),
                                 "launch Newton itemwise Jacobian");
                    } else {
                        const unsigned grid_x = grid_for(n_active * static_cast<std::size_t>(jacobian_lanes), 256);
                        cu_check(api, api.cuLaunchKernel_fn(grouped_jacobian, grid_x, static_cast<unsigned>(batch), 1,
                                                            256, 1, 1, 0, reinterpret_cast<CUstream>(stream.value),
                                                            grouped_launch_args, nullptr),
                                 "launch Newton grouped Jacobian");
                    }
                }));
                if (mixed_newton_mode) {
                    newton_cast_ms.push_back(timed(stream, [&] {
                        cu_check(api, api.cuLaunchKernel_fn(kernels.cast_fp32, 65535, 1, 1, 256, 1, 1, 0,
                                                            reinterpret_cast<CUstream>(stream.value), cast_values_args, nullptr),
                                 "cast Newton Jacobian to FP32");
                        cu_check(api, api.cuLaunchKernel_fn(kernels.cast_fp32, 65535, 1, 1, 256, 1, 1, 0,
                                                            reinterpret_cast<CUstream>(stream.value), cast_rhs_args, nullptr),
                                 "cast Newton RHS to FP32");
                    }));
                }
                cudss_check(api.cudssDataSet_fn(handle, data, CUDSS_DATA_UBATCH_MASK, d_mask.pointer, d_mask.bytes),
                            "refresh cuDSS active mask");
                const int phase = iteration == 0 ? CUDSS_PHASE_FACTORIZATION : CUDSS_PHASE_REFACTORIZATION;
                factor_ms.push_back(timed(stream, [&] {
                    cudss_check(api.cudssExecute_fn(handle, phase, config, data, matrix, solution, rhs_matrix),
                                "cuDSS Newton factorization");
                }));
                solve_ms.push_back(timed(stream, [&] {
                    cudss_check(api.cudssExecute_fn(handle, CUDSS_PHASE_SOLVE, config, data, matrix, solution, rhs_matrix),
                                "cuDSS Newton solve");
                }));
                newton_update_ms.push_back(timed(stream, [&] {
                    const CUfunction update_kernel = mixed_newton_mode ? newton_kernels.update_fp32 : newton_kernels.update;
                    cu_check(api, api.cuLaunchKernel_fn(update_kernel, 65535, 1, 1, 256, 1, 1, 0,
                                                        reinterpret_cast<CUstream>(stream.value), update_args, nullptr),
                             "launch Newton voltage update");
                }));
                std::cout << "  update " << iteration + 1 << ": factor " << factor_ms.back() << " ms, solve "
                          << solve_ms.back() << " ms\n" << std::flush;
            }
            newton_wall_ms = std::chrono::duration<double, std::milli>(Clock::now() - newton_start).count();
            require(newton_converged, "batch contains a non-converged scenario");
        } else {
            for (int repeat = 0; repeat < repeats; ++repeat) {
                const int phase = repeat == 0 ? CUDSS_PHASE_FACTORIZATION : CUDSS_PHASE_REFACTORIZATION;
                factor_ms.push_back(timed(stream, [&] {
                    cudss_check(api.cudssExecute_fn(handle, phase, config, data, matrix, solution, rhs_matrix),
                                "cuDSS batch factorization");
                }));
                solve_ms.push_back(timed(stream, [&] {
                    cudss_check(api.cudssExecute_fn(handle, CUDSS_PHASE_SOLVE, config, data, matrix, solution, rhs_matrix),
                                "cuDSS batch solve");
                }));
                std::cout << "repeat " << repeat << ": factor " << factor_ms.back() << " ms, solve "
                          << solve_ms.back() << " ms\n" << std::flush;
            }
        }

        std::int64_t estimates[16]{};
        std::size_t written{};
        cudss_check(api.cudssDataGet_fn(handle, data, CUDSS_DATA_MEMORY_ESTIMATES, estimates,
                                        sizeof(estimates), &written),
                    "get cuDSS memory estimates");
        std::int64_t lu_nnz{};
        cudss_check(api.cudssDataGet_fn(handle, data, CUDSS_DATA_LU_NNZ, &lu_nnz, sizeof(lu_nnz), &written),
                    "get cuDSS LU nnz");
        std::int32_t solver_info{};
        cudss_check(api.cudssDataGet_fn(handle, data, CUDSS_DATA_INFO, &solver_info, sizeof(solver_info), &written),
                    "get cuDSS solver info");
        require(solver_info == 0, "cuDSS reports numerical failure: " + std::to_string(solver_info));

        double max_abs_error = 0.0;
        double max_expected = 1.0;
        std::size_t worst_scenario = 0;
        std::size_t nonfinite = 0;
        double max_final_mismatch = 0.0;
        if (newton_mode) {
            std::vector<double> actual_re(batch * nbus), actual_im(batch * nbus);
            d_v_re.download(actual_re.data(), actual_re.size() * sizeof(double));
            d_v_im.download(actual_im.data(), actual_im.size() * sizeof(double));
            for (std::size_t p = 0; p < actual_re.size(); ++p) {
                if (!std::isfinite(actual_re[p]) || !std::isfinite(actual_im[p])) ++nonfinite;
            }
            for (std::size_t bus = 0; bus < nbus; ++bus) {
                max_abs_error = std::max(max_abs_error, std::abs(actual_re[bus] - fixture.converged_v_re[bus]));
                max_abs_error = std::max(max_abs_error, std::abs(actual_im[bus] - fixture.converged_v_im[bus]));
            }
            for (std::size_t scenario = 0; scenario < batch; ++scenario) {
                require(host_mask[scenario] == 0 && host_norms[scenario] <= tolerance,
                        "scenario failed final convergence validation");
                max_final_mismatch = std::max(max_final_mismatch, host_norms[scenario]);
            }
            require(mixed_newton_mode || newton_iterations[0] == fixture.converged_iterations,
                    "scenario 0 FP64 iteration count differs from the reference");
            require(nonfinite == 0 && max_abs_error < 1e-8,
                    "converged voltage differs from reference: " + std::to_string(max_abs_error));
        } else {
            std::vector<double> result(batch * n);
            if (fp32_mode) {
                std::vector<float> result32(batch * n);
                d_solution32->download(result32.data(), result32.size() * sizeof(float));
                std::copy(result32.begin(), result32.end(), result.begin());
            } else {
                d_solution.download(result.data(), result.size() * sizeof(double));
            }
            max_expected = 0.0;
            for (std::size_t scenario = 0; scenario < batch; ++scenario) {
                for (std::size_t row = 0; row < n; ++row) {
                    const double actual = result[scenario * n + row];
                    const double wanted = 0.5 + 0.01 * ((row + 3 * scenario) % 11);
                    if (!std::isfinite(actual)) ++nonfinite;
                    const double error = std::abs(actual - wanted);
                    if (error > max_abs_error) {
                        max_abs_error = error;
                        worst_scenario = scenario;
                    }
                    max_expected = std::max(max_expected, std::abs(wanted));
                }
            }
        }
        const double relative_error = max_abs_error / max_expected;
        const double solution_tolerance = fp32_mode ? 1e-2 : 1e-9;
        require(nonfinite == 0 && (newton_mode || relative_error < solution_tolerance),
                "batched solution validation failed: nonfinite=" + std::to_string(nonfinite) +
                    ", relative error=" + std::to_string(relative_error));

        int version[3]{};
        for (int i = 0; i < 3; ++i) {
            cudss_check(api.cudssGetProperty_fn(static_cast<libraryPropertyType>(i), &version[i]),
                        "cudssGetProperty");
        }
        const auto free_after = free_memory(api);
        const std::string jacobian_variant = mapped_jacobian ? "mapped" : (jacobian_lanes == 1 ? "itemwise" : "l" + std::to_string(jacobian_lanes));
        const std::string newton_prefix = mixed_newton_mode ? "newton-mixed-" : "newton-fp64-";
        const std::string result_prefix = newton_mode ? newton_prefix + jacobian_variant + "-b"
                                                      : (fp32_mode ? "linear-fp32-b" : "assembled-fp64-b");
        const auto result_path = root / ("target/acpf-cuda/" + result_prefix + std::to_string(batch) + ".json");
        std::ofstream json(result_path);
        json << std::setprecision(17);
        json << "{\n  \"schema_version\": 3,\n  \"status\": \"complete\",\n";
        json << "  \"mode\": \"" << (newton_mode ? (mixed_newton_mode ? "full_newton_mixed" : "full_newton")
                                                     : (fp32_mode ? "single_step_fp32" : "single_step")) << "\",\n";
        json << "  \"jacobian_variant\": \"" << jacobian_variant << "\",\n";
        json << "  \"gpu\": \"" << properties.name << "\",\n";
        json << "  \"cudss_version\": [" << version[0] << ", " << version[1] << ", " << version[2] << "],\n";
        json << "  \"precision\": \"" << (mixed_newton_mode ? "fp32_solve_fp64_residual" : (fp32_mode ? "float32" : "float64"))
             << "\",\n  \"batch_size\": " << batch << ",\n";
        json << "  \"matrix_order\": " << n << ",\n  \"nnz_per_system\": " << nnz << ",\n";
        json << "  \"analysis_ms\": " << analysis_ms << ",\n  \"prepare_batch_ms\": " << prepare_ms
             << ",\n  \"batch_power_mismatch_ms\": " << power_ms
             << ",\n  \"batch_jacobian_assembly_ms\": " << jacobian_ms
             << ",\n  \"batch_rhs_matvec_ms\": " << matvec_ms << ",\n";
        if (fp32_mode) {
            json << "  \"fp32_cast_ms\": " << fp32_cast_ms
                 << ",\n  \"fp32_batch_rhs_matvec_ms\": " << fp32_matvec_ms << ",\n";
        }
        json << "  \"base_jacobian_max_abs_error\": " << jacobian_error
             << ",\n  \"base_mismatch_max_abs_error\": " << mismatch_error
             << ",\n  \"base_scalc_re_max_abs_error\": " << scalc_re_error
             << ",\n  \"base_scalc_im_max_abs_error\": " << scalc_im_error << ",\n";
        json << "  \"factorization_median_ms\": " << median(factor_ms) << ",\n";
        json << "  \"solve_median_ms\": " << median(solve_ms) << ",\n";
        json << "  \"factorization_all_ms\": [";
        for (std::size_t i = 0; i < factor_ms.size(); ++i) json << (i ? ", " : "") << factor_ms[i];
        json << "],\n  \"solve_all_ms\": [";
        for (std::size_t i = 0; i < solve_ms.size(); ++i) json << (i ? ", " : "") << solve_ms[i];
        if (newton_mode) {
            const auto minmax = std::minmax_element(newton_iterations.begin(), newton_iterations.end());
            json << "],\n  \"newton_converged\": true,\n";
            json << "  \"newton_tolerance\": " << tolerance << ",\n";
            json << "  \"newton_wall_ms\": " << newton_wall_ms << ",\n";
            json << "  \"newton_iterations_min\": " << *minmax.first
                 << ",\n  \"newton_iterations_max\": " << *minmax.second << ",\n";
            json << "  \"reference_iterations\": " << fixture.converged_iterations
                 << ",\n  \"reference_mismatch_inf_norm\": " << fixture.converged_norm << ",\n";
            json << "  \"max_final_mismatch_inf_norm\": " << max_final_mismatch << ",\n";
            json << "  \"newton_power_total_ms\": " << std::accumulate(newton_power_ms.begin(), newton_power_ms.end(), 0.0)
                 << ",\n  \"newton_jacobian_total_ms\": "
                 << std::accumulate(newton_jacobian_ms.begin(), newton_jacobian_ms.end(), 0.0)
                 << ",\n  \"newton_convergence_total_ms\": "
                 << std::accumulate(newton_convergence_ms.begin(), newton_convergence_ms.end(), 0.0)
                 << ",\n  \"newton_update_total_ms\": " << std::accumulate(newton_update_ms.begin(), newton_update_ms.end(), 0.0)
                 << ",\n  \"newton_fp32_cast_total_ms\": " << std::accumulate(newton_cast_ms.begin(), newton_cast_ms.end(), 0.0)
                 << ",\n  \"newton_factor_total_ms\": " << std::accumulate(factor_ms.begin(), factor_ms.end(), 0.0)
                 << ",\n  \"newton_solve_total_ms\": " << std::accumulate(solve_ms.begin(), solve_ms.end(), 0.0) << ",\n";
            json << "  \"newton_iterations_per_scenario\": [";
            for (std::size_t i = 0; i < batch; ++i) json << (i ? ", " : "") << newton_iterations[i];
            json << "],\n  \"final_mismatch_per_scenario\": [";
            for (std::size_t i = 0; i < batch; ++i) json << (i ? ", " : "") << host_norms[i];
            json << "],\n  \"newton_power_all_ms\": [";
            for (std::size_t i = 0; i < newton_power_ms.size(); ++i) json << (i ? ", " : "") << newton_power_ms[i];
            json << "],\n  \"newton_jacobian_all_ms\": [";
            for (std::size_t i = 0; i < newton_jacobian_ms.size(); ++i) json << (i ? ", " : "") << newton_jacobian_ms[i];
            json << "],\n  \"newton_convergence_all_ms\": [";
            for (std::size_t i = 0; i < newton_convergence_ms.size(); ++i) json << (i ? ", " : "") << newton_convergence_ms[i];
            json << "],\n  \"newton_update_all_ms\": [";
            for (std::size_t i = 0; i < newton_update_ms.size(); ++i) json << (i ? ", " : "") << newton_update_ms[i];
            json << "],\n  \"lu_nnz_reported\": " << lu_nnz << ",\n";
        } else {
            json << "],\n  \"lu_nnz_reported\": " << lu_nnz << ",\n";
        }
        json << "  \"memory_estimates\": {\"permanent_device\": " << estimates[0]
             << ", \"peak_device\": " << estimates[1] << ", \"permanent_host\": " << estimates[2]
             << ", \"peak_host\": " << estimates[3] << "},\n";
        json << "  \"free_memory_before\": " << free_before << ",\n  \"free_memory_after\": " << free_after << ",\n";
        json << "  \"max_abs_solution_error\": " << max_abs_error << ",\n";
        json << "  \"max_relative_solution_error\": " << relative_error << ",\n";
        json << "  \"worst_scenario\": " << worst_scenario << ",\n  \"nonfinite_solutions\": " << nonfinite
             << ",\n  \"validation_passed\": true\n}\n";
        json.close();

        std::cout << std::setprecision(10);
        if (newton_mode) {
            const auto minmax = std::minmax_element(newton_iterations.begin(), newton_iterations.end());
            std::cout << "FULL NEWTON CONVERGED: iterations " << *minmax.first << ".." << *minmax.second
                      << ", wall " << newton_wall_ms << " ms\n";
            std::cout << "max final mismatch: " << max_final_mismatch
                      << "; scenario 0 voltage max abs error: " << max_abs_error << "\n";
        }
        std::cout << "analysis: " << analysis_ms << " ms\n";
        std::cout << "voltage preparation: " << prepare_ms << " ms\n";
        std::cout << "batch power + mismatch: " << power_ms << " ms\n";
        std::cout << "batch CSR Jacobian assembly: " << jacobian_ms << " ms\n";
        std::cout << "batched RHS SpMV: " << matvec_ms << " ms\n";
        std::cout << "base J / mismatch max abs errors: " << jacobian_error << " / " << mismatch_error << "\n";
        std::cout << "max relative solution error: " << relative_error << "\n";
        std::cout << "cuDSS peak device estimate: " << estimates[1] << " bytes\n";
        std::cout << "result: " << result_path << "\n";

        api.cudssMatrixDestroy_fn(matrix);
        api.cudssMatrixDestroy_fn(solution);
        api.cudssMatrixDestroy_fn(rhs_matrix);
        api.cudssDataDestroy_fn(handle, data);
        api.cudssConfigDestroy_fn(config);
        api.cudssDestroy_fn(handle);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "acpf_batch_demo: " << error.what() << "\n";
        return 2;
    }
}
