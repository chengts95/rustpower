//! Experimental FP64 CUDA executor. One instance owns one stream and one cuDSS queue.
//! Multiple instances are the intended multi-stream experiment; no global device state.
use super::{
    BranchCatalog, BranchKind, LinearSolveStats, N1Config, N1Event, N1EventBatch, N1FaultSet,
    N1Initialization, N1Outcome, N1Reporting, N1Study, N1StudyError, N1StudyReport, N1Summary,
    PreparedN1, SharedBaseline, SolveStage, execution::worker_shared, outage::OutageTask,
    watch::WatchHits,
};
use crate::basic::branch::BranchFlowResult;
use num_complex::Complex64;
use std::{
    cell::Cell,
    ffi::{CStr, c_char},
    marker::PhantomData,
    mem,
    ptr::{self, NonNull},
    time::Instant,
};

const ABI_VERSION: u32 = 1;
const SUCCESS: i32 = 0;

#[repr(C)]
struct FfiExecutor {
    _private: [u8; 0],
}

#[repr(C)]
struct FfiBaseline {
    struct_size: u32,
    abi_version: u32,
    nbus: i32,
    npv: i32,
    npq: i32,
    j_nnz: i32,
    y_nnz: i32,
    j_row_ptr: *const i32,
    j_col_indices: *const i32,
    y_col_ptr: *const i32,
    y_row_indices: *const i32,
    mirror: *const i32,
    pq_ends: *const i32,
    active_ends: *const i32,
    diag_ptrs: *const i32,
    y_re: *const f64,
    y_im: *const f64,
    sbus_re: *const f64,
    sbus_im: *const f64,
    initial_v_re: *const f64,
    initial_v_im: *const f64,
}

#[repr(C)]
struct FfiSolveOptions {
    struct_size: u32,
    max_iterations: i32,
    tolerance: f64,
}

#[repr(C)]
struct FfiOutageBatch {
    struct_size: u32,
    patch_counts: *const i32,
    slots: *const i32,
    y_re: *const f64,
    y_im: *const f64,
}

#[repr(C)]
struct FfiBranchCatalog {
    _private: [u8; 0],
}

#[repr(C)]
struct FfiBranchCatalogInput {
    struct_size: u32,
    nbus: i32,
    branch_count: i32,
    from: *const i32,
    to: *const i32,
    y_ff_re: *const f64,
    y_ff_im: *const f64,
    y_tf_re: *const f64,
    y_tf_im: *const f64,
    y_ft_re: *const f64,
    y_ft_im: *const f64,
    y_tt_re: *const f64,
    y_tt_im: *const f64,
    base_i_from_ka: *const f64,
    base_i_to_ka: *const f64,
    inv_line_rating: *const f64,
    transformer_loading_scale: *const f64,
    in_service: *const u8,
    kind: *const u8,
    sbase: f64,
}

#[repr(C)]
struct FfiPostprocessStats {
    struct_size: u32,
    kernel_milliseconds: f64,
    download_milliseconds: f64,
}

struct CudaApi {
    // Keep the library alive until process exit; executors may live on any worker.
    _library: libloading::Library,
    probe: unsafe extern "C" fn() -> i32,
    create: unsafe extern "C" fn(*const FfiBaseline, i32, *mut *mut FfiExecutor) -> i32,
    solve_base_batch: unsafe extern "C" fn(
        *mut FfiExecutor,
        i32,
        *const FfiSolveOptions,
        *mut i32,
        *mut i32,
        *mut f64,
        *mut f64,
        *mut f64,
    ) -> i32,
    solve_outage_batch: unsafe extern "C" fn(
        *mut FfiExecutor,
        i32,
        *const FfiOutageBatch,
        *const FfiSolveOptions,
        *mut i32,
        *mut i32,
        *mut f64,
        *mut f64,
        *mut f64,
    ) -> i32,
    branch_catalog_create:
        unsafe extern "C" fn(*const FfiBranchCatalogInput, *mut *mut FfiBranchCatalog) -> i32,
    branch_catalog_destroy: unsafe extern "C" fn(*mut FfiBranchCatalog),
    postprocess_branches: unsafe extern "C" fn(
        *mut FfiExecutor,
        *const FfiBranchCatalog,
        *const i32,
        *mut f64,
        *mut FfiPostprocessStats,
    ) -> i32,
    last_error: unsafe extern "C" fn(*const FfiExecutor) -> *const c_char,
    destroy: unsafe extern "C" fn(*mut FfiExecutor),
}

impl CudaApi {
    fn load() -> Result<Self, String> {
        unsafe fn symbol<T: Copy>(
            library: &libloading::Library,
            name: &'static [u8],
        ) -> Result<T, String> {
            let name_text = String::from_utf8_lossy(&name[..name.len() - 1]);
            unsafe { library.get::<T>(name) }
                .map(|value| *value)
                .map_err(|error| format!("load CUDA symbol {name_text}: {error}"))
        }
        let library = open_cuda_library()?;
        Ok(Self {
            probe: unsafe { symbol(&library, b"rp_cuda_probe\0")? },
            create: unsafe { symbol(&library, b"rp_cuda_create\0")? },
            solve_base_batch: unsafe { symbol(&library, b"rp_cuda_solve_base_batch\0")? },
            solve_outage_batch: unsafe { symbol(&library, b"rp_cuda_solve_outage_batch\0")? },
            branch_catalog_create: unsafe { symbol(&library, b"rp_cuda_branch_catalog_create\0")? },
            branch_catalog_destroy: unsafe {
                symbol(&library, b"rp_cuda_branch_catalog_destroy\0")?
            },
            postprocess_branches: unsafe { symbol(&library, b"rp_cuda_postprocess_branches\0")? },
            last_error: unsafe { symbol(&library, b"rp_cuda_last_error\0")? },
            destroy: unsafe { symbol(&library, b"rp_cuda_destroy\0")? },
            _library: library,
        })
    }
}

fn cuda_api() -> Result<&'static CudaApi, String> {
    static API: std::sync::OnceLock<Result<CudaApi, String>> = std::sync::OnceLock::new();
    API.get_or_init(CudaApi::load)
        .as_ref()
        .map_err(Clone::clone)
}

#[cfg(windows)]
fn open_cuda_library() -> Result<libloading::Library, String> {
    use libloading::os::windows::Library;
    use std::{
        ffi::c_void,
        os::windows::ffi::{OsStrExt, OsStringExt},
        path::{Path, PathBuf},
    };

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetModuleHandleExW(flags: u32, address: *const u16, module: *mut *mut c_void) -> i32;
        fn GetModuleFileNameW(module: *mut c_void, path: *mut u16, size: u32) -> u32;
        fn AddDllDirectory(directory: *const u16) -> *mut c_void;
    }
    const FROM_ADDRESS: u32 = 0x0000_0004;
    const UNCHANGED_REFCOUNT: u32 = 0x0000_0002;
    const LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR: u32 = 0x0000_0100;
    const LOAD_LIBRARY_SEARCH_DEFAULT_DIRS: u32 = 0x0000_1000;
    // Resolve beside this .pyd/.dll, not beside python.exe.
    let mut module = ptr::null_mut();
    if unsafe {
        GetModuleHandleExW(
            FROM_ADDRESS | UNCHANGED_REFCOUNT,
            open_cuda_library as *const () as usize as *const u16,
            &mut module,
        )
    } == 0
    {
        return Err(format!(
            "locate the RustPower native module: {}",
            std::io::Error::last_os_error()
        ));
    }
    let mut wide = vec![0_u16; 32_768];
    let length = unsafe { GetModuleFileNameW(module, wide.as_mut_ptr(), wide.len() as u32) };
    if length == 0 || length as usize == wide.len() {
        return Err("read the RustPower native module path".into());
    }
    wide.truncate(length as usize);
    let mut module_dir = PathBuf::from(std::ffi::OsString::from_wide(&wide));
    module_dir.pop();

    fn add_search_directory(path: &Path) -> Result<(), String> {
        if !path.is_dir() {
            return Ok(());
        }
        let wide: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
        // Cookies are deliberately retained for process lifetime. Removing one
        // while a worker owns a CUDA executor could invalidate dependency lookup.
        if unsafe { AddDllDirectory(wide.as_ptr()) }.is_null() {
            return Err(format!(
                "register CUDA dependency directory {}: {}",
                path.display(),
                std::io::Error::last_os_error()
            ));
        }
        Ok(())
    }
    add_search_directory(&module_dir)?;
    // Development uses a merged nvidia/cu12 directory; pip installs each
    // component separately beside the rustpower package. Neither layout
    // requires mutating process PATH.
    let mut search_roots = vec![module_dir.clone()];
    if let Some(site_packages) = module_dir.parent() {
        search_roots.push(site_packages.to_path_buf());
    }
    for root in search_roots {
        for relative in [
            "nvidia/cu12/bin",
            "nvidia/cuda_runtime/bin",
            "nvidia/cublas/bin",
            "nvidia/cuda_nvrtc/bin",
            "nvidia/nvrtc/bin",
            "nvidia/nvjitlink/bin",
            "nvidia/cudss/bin",
        ] {
            add_search_directory(&root.join(relative))?;
        }
    }
    let path = module_dir.join("acpf_cuda.dll");
    unsafe {
        Library::load_with_flags(
            &path,
            LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS,
        )
    }
    .map(libloading::Library::from)
    .map_err(|error| {
        format!(
            "load {}: {error}; install the 'cuda' Python extra for the required runtime libraries",
            path.display()
        )
    })
}

#[cfg(not(windows))]
fn open_cuda_library() -> Result<libloading::Library, String> {
    Err("the experimental CUDA N-1 backend currently supports Windows only".into())
}

/// Probe before choosing a backend. This never changes a partially completed run.
pub fn cuda_status() -> Result<(), String> {
    if std::env::var_os("RUSTPOWER_N1_DISABLE_CUDA").is_some() {
        return Err("CUDA was disabled by RUSTPOWER_N1_DISABLE_CUDA".into());
    }
    let status = unsafe { (cuda_api()?.probe)() };
    if status == SUCCESS {
        Ok(())
    } else {
        Err(format!(
            "CUDA device probe failed ({status}): {}",
            last_error(None)
        ))
    }
}

// These narrow wrappers preserve the original FFI call sites while turning a
// missing runtime into an ordinary executor error instead of a loader failure.
unsafe fn rp_cuda_create(
    baseline: *const FfiBaseline,
    batch: i32,
    executor: *mut *mut FfiExecutor,
) -> i32 {
    match cuda_api() {
        Ok(api) => unsafe { (api.create)(baseline, batch, executor) },
        Err(_) => 4,
    }
}
#[allow(clippy::too_many_arguments)]
unsafe fn rp_cuda_solve_base_batch(
    executor: *mut FfiExecutor,
    batch: i32,
    options: *const FfiSolveOptions,
    case_status: *mut i32,
    iterations: *mut i32,
    residuals: *mut f64,
    voltage_re: *mut f64,
    voltage_im: *mut f64,
) -> i32 {
    match cuda_api() {
        Ok(api) => unsafe {
            (api.solve_base_batch)(
                executor,
                batch,
                options,
                case_status,
                iterations,
                residuals,
                voltage_re,
                voltage_im,
            )
        },
        Err(_) => 4,
    }
}
#[allow(clippy::too_many_arguments)]
unsafe fn rp_cuda_solve_outage_batch(
    executor: *mut FfiExecutor,
    batch: i32,
    outages: *const FfiOutageBatch,
    options: *const FfiSolveOptions,
    case_status: *mut i32,
    iterations: *mut i32,
    residuals: *mut f64,
    voltage_re: *mut f64,
    voltage_im: *mut f64,
) -> i32 {
    match cuda_api() {
        Ok(api) => unsafe {
            (api.solve_outage_batch)(
                executor,
                batch,
                outages,
                options,
                case_status,
                iterations,
                residuals,
                voltage_re,
                voltage_im,
            )
        },
        Err(_) => 4,
    }
}
unsafe fn rp_cuda_branch_catalog_create(
    catalog: *const FfiBranchCatalogInput,
    out_catalog: *mut *mut FfiBranchCatalog,
) -> i32 {
    match cuda_api() {
        Ok(api) => unsafe { (api.branch_catalog_create)(catalog, out_catalog) },
        Err(_) => 4,
    }
}
unsafe fn rp_cuda_branch_catalog_destroy(catalog: *mut FfiBranchCatalog) {
    if let Ok(api) = cuda_api() {
        unsafe { (api.branch_catalog_destroy)(catalog) }
    }
}
unsafe fn rp_cuda_postprocess_branches(
    executor: *mut FfiExecutor,
    catalog: *const FfiBranchCatalog,
    outage_indices: *const i32,
    output: *mut f64,
    stats: *mut FfiPostprocessStats,
) -> i32 {
    match cuda_api() {
        Ok(api) => unsafe {
            (api.postprocess_branches)(executor, catalog, outage_indices, output, stats)
        },
        Err(_) => 4,
    }
}
unsafe fn rp_cuda_last_error(executor: *const FfiExecutor) -> *const c_char {
    match cuda_api() {
        Ok(api) => unsafe { (api.last_error)(executor) },
        Err(_) => ptr::null(),
    }
}
unsafe fn rp_cuda_destroy(executor: *mut FfiExecutor) {
    if let Ok(api) = cuda_api() {
        unsafe { (api.destroy)(executor) }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CudaCaseStatus {
    Converged,
    MaxIterations,
    NonFinite,
    Unknown(i32),
}
impl CudaCaseStatus {
    fn from_raw(value: i32) -> Self {
        match value {
            0 => Self::Converged,
            1 => Self::MaxIterations,
            2 => Self::NonFinite,
            other => Self::Unknown(other),
        }
    }
}

#[derive(Debug)]
pub struct CudaCaseSolution {
    pub status: CudaCaseStatus,
    pub iterations: usize,
    pub residual: f64,
    /// Voltage in the shared solver's [PQ | PV | Slack] order.
    pub voltage: Vec<Complex64>,
}

#[derive(Debug)]
pub struct CudaBatchResult {
    pub cases: Vec<CudaCaseSolution>,
}

/// Immutable two-port constants uploaded once and shared by all postprocessing calls.
pub struct CudaBranchCatalog {
    handle: NonNull<FfiBranchCatalog>,
    nbus: usize,
    branch_count: usize,
    _not_sync: PhantomData<Cell<()>>,
}
unsafe impl Send for CudaBranchCatalog {}
impl CudaBranchCatalog {
    pub fn new(catalog: &BranchCatalog) -> Result<Self, String> {
        let branch_count = catalog.len();
        if branch_count == 0 {
            return Err("CUDA branch catalog must not be empty".into());
        }
        let nbus = catalog.solver_bus_count();
        let from = to_i32(catalog.from_buses(), "branch from-bus indices")?;
        let to = to_i32(catalog.to_buses(), "branch to-bus indices")?;
        if from
            .iter()
            .chain(&to)
            .any(|&bus| bus < 0 || bus as usize >= nbus)
        {
            return Err("branch endpoint exceeds solver bus count".into());
        }
        let [y_ff, y_tf, y_ft, y_tt] = catalog.admittances();
        let (y_ff_re, y_ff_im) = split_complex(y_ff);
        let (y_tf_re, y_tf_im) = split_complex(y_tf);
        let (y_ft_re, y_ft_im) = split_complex(y_ft);
        let (y_tt_re, y_tt_im) = split_complex(y_tt);
        let [base_i_from_ka, base_i_to_ka] = catalog.base_currents_ka();
        let [inv_line_rating, transformer_loading_scale] = catalog.loading_scales();
        let in_service: Vec<u8> = catalog
            .in_service()
            .iter()
            .map(|&value| value.into())
            .collect();
        let kind: Vec<u8> = catalog
            .identities()
            .iter()
            .map(|identity| match identity.kind {
                BranchKind::Line => 0,
                BranchKind::Transformer => 1,
            })
            .collect();
        let ffi = FfiBranchCatalogInput {
            struct_size: mem::size_of::<FfiBranchCatalogInput>() as u32,
            nbus: i32::try_from(nbus).map_err(|_| "branch bus count exceeds i32")?,
            branch_count: i32::try_from(branch_count).map_err(|_| "branch count exceeds i32")?,
            from: from.as_ptr(),
            to: to.as_ptr(),
            y_ff_re: y_ff_re.as_ptr(),
            y_ff_im: y_ff_im.as_ptr(),
            y_tf_re: y_tf_re.as_ptr(),
            y_tf_im: y_tf_im.as_ptr(),
            y_ft_re: y_ft_re.as_ptr(),
            y_ft_im: y_ft_im.as_ptr(),
            y_tt_re: y_tt_re.as_ptr(),
            y_tt_im: y_tt_im.as_ptr(),
            base_i_from_ka: base_i_from_ka.as_ptr(),
            base_i_to_ka: base_i_to_ka.as_ptr(),
            inv_line_rating: inv_line_rating.as_ptr(),
            transformer_loading_scale: transformer_loading_scale.as_ptr(),
            in_service: in_service.as_ptr(),
            kind: kind.as_ptr(),
            sbase: catalog.sbase(),
        };
        let mut handle = ptr::null_mut();
        let status = unsafe { rp_cuda_branch_catalog_create(&ffi, &mut handle) };
        if status != SUCCESS {
            return Err(format!(
                "CUDA branch catalog creation failed ({status}): {}",
                last_error(None)
            ));
        }
        let handle = NonNull::new(handle).ok_or("CUDA returned a null branch catalog")?;
        Ok(Self {
            handle,
            nbus,
            branch_count,
            _not_sync: PhantomData,
        })
    }

    pub fn len(&self) -> usize {
        self.branch_count
    }
}
impl Drop for CudaBranchCatalog {
    fn drop(&mut self) {
        unsafe { rp_cuda_branch_catalog_destroy(self.handle.as_ptr()) }
    }
}

#[derive(Debug)]
pub struct CudaBranchResults {
    pub flows: Vec<BranchFlowResult>,
    branch_count: usize,
    pub kernel_seconds: f64,
    pub download_seconds: f64,
}
impl CudaBranchResults {
    pub fn scenario(&self, scenario: usize) -> Option<&[BranchFlowResult]> {
        let start = scenario.checked_mul(self.branch_count)?;
        self.flows.get(start..start + self.branch_count)
    }
}

/// A fixed-batch, single-stream executor. It can move to a worker thread, but &mut
/// is required for every solve; separate instances own independent cuDSS queues.
pub struct CudaExecutor {
    handle: NonNull<FfiExecutor>,
    batch: usize,
    nbus: usize,
    _not_sync: PhantomData<Cell<()>>,
}
// CUDA streams and cuDSS handles have no Rust-thread affinity. The C++ object is
// exclusively accessed through &mut self and all host baseline pointers are borrowed
// only during construction, so transferring unique ownership between threads is safe.
unsafe impl Send for CudaExecutor {}
impl CudaExecutor {
    pub fn new(base: &SharedBaseline, batch: usize) -> Result<Self, String> {
        Self::new_inner(base, batch, false)
    }

    /// N-1 executor: every scenario starts from the prepared converged AC base.
    pub fn new_ac(base: &SharedBaseline, batch: usize) -> Result<Self, String> {
        Self::new_inner(base, batch, true)
    }

    fn new_inner(base: &SharedBaseline, batch: usize, use_ac_start: bool) -> Result<Self, String> {
        let batch_i32 = i32::try_from(batch).map_err(|_| "CUDA batch exceeds i32".to_string())?;
        if batch == 0 {
            return Err("CUDA batch must be positive".into());
        }
        let nbus = base.v_init().len();
        let active = base
            .npv()
            .checked_add(base.npq())
            .ok_or("CUDA bus partition overflow")?;
        let expected_n = active
            .checked_add(base.npq())
            .ok_or("CUDA equation count overflow")?;
        let j = base.j_pattern();
        if base.y_bus().nrows() != nbus
            || base.y_bus().ncols() != nbus
            || base.s_bus().len() != nbus
            || j.j_col_ptrs.len() != expected_n + 1
            || j.pq_ends.len() != active
            || j.active_ends.len() != active
            || j.diag_ptrs.len() != active
        {
            return Err("baseline dimensions do not match the CUDA ABI".into());
        }

        let j_row_ptr = to_i32(&j.j_col_ptrs, "Jacobian row pointers")?;
        let j_col_indices = to_i32(&j.j_row_indices, "Jacobian column indices")?;
        let y_col_ptr = to_i32(base.y_bus().col_offsets(), "Ybus column pointers")?;
        let y_row_indices = to_i32(base.y_bus().row_indices(), "Ybus row indices")?;
        let pq_ends = to_i32(&j.pq_ends, "PQ boundaries")?;
        let active_ends = to_i32(&j.active_ends, "active boundaries")?;
        let diag_ptrs = to_i32(&j.diag_ptrs, "diagonal positions")?;
        let mirror = mirror_positions(&y_col_ptr, &y_row_indices)?;
        let (y_re, y_im) = split_complex(base.y_bus().values());
        let (sbus_re, sbus_im) = split_complex(base.s_bus());
        let initial = if use_ac_start {
            base.ac_start()
                .ok_or("CUDA N-1 requires a prepared converged base AC voltage")?
        } else {
            base.v_init()
        };
        let (initial_v_re, initial_v_im) = split_complex(initial);
        let ffi = FfiBaseline {
            struct_size: mem::size_of::<FfiBaseline>() as u32,
            abi_version: ABI_VERSION,
            nbus: i32::try_from(nbus).map_err(|_| "bus count exceeds i32")?,
            npv: i32::try_from(base.npv()).map_err(|_| "PV count exceeds i32")?,
            npq: i32::try_from(base.npq()).map_err(|_| "PQ count exceeds i32")?,
            j_nnz: i32::try_from(j_col_indices.len()).map_err(|_| "Jacobian nnz exceeds i32")?,
            y_nnz: i32::try_from(y_row_indices.len()).map_err(|_| "Ybus nnz exceeds i32")?,
            j_row_ptr: j_row_ptr.as_ptr(),
            j_col_indices: j_col_indices.as_ptr(),
            y_col_ptr: y_col_ptr.as_ptr(),
            y_row_indices: y_row_indices.as_ptr(),
            mirror: mirror.as_ptr(),
            pq_ends: pq_ends.as_ptr(),
            active_ends: active_ends.as_ptr(),
            diag_ptrs: diag_ptrs.as_ptr(),
            y_re: y_re.as_ptr(),
            y_im: y_im.as_ptr(),
            sbus_re: sbus_re.as_ptr(),
            sbus_im: sbus_im.as_ptr(),
            initial_v_re: initial_v_re.as_ptr(),
            initial_v_im: initial_v_im.as_ptr(),
        };
        let mut handle = ptr::null_mut();
        // Every baseline allocation outlives this call; C++ uploads all data before returning.
        let status = unsafe { rp_cuda_create(&ffi, batch_i32, &mut handle) };
        if status != SUCCESS {
            return Err(format!(
                "CUDA executor creation failed ({status}): {}",
                last_error(None)
            ));
        }
        let handle = NonNull::new(handle).ok_or("CUDA returned success with a null executor")?;
        Ok(Self {
            handle,
            batch,
            nbus,
            _not_sync: PhantomData,
        })
    }

    pub fn batch_size(&self) -> usize {
        self.batch
    }

    pub fn solve_base(
        &mut self,
        max_iter: usize,
        tolerance: f64,
    ) -> Result<CudaBatchResult, String> {
        let options = FfiSolveOptions {
            struct_size: mem::size_of::<FfiSolveOptions>() as u32,
            max_iterations: i32::try_from(max_iter).map_err(|_| "max_iter exceeds i32")?,
            tolerance,
        };
        let count = self
            .batch
            .checked_mul(self.nbus)
            .ok_or("CUDA voltage output size overflow")?;
        let mut status = vec![0; self.batch];
        let mut iterations = vec![0; self.batch];
        let mut residuals = vec![0.; self.batch];
        let mut voltage_re = vec![0.; count];
        let mut voltage_im = vec![0.; count];
        // The executor is mutably borrowed, so its stream and error buffer cannot race in Rust.
        let code = unsafe {
            rp_cuda_solve_base_batch(
                self.handle.as_ptr(),
                self.batch as i32,
                &options,
                status.as_mut_ptr(),
                iterations.as_mut_ptr(),
                residuals.as_mut_ptr(),
                voltage_re.as_mut_ptr(),
                voltage_im.as_mut_ptr(),
            )
        };
        if code != SUCCESS {
            return Err(format!(
                "CUDA batch solve failed ({code}): {}",
                last_error(Some(self.handle))
            ));
        }
        let cases = (0..self.batch)
            .map(|scenario| {
                let start = scenario * self.nbus;
                let voltage = (0..self.nbus)
                    .map(|k| Complex64::new(voltage_re[start + k], voltage_im[start + k]))
                    .collect();
                CudaCaseSolution {
                    status: CudaCaseStatus::from_raw(status[scenario]),
                    iterations: iterations[scenario] as usize,
                    residual: residuals[scenario],
                    voltage,
                }
            })
            .collect();
        Ok(CudaBatchResult { cases })
    }

    pub fn solve_outages(
        &mut self,
        tasks: &[super::outage::OutageTask],
        max_iter: usize,
        tolerance: f64,
    ) -> Result<CudaBatchResult, String> {
        if tasks.len() != self.batch {
            return Err("CUDA outage task count must equal executor batch".into());
        }
        let mut counts = Vec::with_capacity(self.batch);
        let mut slots = vec![0_i32; self.batch * 4];
        let mut y_re = vec![0.0; self.batch * 4];
        let mut y_im = vec![0.0; self.batch * 4];
        for (scenario, task) in tasks.iter().enumerate() {
            let patch = task.y_patch();
            counts.push(i32::try_from(patch.len()).map_err(|_| "outage patch count exceeds i32")?);
            for k in 0..patch.len() {
                let offset = scenario * 4 + k;
                slots[offset] =
                    i32::try_from(patch.slots()[k]).map_err(|_| "outage Ybus slot exceeds i32")?;
                y_re[offset] = patch.values()[k].re;
                y_im[offset] = patch.values()[k].im;
            }
        }
        let outages = FfiOutageBatch {
            struct_size: mem::size_of::<FfiOutageBatch>() as u32,
            patch_counts: counts.as_ptr(),
            slots: slots.as_ptr(),
            y_re: y_re.as_ptr(),
            y_im: y_im.as_ptr(),
        };
        let options = FfiSolveOptions {
            struct_size: mem::size_of::<FfiSolveOptions>() as u32,
            max_iterations: i32::try_from(max_iter).map_err(|_| "max_iter exceeds i32")?,
            tolerance,
        };
        let count = self
            .batch
            .checked_mul(self.nbus)
            .ok_or("CUDA output size overflow")?;
        let mut status = vec![0; self.batch];
        let mut iterations = vec![0; self.batch];
        let mut residuals = vec![0.0; self.batch];
        let mut voltage_re = vec![0.0; count];
        let mut voltage_im = vec![0.0; count];
        let code = unsafe {
            rp_cuda_solve_outage_batch(
                self.handle.as_ptr(),
                self.batch as i32,
                &outages,
                &options,
                status.as_mut_ptr(),
                iterations.as_mut_ptr(),
                residuals.as_mut_ptr(),
                voltage_re.as_mut_ptr(),
                voltage_im.as_mut_ptr(),
            )
        };
        if code != SUCCESS {
            return Err(format!(
                "CUDA outage batch failed ({code}): {}",
                last_error(Some(self.handle))
            ));
        }
        let cases = (0..self.batch)
            .map(|scenario| {
                let start = scenario * self.nbus;
                CudaCaseSolution {
                    status: CudaCaseStatus::from_raw(status[scenario]),
                    iterations: iterations[scenario] as usize,
                    residual: residuals[scenario],
                    voltage: (0..self.nbus)
                        .map(|k| Complex64::new(voltage_re[start + k], voltage_im[start + k]))
                        .collect(),
                }
            })
            .collect();
        Ok(CudaBatchResult { cases })
    }

    /// Postprocess the last device-side solutions without another voltage upload.
    pub fn postprocess_branches(
        &mut self,
        catalog: &CudaBranchCatalog,
        outages: &[Option<usize>],
    ) -> Result<CudaBranchResults, String> {
        if outages.len() != self.batch {
            return Err("CUDA outage count must equal executor batch".into());
        }
        if catalog.nbus != self.nbus {
            return Err("CUDA branch catalog and executor bus counts differ".into());
        }
        let outage_indices: Vec<i32> = outages
            .iter()
            .map(|outage| match outage {
                None => Ok(-1),
                Some(index) if *index < catalog.branch_count => {
                    i32::try_from(*index).map_err(|_| "CUDA outage index exceeds i32".to_string())
                }
                Some(_) => Err("CUDA branch outage index is out of bounds".to_string()),
            })
            .collect::<Result<_, String>>()?;
        let count = self
            .batch
            .checked_mul(catalog.branch_count)
            .and_then(|value| value.checked_mul(14))
            .ok_or("CUDA branch output size overflow")?;
        let mut raw = vec![0.0; count];
        let mut stats = FfiPostprocessStats {
            struct_size: mem::size_of::<FfiPostprocessStats>() as u32,
            kernel_milliseconds: 0.0,
            download_milliseconds: 0.0,
        };
        let code = unsafe {
            rp_cuda_postprocess_branches(
                self.handle.as_ptr(),
                catalog.handle.as_ptr(),
                outage_indices.as_ptr(),
                raw.as_mut_ptr(),
                &mut stats,
            )
        };
        if code != SUCCESS {
            return Err(format!(
                "CUDA branch postprocessing failed ({code}): {}",
                last_error(Some(self.handle))
            ));
        }
        let flows = raw
            .chunks_exact(14)
            .map(|v| BranchFlowResult {
                p_from_mw: v[0],
                q_from_mvar: v[1],
                p_to_mw: v[2],
                q_to_mvar: v[3],
                pl_mw: v[4],
                ql_mvar: v[5],
                i_from_ka: v[6],
                i_to_ka: v[7],
                i_ka: v[8],
                vm_from_pu: v[9],
                vm_to_pu: v[10],
                va_from_degree: v[11],
                va_to_degree: v[12],
                loading_percent: v[13],
            })
            .collect();
        Ok(CudaBranchResults {
            flows,
            branch_count: catalog.branch_count,
            kernel_seconds: stats.kernel_milliseconds / 1000.0,
            download_seconds: stats.download_milliseconds / 1000.0,
        })
    }
}
impl Drop for CudaExecutor {
    fn drop(&mut self) {
        unsafe { rp_cuda_destroy(self.handle.as_ptr()) }
    }
}

/// CUDA-only execution settings.
///
/// Each stream owns an independent fixed-batch executor on device 0. CUDA studies
/// currently require [`N1Initialization::BaseAc`](super::N1Initialization::BaseAc).
#[derive(Clone, Copy, Debug)]
pub struct CudaN1Config {
    /// Scenarios uploaded and solved per device batch.
    pub batch_size: usize,
    /// Number of independent CUDA streams and executors.
    pub streams: usize,
    pub max_iter: usize,
    pub tolerance: f64,
    pub reporting: N1Reporting,
}
impl Default for CudaN1Config {
    fn default() -> Self {
        Self {
            batch_size: 64,
            streams: 1,
            max_iter: 30,
            tolerance: 1e-10,
            reporting: N1Reporting::StatusOnly,
        }
    }
}

/// CUDA receives tasks first; only a full CUDA queue overflows into the CPU queue.
/// Either backend can be disabled by setting its worker/stream count to zero.
#[derive(Clone, Copy, Debug)]
pub struct HybridN1Config {
    pub cpu_workers: usize,
    pub cpu_batch_size: usize,
    pub cuda_batch_size: usize,
    pub cuda_streams: usize,
    /// Number of CUDA batches retained per stream before tasks overflow to the CPU.
    pub cuda_queue_batches: usize,
    pub max_iter: usize,
    pub tolerance: f64,
    pub reporting: N1Reporting,
}
impl Default for HybridN1Config {
    fn default() -> Self {
        Self {
            cpu_workers: 1,
            cpu_batch_size: 8,
            cuda_batch_size: 64,
            cuda_streams: 1,
            cuda_queue_batches: 2,
            max_iter: 30,
            tolerance: 1e-10,
            reporting: N1Reporting::StatusOnly,
        }
    }
}
impl HybridN1Config {
    fn cpu(self) -> N1Config {
        N1Config {
            workers: self.cpu_workers.max(1),
            batch_size: self.cpu_batch_size,
            max_iter: self.max_iter,
            tolerance: self.tolerance,
            reporting: self.reporting,
            ..Default::default()
        }
    }
    fn cuda(self) -> CudaN1Config {
        CudaN1Config {
            batch_size: self.cuda_batch_size,
            streams: self.cuda_streams,
            max_iter: self.max_iter,
            tolerance: self.tolerance,
            reporting: self.reporting,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct HybridN1Summary {
    pub summary: N1Summary,
    pub cpu: N1Summary,
    pub cuda: N1Summary,
}

/// Entity-oriented hybrid entry point with exactly the same island screening as
/// the CPU-only and CUDA-only study runners.
pub fn run_n1_study_hybrid<F>(
    study: &N1Study,
    faults: &N1FaultSet,
    config: HybridN1Config,
    collect: F,
) -> Result<N1StudyReport, N1StudyError>
where
    F: FnMut(N1EventBatch) + Send,
{
    let (candidates, islands) = study.partition(faults)?;
    let solve = run_n1_hybrid(study.prepared(), config, candidates, collect)
        .map_err(N1StudyError::Execution)?
        .summary;
    Ok(N1StudyReport { solve, islands })
}

/// Entity-oriented entry point: both executors share the study's island screening.
pub fn run_n1_study_cuda<F>(
    study: &N1Study,
    faults: &N1FaultSet,
    config: CudaN1Config,
    collect: F,
) -> Result<N1StudyReport, N1StudyError>
where
    F: FnMut(N1EventBatch) + Send,
{
    let (candidates, islands) = study.partition(faults)?;
    let solve = run_n1_cuda(study.prepared(), config, candidates, collect)
        .map_err(N1StudyError::Execution)?;
    Ok(N1StudyReport { solve, islands })
}

/// Full single-branch N-1 sweep. Extra streams own independent executors and
/// consume contiguous partitions; single-stream execution keeps the original path.
pub fn run_n1_cuda<I, F>(
    prepared: &PreparedN1,
    config: CudaN1Config,
    candidates: I,
    mut collect: F,
) -> Result<N1Summary, String>
where
    I: IntoIterator<Item = usize>,
    F: FnMut(N1EventBatch) + Send,
{
    if config.streams == 0 {
        return Err("CUDA N-1 stream count must be positive".into());
    }
    if config.streams == 1 {
        return run_n1_cuda_single(prepared, config, candidates, collect);
    }
    let candidates: Vec<_> = candidates.into_iter().collect();
    if candidates.is_empty() {
        return run_n1_cuda_single(prepared, config, candidates, collect);
    }
    let stream_count = config.streams.min(candidates.len());
    let chunk_size = candidates.len().div_ceil(stream_count);
    let (result_tx, result_rx) = crossbeam_channel::bounded(stream_count * 2);
    std::thread::scope(|scope| {
        let collector = std::thread::Builder::new()
            .name("n1-cuda-collector".into())
            .spawn_scoped(scope, move || {
                while let Ok(batch) = result_rx.recv() {
                    collect(batch);
                }
            })
            .map_err(|e| format!("start CUDA N-1 collector: {e}"))?;
        let mut handles = Vec::new();
        let mut error = None;
        for (worker_id, chunk) in candidates.chunks(chunk_size).enumerate() {
            let tx = result_tx.clone();
            let worker_config = CudaN1Config {
                streams: 1,
                ..config
            };
            match std::thread::Builder::new()
                .name(format!("n1-cuda-worker-{worker_id}"))
                .spawn_scoped(scope, move || {
                    let mut disconnected = false;
                    let summary = run_n1_cuda_single(
                        prepared,
                        worker_config,
                        chunk.iter().copied(),
                        |batch| {
                            let events = batch
                                .events
                                .into_iter()
                                .map(|mut event| {
                                    event.worker_id = worker_id;
                                    event
                                })
                                .collect();
                            disconnected |= tx.send(N1EventBatch { events }).is_err();
                        },
                    )?;
                    if disconnected {
                        Err("CUDA N-1 collector disconnected".into())
                    } else {
                        Ok(summary)
                    }
                }) {
                Ok(handle) => handles.push(handle),
                Err(e) => {
                    error = Some(format!("start CUDA N-1 worker: {e}"));
                    break;
                }
            }
        }
        drop(result_tx);
        let mut summary = N1Summary::default();
        for handle in handles {
            match handle.join() {
                Ok(Ok(part)) => summary.add(part),
                Ok(Err(e)) => error = Some(e),
                Err(_) => error = Some("CUDA N-1 worker panicked".into()),
            }
        }
        if collector.join().is_err() {
            error = Some("CUDA N-1 collector panicked".into());
        }
        if let Some(error) = error {
            Err(error)
        } else {
            Ok(summary)
        }
    })
}

/// Keep the CUDA queue full before overflowing work into the CPU queue. This gives
/// batch creation priority over CPU prefetching without statically partitioning cases.
pub fn run_n1_hybrid<I, F>(
    prepared: &PreparedN1,
    config: HybridN1Config,
    candidates: I,
    mut collect: F,
) -> Result<HybridN1Summary, String>
where
    I: IntoIterator<Item = usize>,
    F: FnMut(N1EventBatch) + Send,
{
    if config.cpu_workers == 0 && config.cuda_streams == 0 {
        return Err("hybrid N-1 requires at least one CPU worker or CUDA stream".into());
    }
    if prepared.initialization() != N1Initialization::BaseAc {
        return Err("hybrid N-1 currently supports BaseAc initialization only".into());
    }
    if config.cpu_workers > 0 {
        config.cpu().validate(prepared)?;
    }
    if config.cuda_streams > 0 {
        validate_cuda_n1(prepared, config.cuda())?;
        if config.cuda_queue_batches == 0 {
            return Err("hybrid CUDA queue depth must be positive".into());
        }
    }
    let cpu_depth = config
        .cpu_workers
        .checked_mul(config.cpu_batch_size)
        .and_then(|depth| depth.max(1).checked_mul(2))
        .ok_or("hybrid CPU queue size overflow")?;
    let cuda_depth = config
        .cuda_streams
        .checked_mul(config.cuda_batch_size)
        .and_then(|depth| depth.checked_mul(config.cuda_queue_batches))
        .map(|depth| depth.max(1))
        .ok_or("hybrid CUDA queue size overflow")?;
    let (cpu_tx, cpu_rx) = crossbeam_channel::bounded::<OutageTask>(cpu_depth);
    let (cuda_tx, cuda_rx) = crossbeam_channel::bounded::<OutageTask>(cuda_depth);
    let (result_tx, result_rx) = crossbeam_channel::bounded::<N1EventBatch>(
        (config.cpu_workers + config.cuda_streams).max(1) * 2,
    );
    std::thread::scope(|scope| {
        let collector = std::thread::Builder::new()
            .name("n1-hybrid-collector".into())
            .spawn_scoped(scope, move || {
                while let Ok(batch) = result_rx.recv() {
                    collect(batch);
                }
            })
            .map_err(|e| format!("start hybrid N-1 collector: {e}"))?;
        let mut cpu_handles = Vec::new();
        let mut cuda_handles = Vec::new();
        let mut error = None;
        for id in 0..config.cpu_workers {
            let rx = cpu_rx.clone();
            let tx = result_tx.clone();
            match std::thread::Builder::new()
                .name(format!("n1-hybrid-cpu-{id}"))
                .spawn_scoped(scope, move || {
                    worker_shared(prepared, config.cpu(), N1Initialization::BaseAc, id, rx, tx)
                }) {
                Ok(handle) => cpu_handles.push(handle),
                Err(e) => {
                    error = Some(format!("start hybrid CPU worker: {e}"));
                    break;
                }
            }
        }
        if error.is_none() {
            for stream in 0..config.cuda_streams {
                let rx = cuda_rx.clone();
                let tx = result_tx.clone();
                match std::thread::Builder::new()
                    .name(format!("n1-hybrid-cuda-{stream}"))
                    .spawn_scoped(scope, move || {
                        run_n1_cuda_shared(
                            prepared,
                            config.cuda(),
                            config.cpu_workers + stream,
                            rx,
                            tx,
                        )
                    }) {
                    Ok(handle) => cuda_handles.push(handle),
                    Err(e) => {
                        error = Some(format!("start hybrid CUDA worker: {e}"));
                        break;
                    }
                }
            }
        }
        drop(cpu_rx);
        drop(cuda_rx);
        drop(result_tx);
        if error.is_none() {
            for (case_id, branch_index) in candidates.into_iter().enumerate() {
                let task = match OutageTask::for_ac_branch(
                    prepared.base().baseline(),
                    prepared.base().catalog(),
                    branch_index,
                    case_id,
                ) {
                    Ok(task) => task,
                    Err(e) => {
                        error = Some(e);
                        break;
                    }
                };
                let routed = if config.cuda_streams > 0 && config.cpu_workers > 0 {
                    match cuda_tx.try_send(task) {
                        Ok(()) => Ok(()),
                        Err(crossbeam_channel::TrySendError::Full(task))
                        | Err(crossbeam_channel::TrySendError::Disconnected(task)) => {
                            cpu_tx.send(task)
                        }
                    }
                } else if config.cuda_streams > 0 {
                    cuda_tx.send(task)
                } else {
                    cpu_tx.send(task)
                };
                if routed.is_err() {
                    error = Some("hybrid N-1 workers stopped before submission completed".into());
                    break;
                }
            }
        }
        drop(cpu_tx);
        drop(cuda_tx);
        let mut combined = HybridN1Summary::default();
        for handle in cpu_handles {
            match handle.join() {
                Ok(Ok(part)) => {
                    combined.summary.add(part);
                    combined.cpu.add(part);
                }
                Ok(Err(e)) => error = Some(e),
                Err(_) => error = Some("hybrid CPU worker panicked".into()),
            }
        }
        for handle in cuda_handles {
            match handle.join() {
                Ok(Ok(part)) => {
                    combined.summary.add(part);
                    combined.cuda.add(part);
                }
                Ok(Err(e)) => error = Some(e),
                Err(_) => error = Some("hybrid CUDA worker panicked".into()),
            }
        }
        if collector.join().is_err() {
            error = Some("hybrid N-1 collector panicked".into());
        }
        if let Some(error) = error {
            Err(error)
        } else {
            Ok(combined)
        }
    })
}

fn validate_cuda_n1(prepared: &PreparedN1, config: CudaN1Config) -> Result<(), String> {
    if prepared.initialization() != N1Initialization::BaseAc {
        return Err("CUDA N-1 currently supports BaseAc initialization only".into());
    }
    if config.batch_size == 0 || !config.tolerance.is_finite() || config.tolerance <= 0.0 {
        return Err("CUDA N-1 batch must be positive and tolerance finite and positive".into());
    }
    Ok(())
}

struct CudaN1Runner<'a> {
    prepared: &'a PreparedN1,
    config: CudaN1Config,
    executor: CudaExecutor,
    worker_id: usize,
    summary: N1Summary,
}
impl<'a> CudaN1Runner<'a> {
    fn new(
        prepared: &'a PreparedN1,
        config: CudaN1Config,
        worker_id: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            prepared,
            config,
            executor: CudaExecutor::new_ac(prepared.base().baseline(), config.batch_size)?,
            worker_id,
            summary: N1Summary {
                workers_started: 1,
                ..Default::default()
            },
        })
    }

    fn execute(&mut self, mut tasks: Vec<OutageTask>) -> Result<Option<N1EventBatch>, String> {
        let prepared = self.prepared;
        let config = self.config;
        let catalog = prepared.base().catalog();
        let actual = tasks.len();
        self.summary.submitted += actual;
        while tasks.len() < config.batch_size {
            tasks.push(tasks[0]);
        }
        let solve_started = Instant::now();
        let solved = self
            .executor
            .solve_outages(&tasks, config.max_iter, config.tolerance)?;
        self.summary.ac_time += solve_started.elapsed();
        let mut events = Vec::new();
        for scenario in 0..actual {
            let task = tasks[scenario];
            let case = &solved.cases[scenario];
            let converged = case.status == CudaCaseStatus::Converged;
            self.summary.processed += 1;
            self.summary.total_iterations += case.iterations;
            self.summary.attempted_iterations += case.iterations;
            self.summary.failed += usize::from(!converged);
            self.summary.newton_failed += usize::from(!converged);
            let watch_started = Instant::now();
            let (hits, violated) = match config.reporting {
                N1Reporting::Detailed => {
                    let hits = prepared.watch().evaluate(
                        catalog,
                        &case.voltage,
                        Some(task.branch_index),
                        converged,
                    )?;
                    let violated = hits.violation_count() > 0;
                    (hits, violated)
                }
                N1Reporting::StatusOnly if converged => {
                    let violated = prepared.watch().is_violated(
                        catalog,
                        &case.voltage,
                        Some(task.branch_index),
                    )?;
                    (WatchHits::default(), violated)
                }
                N1Reporting::StatusOnly => (WatchHits::default(), false),
            };
            self.summary.watch_time += watch_started.elapsed();
            self.summary.violated_cases += usize::from(violated);
            self.summary.violation_hits += hits.violation_count();
            let outcome = if !converged {
                N1Outcome::NumericalFailure
            } else if violated {
                N1Outcome::ConstraintViolation
            } else {
                N1Outcome::Passed
            };
            let emit = match config.reporting {
                N1Reporting::Detailed => true,
                N1Reporting::StatusOnly => outcome != N1Outcome::Passed,
            };
            if emit {
                self.summary.emitted_cases += 1;
                self.summary.emitted_hits += hits.len();
                events.push(N1Event {
                    case_id: task.case_id,
                    outage_index: task.branch_index,
                    outage: catalog.identities()[task.branch_index],
                    outage_entity: catalog.source_entities()[task.branch_index],
                    worker_id: self.worker_id,
                    stage: if converged {
                        SolveStage::Converged
                    } else {
                        SolveStage::NewtonFailed
                    },
                    outcome,
                    iterations: case.iterations,
                    residual: case.residual,
                    error: (!converged).then(|| format!("CUDA Newton status: {:?}", case.status)),
                    hits,
                    linear: LinearSolveStats::default(),
                });
            }
        }
        Ok((!events.is_empty()).then_some(N1EventBatch { events }))
    }
}

fn run_n1_cuda_single<I, F>(
    prepared: &PreparedN1,
    config: CudaN1Config,
    candidates: I,
    mut collect: F,
) -> Result<N1Summary, String>
where
    I: IntoIterator<Item = usize>,
    F: FnMut(N1EventBatch),
{
    validate_cuda_n1(prepared, config)?;
    let base = prepared.base().baseline();
    let catalog = prepared.base().catalog();
    let mut runner = CudaN1Runner::new(prepared, config, 0)?;
    let mut input = candidates.into_iter();
    let mut case_id = 0_usize;
    loop {
        let mut tasks = Vec::with_capacity(config.batch_size);
        while tasks.len() < config.batch_size {
            let Some(branch_index) = input.next() else {
                break;
            };
            tasks.push(OutageTask::for_ac_branch(
                base,
                catalog,
                branch_index,
                case_id,
            )?);
            case_id += 1;
        }
        if tasks.is_empty() {
            break;
        }
        if let Some(events) = runner.execute(tasks)? {
            collect(events);
        }
    }
    Ok(runner.summary)
}

fn run_n1_cuda_shared(
    prepared: &PreparedN1,
    config: CudaN1Config,
    worker_id: usize,
    rx: crossbeam_channel::Receiver<OutageTask>,
    tx: crossbeam_channel::Sender<N1EventBatch>,
) -> Result<N1Summary, String> {
    let mut runner = CudaN1Runner::new(prepared, config, worker_id)?;
    loop {
        let mut tasks = Vec::with_capacity(config.batch_size);
        match rx.recv() {
            Ok(task) => tasks.push(task),
            Err(_) => break,
        }
        while tasks.len() < config.batch_size {
            match rx.try_recv() {
                Ok(task) => tasks.push(task),
                Err(
                    crossbeam_channel::TryRecvError::Empty
                    | crossbeam_channel::TryRecvError::Disconnected,
                ) => break,
            }
        }
        if let Some(events) = runner.execute(tasks)? {
            if tx.send(events).is_err() {
                return Err("N-1 collector disconnected".into());
            }
        }
    }
    Ok(runner.summary)
}

fn to_i32(values: &[usize], name: &str) -> Result<Vec<i32>, String> {
    values
        .iter()
        .map(|&value| i32::try_from(value).map_err(|_| format!("{name} exceed i32")))
        .collect()
}

fn split_complex(values: &[Complex64]) -> (Vec<f64>, Vec<f64>) {
    values.iter().map(|z| (z.re, z.im)).unzip()
}

fn mirror_positions(y_col_ptr: &[i32], y_row_indices: &[i32]) -> Result<Vec<i32>, String> {
    let nbus = y_col_ptr.len() - 1;
    let mut mirror = vec![0; y_row_indices.len()];
    for k in 0..nbus {
        for p in y_col_ptr[k] as usize..y_col_ptr[k + 1] as usize {
            let i = y_row_indices[p] as usize;
            if i >= nbus {
                return Err("Ybus row index exceeds bus count".into());
            }
            let start = y_col_ptr[i] as usize;
            let rows = &y_row_indices[start..y_col_ptr[i + 1] as usize];
            let offset = rows
                .binary_search(&(k as i32))
                .map_err(|_| "Ybus pattern is not structurally symmetric")?;
            mirror[p] = y_col_ptr[i] + offset as i32;
        }
    }
    Ok(mirror)
}

fn last_error(handle: Option<NonNull<FfiExecutor>>) -> String {
    if let Err(error) = cuda_api() {
        return error;
    }
    let pointer = unsafe { rp_cuda_last_error(handle.map_or(ptr::null(), |h| h.as_ptr())) };
    if pointer.is_null() {
        return "no diagnostic".into();
    }
    unsafe { CStr::from_ptr(pointer) }
        .to_string_lossy()
        .into_owned()
}
