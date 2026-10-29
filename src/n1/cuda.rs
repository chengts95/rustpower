//! Experimental FP64 CUDA executor. One instance owns one stream and one cuDSS queue.
//! Multiple instances are the intended multi-stream experiment; no global device state.
use super::SharedBaseline;
use num_complex::Complex64;
use std::{
    cell::Cell,
    ffi::{CStr, c_char},
    marker::PhantomData,
    mem,
    ptr::{self, NonNull},
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

unsafe extern "C" {
    fn rp_cuda_create(
        baseline: *const FfiBaseline,
        batch: i32,
        executor: *mut *mut FfiExecutor,
    ) -> i32;
    fn rp_cuda_solve_base_batch(
        executor: *mut FfiExecutor,
        batch: i32,
        options: *const FfiSolveOptions,
        case_status: *mut i32,
        iterations: *mut i32,
        residuals: *mut f64,
        voltage_re: *mut f64,
        voltage_im: *mut f64,
    ) -> i32;
    fn rp_cuda_last_error(executor: *const FfiExecutor) -> *const c_char;
    fn rp_cuda_destroy(executor: *mut FfiExecutor);
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
        let (initial_v_re, initial_v_im) = split_complex(base.v_init());
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
}
impl Drop for CudaExecutor {
    fn drop(&mut self) {
        unsafe { rp_cuda_destroy(self.handle.as_ptr()) }
    }
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
    let pointer = unsafe { rp_cuda_last_error(handle.map_or(ptr::null(), |h| h.as_ptr())) };
    if pointer.is_null() {
        return "no diagnostic".into();
    }
    unsafe { CStr::from_ptr(pointer) }
        .to_string_lossy()
        .into_owned()
}
