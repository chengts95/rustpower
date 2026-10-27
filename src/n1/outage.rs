//! Main-thread compilation of a single branch outage into fixed numerical slots.
//! Worker application touches only this task's and the preceding task's slots.
use super::{
    postprocess::BranchCatalog,
    solver::{N1Worker, SharedBaseline},
};
use num_complex::Complex64;
use std::ops::AddAssign;

#[derive(Clone, Copy, Debug)]
pub struct ValuePatch<T, const N: usize> {
    len: usize,
    slots: [usize; N],
    values: [T; N],
}
impl<T: Copy + Default + AddAssign, const N: usize> ValuePatch<T, N> {
    fn empty() -> Self {
        Self {
            len: 0,
            slots: [0; N],
            values: [T::default(); N],
        }
    }
    fn add(&mut self, slot: usize, delta: T) -> Result<(), String> {
        if let Some(k) = self.slots[..self.len].iter().position(|&s| s == slot) {
            self.values[k] += delta;
            return Ok(());
        }
        if self.len == N {
            return Err("outage exceeds its fixed numerical stencil".into());
        }
        self.slots[self.len] = slot;
        self.values[self.len] = delta;
        self.len += 1;
        Ok(())
    }
    fn absolute(mut self, base: &[T]) -> Result<Self, String> {
        for k in 0..self.len {
            let mut value = *base
                .get(self.slots[k])
                .ok_or("outage slot is outside baseline values")?;
            value += self.values[k];
            self.values[k] = value;
        }
        Ok(self)
    }
}
impl<T, const N: usize> ValuePatch<T, N> {
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn slots(&self) -> &[usize] {
        &self.slots[..self.len]
    }
    pub fn values(&self) -> &[T] {
        &self.values[..self.len]
    }
    fn valid_for(&self, n: usize) -> bool {
        self.len <= N && self.slots[..self.len].iter().all(|&slot| slot < n)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BaselineKey(u64);
impl BaselineKey {
    fn of(base: &SharedBaseline) -> Self {
        Self(base.generation())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct OutageTask {
    pub case_id: usize,
    pub branch_index: usize,
    y_patch: ValuePatch<Complex64, 4>,
    dc_patch: ValuePatch<f64, 4>,
    dc_shift_patch: ValuePatch<f64, 2>,
    baseline: BaselineKey,
    includes_dc: bool,
}
fn slot(offsets: &[usize], rows: &[usize], row: usize, col: usize) -> Result<usize, String> {
    let start = *offsets
        .get(col)
        .ok_or("outage column is outside fixed pattern")?;
    let end = *offsets
        .get(col + 1)
        .ok_or("outage column is outside fixed pattern")?;
    rows[start..end]
        .binary_search(&row)
        .map(|k| start + k)
        .map_err(|_| format!("outage coordinate ({row},{col}) is absent from the fixed pattern"))
}
fn y_stencil(
    base: &SharedBaseline,
    catalog: &BranchCatalog,
    branch_index: usize,
) -> Result<(usize, usize, ValuePatch<Complex64, 4>, [(usize, usize); 4]), String> {
    if branch_index >= catalog.len() {
        return Err("outage branch index is out of bounds".into());
    }
    if !catalog.in_service()[branch_index] {
        return Err("N-1 candidates must be in service in the baseline".into());
    }
    let (f, t) = (
        catalog.from_buses()[branch_index],
        catalog.to_buses()[branch_index],
    );
    let [y_ff, y_tf, y_ft, y_tt] = catalog.admittances();
    let y = base.y_bus();
    if f >= y.nrows() || t >= y.nrows() {
        return Err("outage branch endpoints exceed baseline dimensions".into());
    }
    let coordinates = [
        (f, f, y_ff[branch_index]),
        (f, t, y_ft[branch_index]),
        (t, f, y_tf[branch_index]),
        (t, t, y_tt[branch_index]),
    ];
    let mut y_delta = ValuePatch::<Complex64, 4>::empty();
    // Keep the unique coordinate alongside each unique final CSC slot.
    let mut unique = [(0usize, 0usize); 4];
    for (row, col, contribution) in coordinates {
        let at = slot(y.col_offsets(), y.row_indices(), row, col)?;
        let index = y_delta
            .slots()
            .iter()
            .position(|&s| s == at)
            .unwrap_or(y_delta.len());
        y_delta.add(at, -contribution)?;
        unique[index] = (row, col);
    }
    Ok((f, t, y_delta, unique))
}
impl OutageTask {
    /// Compile absolute values from the shared baseline, never from another case.
    pub fn for_branch(
        base: &SharedBaseline,
        catalog: &BranchCatalog,
        branch_index: usize,
        case_id: usize,
    ) -> Result<Self, String> {
        let (f, t, y_delta, unique) = y_stencil(base, catalog, branch_index)?;
        let y = base.y_bus();
        let active = base.npq() + base.npv();
        let dc = base.dc_model();
        let mut b_delta = ValuePatch::<f64, 4>::empty();
        let mut shift_delta = ValuePatch::<f64, 2>::empty();
        for k in 0..y_delta.len() {
            let (row, col) = unique[k];
            if row == col || col >= active {
                continue;
            }
            let old = y.values()[y_delta.slots[k]];
            let new = old + y_delta.values[k];
            // DCPF is built from abs(aggregated Y), including parallel branches.
            let delta = new.norm() - old.norm();
            let diagonal = slot(&dc.col_ptrs, &dc.row_indices, col, col)?;
            b_delta.add(diagonal, delta)?;
            if row < active {
                let off_diagonal = slot(&dc.col_ptrs, &dc.row_indices, row, col)?;
                b_delta.add(off_diagonal, -delta)?;
            } else {
                shift_delta.add(col, -delta * base.v_init()[row].arg())?;
            }
        }
        let phase = *catalog
            .phase_injections()
            .get(branch_index)
            .ok_or("outage phase metadata is incomplete")?;
        if f < active {
            shift_delta.add(f, -phase)?;
        }
        if t < active {
            shift_delta.add(t, phase)?;
        }
        Ok(Self {
            case_id,
            branch_index,
            y_patch: y_delta.absolute(y.values())?,
            dc_patch: b_delta.absolute(&dc.values)?,
            dc_shift_patch: shift_delta.absolute(dc.p_shift.as_slice())?,
            baseline: BaselineKey::of(base),
            includes_dc: true,
        })
    }
    /// Compile only the AC admittance stencil. No DC arrays are read or patched.
    pub fn for_ac_branch(
        base: &SharedBaseline,
        catalog: &BranchCatalog,
        branch_index: usize,
        case_id: usize,
    ) -> Result<Self, String> {
        let (_, _, delta, _) = y_stencil(base, catalog, branch_index)?;
        Ok(Self {
            case_id,
            branch_index,
            y_patch: delta.absolute(base.y_bus().values())?,
            dc_patch: ValuePatch::empty(),
            dc_shift_patch: ValuePatch::empty(),
            baseline: BaselineKey::of(base),
            includes_dc: false,
        })
    }
    pub fn y_patch(&self) -> &ValuePatch<Complex64, 4> {
        &self.y_patch
    }
    pub fn dc_patch(&self) -> &ValuePatch<f64, 4> {
        &self.dc_patch
    }
    pub fn dc_shift_patch(&self) -> &ValuePatch<f64, 2> {
        &self.dc_shift_patch
    }
}

/// Stores one small preceding stencil, not a matrix clone or an undo-value log.
#[derive(Default)]
pub struct WorkerPatchState {
    previous: Option<OutageTask>,
}
fn restore<T: Copy, const N: usize>(patch: &ValuePatch<T, N>, baseline: &[T], current: &mut [T]) {
    for &slot in patch.slots() {
        current[slot] = baseline[slot];
    }
}
fn write<T: Copy, const N: usize>(patch: &ValuePatch<T, N>, current: &mut [T]) {
    for (&slot, &value) in patch.slots().iter().zip(patch.values()) {
        current[slot] = value;
    }
}
impl WorkerPatchState {
    fn validate(
        &self,
        worker: &N1Worker,
        base: &SharedBaseline,
        task: Option<&OutageTask>,
    ) -> Result<(), String> {
        let key = BaselineKey::of(base);
        if worker.baseline_generation() != base.generation() {
            return Err("worker belongs to a different fixed baseline".into());
        }
        if task.is_some_and(|task| task.baseline != key)
            || self
                .previous
                .as_ref()
                .is_some_and(|task| task.baseline != key)
        {
            return Err(
                "outage tasks and worker patch state must use their original baseline".into(),
            );
        }
        let uses_dc = worker.uses_dc();
        if task.is_some_and(|task| task.includes_dc != uses_dc)
            || self
                .previous
                .as_ref()
                .is_some_and(|task| task.includes_dc != uses_dc)
        {
            return Err("outage task and worker computation modes differ".into());
        }
        if worker.y_values.len() != base.y_bus().values().len()
            || (uses_dc
                && (worker.dc_values.len() != base.dc_model().values.len()
                    || worker.dc_shift.len() != base.dc_model().p_shift.len()))
            || (!uses_dc && (!worker.dc_values.is_empty() || !worker.dc_shift.is_empty()))
        {
            return Err("worker values do not match the fixed baseline dimensions".into());
        }
        for task in task.into_iter().chain(self.previous.as_ref()) {
            if !task.y_patch.valid_for(worker.y_values.len())
                || !task.dc_patch.valid_for(worker.dc_values.len())
                || !task.dc_shift_patch.valid_for(worker.dc_shift.len())
            {
                return Err("outage task contains an invalid numerical slot".into());
            }
        }
        Ok(())
    }
    pub fn apply(
        &mut self,
        worker: &mut N1Worker,
        base: &SharedBaseline,
        task: &OutageTask,
    ) -> Result<(), String> {
        self.validate(worker, base, Some(task))?;
        self.restore_previous(worker, base);
        write(&task.y_patch, &mut worker.y_values);
        write(&task.dc_patch, &mut worker.dc_values);
        write(&task.dc_shift_patch, &mut worker.dc_shift);
        self.previous = Some(*task);
        Ok(())
    }
    /// Return only the touched values to the shared baseline, for base queries.
    pub fn clear(&mut self, worker: &mut N1Worker, base: &SharedBaseline) -> Result<(), String> {
        self.validate(worker, base, None)?;
        self.restore_previous(worker, base);
        Ok(())
    }
    fn restore_previous(&mut self, worker: &mut N1Worker, base: &SharedBaseline) {
        if let Some(previous) = self.previous.take() {
            restore(
                &previous.y_patch,
                base.y_bus().values(),
                &mut worker.y_values,
            );
            if previous.includes_dc {
                restore(
                    &previous.dc_patch,
                    &base.dc_model().values,
                    &mut worker.dc_values,
                );
                restore(
                    &previous.dc_shift_patch,
                    base.dc_model().p_shift.as_slice(),
                    &mut worker.dc_shift,
                );
            }
        }
    }
}
