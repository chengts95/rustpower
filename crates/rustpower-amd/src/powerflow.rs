//! Power-flow-specific symbolic structure and ordering, independent of a solver backend.
//!
//! One immutable Ybus analysis feeds both reduced-J CSC generation and AMD's
//! mutable quotient graph. Solver traits and KLU allocation are not part of this API.
use crate::{amd_2::amd_2, Control, Info, Status};

/// Boundaries in one sorted Ybus column. Counts are relative to its start;
/// diagonal_position is an absolute index into the original Ybus row array.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColumnLayout {
    pub active_count: usize,
    pub pq_count: usize,
    pub diagonal_position: usize,
}

/// Validated `[PQ | PV | Slack]` structure. Borrows Ybus immutably so its
/// analysis cannot silently outlive a mutation of the source arrays.
#[derive(Debug)]
pub struct YbusPattern<'a> {
    offsets: &'a [usize],
    rows: &'a [usize],
    npq: usize,
    active: usize,
    columns: Vec<ColumnLayout>,
    scalar_nnz: usize,
    quotient_entries: usize,
}
impl<'a> YbusPattern<'a> {
    /// Validate once and extract shared column metadata. Currently supports the
    /// same symmetric, diagonal-present, connected active structure as seeded AMD.
    pub fn analyze(
        offsets: &'a [usize],
        rows: &'a [usize],
        npv: usize,
        npq: usize,
    ) -> Result<Self, Status> {
        if !eligible(offsets, rows, npv, npq) {
            return Err(Status::Invalid);
        }
        let active = npv + npq;
        let mut columns = Vec::with_capacity(active);
        let mut scalar_nnz = 0usize;
        let mut quotient_entries = 0usize;
        for j in 0..active {
            let col = &rows[offsets[j]..offsets[j + 1]];
            let active_count = col.partition_point(|&i| i < active);
            let pq_count = col[..active_count].partition_point(|&i| i < npq);
            let diagonal_position = offsets[j] + col[..active_count].binary_search(&j).unwrap();
            scalar_nnz = scalar_nnz
                .checked_add((active_count + pq_count) * if j < npq { 2 } else { 1 })
                .ok_or(Status::Invalid)?;
            quotient_entries = quotient_entries
                .checked_add(active_count - 1)
                .ok_or(Status::Invalid)?;
            columns.push(ColumnLayout {
                active_count,
                pq_count,
                diagonal_position,
            });
        }
        Ok(Self {
            offsets,
            rows,
            npq,
            active,
            columns,
            scalar_nnz,
            quotient_entries,
        })
    }
    pub fn active_bus_count(&self) -> usize {
        self.active
    }
    pub fn pq_bus_count(&self) -> usize {
        self.npq
    }
    pub fn scalar_dimension(&self) -> usize {
        self.active + self.npq
    }
    pub fn scalar_nnz(&self) -> usize {
        self.scalar_nnz
    }
    pub fn quotient_entries(&self) -> usize {
        self.quotient_entries
    }
    pub fn columns(&self) -> &[ColumnLayout] {
        &self.columns
    }
    pub(crate) fn diagonal_local(&self, bus: usize) -> usize {
        self.columns[bus].diagonal_position - self.offsets[bus]
    }
    pub(crate) fn active_rows(&self, j: usize) -> &[usize] {
        &self.rows[self.offsets[j]..self.offsets[j] + self.columns[j].active_count]
    }
    /// Materialize the numerical J's CSC pattern, reusing the analyzed boundaries.
    /// Pattern: [[B, B[:,PQ]], [B[PQ,:], B[PQ,PQ]]].
    pub fn reduced_jacobian(&self) -> ReducedJacobianPattern {
        let mut column_offsets = Vec::with_capacity(self.scalar_dimension() + 1);
        let mut row_indices = Vec::with_capacity(self.scalar_nnz);
        for j in (0..self.active).chain(0..self.npq) {
            column_offsets.push(row_indices.len());
            let rows = self.active_rows(j);
            row_indices.extend_from_slice(rows);
            row_indices.extend(
                rows[..self.columns[j].pq_count]
                    .iter()
                    .map(|&i| self.active + i),
            );
        }
        column_offsets.push(row_indices.len());
        ReducedJacobianPattern {
            column_offsets,
            row_indices,
        }
    }
}

#[derive(Debug)]
pub struct ReducedJacobianPattern {
    column_offsets: Vec<usize>,
    row_indices: Vec<usize>,
}
impl ReducedJacobianPattern {
    pub fn column_offsets(&self) -> &[usize] {
        &self.column_offsets
    }
    pub fn row_indices(&self) -> &[usize] {
        &self.row_indices
    }
    pub fn into_parts(self) -> (Vec<usize>, Vec<usize>) {
        (self.column_offsets, self.row_indices)
    }
}

#[derive(Debug)]
pub struct Ordering {
    permutation: Vec<usize>,
    inverse: Vec<usize>,
    info: Info,
}
impl Ordering {
    pub fn permutation(&self) -> &[usize] {
        &self.permutation
    }
    pub fn inverse_permutation(&self) -> &[usize] {
        &self.inverse
    }
    pub fn info(&self) -> &Info {
        &self.info
    }
    pub fn into_parts(self) -> (Vec<usize>, Vec<usize>, Info) {
        (self.permutation, self.inverse, self.info)
    }
}

/// Owned mutable graph arena. The elimination kernel changes adjacency entries,
/// compresses lists and uses the free tail to represent fill. No caller handles
/// raw Iw, negative parent encodings or workspace-length arguments.
#[derive(Default)]
struct QuotientGraph {
    offsets: Vec<isize>,
    lengths: Vec<usize>,
    adjacency: Vec<isize>,
    external_degrees: Vec<usize>,
    initial_entries: usize,
}
impl QuotientGraph {
    fn initialize(&mut self, pattern: &YbusPattern<'_>) -> Result<(), Status> {
        let n = pattern.scalar_dimension();
        let entries = pattern.quotient_entries;
        let capacity = entries
            .checked_add(entries / 5)
            .and_then(|x| x.checked_add(n))
            .ok_or(Status::Invalid)?;
        self.offsets.resize(n, 0);
        self.offsets.fill(entries as isize);
        self.lengths.resize(n, 0);
        self.lengths.fill(0);
        self.external_degrees.resize(n, 0);
        self.external_degrees.fill(0);
        self.adjacency.resize(capacity, 0);
        self.initial_entries = entries;
        let mut position = 0;
        for (j, col) in pattern.columns.iter().enumerate() {
            self.offsets[j] = position as isize;
            self.lengths[j] = col.active_count - 1;
            let own_weight = if j < pattern.npq { 2 } else { 1 };
            self.external_degrees[j] = col.active_count + col.pq_count - own_weight;
            let rows = pattern.active_rows(j);
            let diag = col.diagonal_position - pattern.offsets[j];
            // Only the mandatory mutable copy remains; no reclassification,
            // recounting or per-edge diagonal/PQ test is necessary.
            for &i in rows[..diag].iter().chain(&rows[diag + 1..]) {
                self.adjacency[position] = i as isize;
                position += 1;
            }
        }
        debug_assert_eq!(position, entries);
        Ok(())
    }
    fn eliminate<const FAST: bool>(
        &mut self,
        pattern: &YbusPattern<'_>,
        control: &Control,
    ) -> Result<Ordering, Status> {
        self.initialize(pattern)?;
        let n = pattern.scalar_dimension();
        let mut info = Info::new(n);
        info.nz = pattern.scalar_nnz;
        info.nz_diag = n;
        info.symmetry = true;
        info.nz_a_plus_at = info.nz - n;
        let capacity = self.adjacency.len();
        let (_, inverse, permutation, _) = amd_2::<usize, FAST, false>(
            n,
            &mut self.offsets,
            &mut self.adjacency,
            &mut self.lengths,
            capacity,
            self.initial_entries,
            control,
            &mut info,
            Some((pattern.active, pattern.npq, &self.external_degrees)),
            None,
        );
        Ok(Ordering {
            permutation,
            inverse,
            info,
        })
    }
}

/// Reusable storage, not a persistent ordering cache. Each order call resets the
/// mutable graph from the supplied analyzed pattern, including on topology change.
#[derive(Default)]
pub struct OrderingWorkspace {
    graph: QuotientGraph,
}
impl OrderingWorkspace {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn order(
        &mut self,
        pattern: &YbusPattern<'_>,
        control: &Control,
    ) -> Result<Ordering, Status> {
        self.graph.eliminate::<false>(pattern, control)
    }
    /// Same algorithm with unchecked indexing in the internal elimination kernel.
    /// # Safety
    /// Experimental: all internal AMD index invariants must hold. The pattern's
    /// structural validity is established by YbusPattern::analyze. Prefer order
    /// for diagnosing kernel invariant violations.
    pub unsafe fn order_unchecked(
        &mut self,
        pattern: &YbusPattern<'_>,
        control: &Control,
    ) -> Result<Ordering, Status> {
        self.graph.eliminate::<true>(pattern, control)
    }
}

pub(crate) fn eligible(p: &[usize], rows: &[usize], npv: usize, npq: usize) -> bool {
    let Some(active) = npv.checked_add(npq) else {
        return false;
    };
    let Some(n) = active.checked_add(npq) else {
        return false;
    };
    if n <= 3
        || n > isize::MAX as usize / 128
        || rows.len() > isize::MAX as usize / 64
        || p.len() < 2
        || active >= p.len() - 1
        || p[0] != 0
        || p.last() != Some(&rows.len())
    {
        return false;
    }
    if p.windows(2).any(|w| w[0] > w[1]) {
        return false;
    }
    let nodes = p.len() - 1;
    if rows.iter().any(|&r| r >= nodes)
        || p.windows(2)
            .any(|w| rows[w[0]..w[1]].windows(2).any(|a| a[0] >= a[1]))
    {
        return false;
    }
    for j in 0..active {
        let col = &rows[p[j]..p[j + 1]];
        if col.binary_search(&j).is_err() {
            return false;
        }
        for &i in col.iter().take_while(|&&i| i < active) {
            if rows[p[i]..p[i + 1]].binary_search(&j).is_err() {
                return false;
            }
        }
    }
    // Conservative: only optimize the proven single active component; otherwise original BTF handles it.
    let mut visited = vec![false; active];
    let mut todo = Vec::with_capacity(active);
    visited[0] = true;
    todo.push(0);
    let mut cursor = 0;
    while cursor < todo.len() {
        let j = todo[cursor];
        cursor += 1;
        for &i in rows[p[j]..p[j + 1]].iter().take_while(|&&i| i < active) {
            if !visited[i] {
                visited[i] = true;
                todo.push(i);
            }
        }
    }
    todo.len() == active
}
