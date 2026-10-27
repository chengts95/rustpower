//! Experimental scalar AMD which obtains initial neighbors from Ybus on demand.
//! No array of all initial scalar neighbor IDs is built, even inside a workspace.
use crate::{powerflow::YbusPattern, Control, Info, Status};
use std::cell::Cell;
use std::mem::MaybeUninit;
#[derive(Clone, Debug, Default)]
pub struct Counts {
    pub initial_neighbor_reads: usize,
    pub generated_neighbor_ids_saved: usize,
    pub dynamic_writes: usize,
    pub distinct_dynamic_slots: usize,
    pub initial_variables_read: usize,
    pub allocated_value_slots: usize,
}
#[derive(Clone, Copy, Default)]
struct Cursor {
    start: usize,
    end: usize,
    bus: usize,
    diag: usize,
    active: usize,
    pq: usize,
}
/// Initial address ranges are only number ranges, not stored adjacency lists.
/// Modified entries occupy uninitialized continuous storage. Unwritten initial positions
/// are decoded from Ybus; an unwritten dynamic position is an invariant error.
pub(crate) struct NeighborStorage<'a, const COUNT: bool> {
    pattern: &'a YbusPattern<'a>,
    starts: Vec<usize>,
    initial_entries: usize,
    data: Vec<MaybeUninit<isize>>,
    written: Vec<u64>,
    cursor: Cell<Cursor>,
    reads: Cell<usize>,
    touched: Vec<Cell<bool>>,
    writes: usize,
    distinct: usize,
}
impl<'a, const COUNT: bool> NeighborStorage<'a, COUNT> {
    pub(crate) fn initial_list(&self, variable: usize) -> InitialList<'a> {
        let a = self.pattern.active_bus_count();
        let bus = if variable < a { variable } else { variable - a };
        let col = self.pattern.columns()[bus];
        InitialList {
            rows: self.pattern.active_rows(bus),
            pq: col.pq_count,
            shift: a,
            skip: if variable < a {
                self.pattern.diagonal_local(bus)
            } else {
                col.active_count + self.pattern.diagonal_local(bus)
            },
        }
    }
    pub(crate) fn note_initial_read(&self, variable: usize, len: usize) {
        if COUNT {
            self.reads.set(self.reads.get() + len);
            self.touched[variable].set(true);
        }
    }
    #[inline(always)]
    pub(crate) fn read_dynamic(&self, index: usize) -> isize {
        debug_assert!(
            self.written
                .get(index / 64)
                .is_some_and(|&x| x & (1u64 << (index % 64)) != 0),
            "dynamic list entry not initialized"
        );
        unsafe { self.data.get_unchecked(index).assume_init() }
    }
    #[inline]
    pub(crate) fn read(&self, index: usize) -> isize {
        assert!(
            COUNT || cfg!(debug_assertions) || cfg!(feature = "trace"),
            "compatibility read must not run in untraced ordering"
        );
        if self
            .written
            .get(index / 64)
            .is_some_and(|&bits| bits & (1u64 << (index % 64)) != 0)
        {
            // A set occupancy bit certifies this slot was allocated and written.
            return unsafe { self.data.get_unchecked(index).assume_init() };
        }
        assert!(
            index < self.initial_entries,
            "read of unwritten dynamic AMD entry {index}"
        );
        if COUNT {
            self.reads.set(self.reads.get() + 1);
        }
        let mut c = self.cursor.get();
        if index < c.start || index >= c.end {
            let scalar = self.starts.partition_point(|&p| p <= index) - 1;
            let a = self.pattern.active_bus_count();
            let bus = if scalar < a { scalar } else { scalar - a };
            let col = self.pattern.columns()[bus];
            let rows = self.pattern.active_rows(bus);
            let diag_in_rows = rows.binary_search(&bus).unwrap();
            c = Cursor {
                start: self.starts[scalar],
                end: self.starts[scalar + 1],
                bus,
                diag: if scalar < a {
                    diag_in_rows
                } else {
                    col.active_count + diag_in_rows
                },
                active: col.active_count,
                pq: col.pq_count,
            };
            self.cursor.set(c);
            if COUNT {
                self.touched[scalar].set(true);
            }
        }
        let local = index - c.start;
        let unskipped = local + usize::from(local >= c.diag);
        let rows = self.pattern.active_rows(c.bus);
        if unskipped < c.active {
            rows[unskipped] as isize
        } else {
            debug_assert!(unskipped - c.active < c.pq);
            (self.pattern.active_bus_count() + rows[unskipped - c.active]) as isize
        }
    }
    #[inline]
    pub(crate) fn write(&mut self, index: usize, value: isize) {
        if self.data.len() <= index {
            self.data.resize_with(index + 1, MaybeUninit::uninit);
        }
        if COUNT || cfg!(debug_assertions) || cfg!(feature = "trace") {
            let word = index / 64;
            if self.written.len() <= word {
                self.written.resize(word + 1, 0);
            }
            let bit = 1u64 << (index % 64);
            if self.written[word] & bit == 0 {
                if COUNT {
                    self.distinct += 1;
                }
                self.written[word] |= bit;
            }
        }
        self.data[index].write(value);
        if COUNT {
            self.writes += 1;
        }
    }
    #[cfg(feature = "trace")]
    pub(crate) fn ordered(&self, begin: usize, end: usize) -> Vec<isize> {
        (begin..end).map(|i| self.read(i)).collect()
    }
    fn counts(&self) -> Counts {
        Counts {
            initial_neighbor_reads: self.reads.get(),
            generated_neighbor_ids_saved: 0,
            dynamic_writes: self.writes,
            distinct_dynamic_slots: self.distinct,
            initial_variables_read: self.touched.iter().filter(|v| v.get()).count(),
            allocated_value_slots: self.data.capacity(),
        }
    }
}
/// Borrowed initial-list rule: holds no expanded scalar-neighbor buffer.
pub(crate) struct InitialList<'a> {
    rows: &'a [usize],
    pq: usize,
    shift: usize,
    skip: usize,
}
impl InitialList<'_> {
    #[inline(always)]
    pub(crate) fn at(&self, index: usize) -> isize {
        debug_assert!(index < self.rows.len() + self.pq - 1);
        let i = index + usize::from(index >= self.skip);
        if i < self.rows.len() {
            (unsafe { *self.rows.get_unchecked(i) }) as isize
        } else {
            (self.shift + unsafe { *self.rows.get_unchecked(i - self.rows.len()) }) as isize
        }
    }
}
#[derive(Debug)]
pub struct Ordering {
    pub permutation: Vec<usize>,
    pub inverse: Vec<usize>,
    pub info: Info,
    pub counts: Counts,
}
/// Full ordering from already analyzed Ybus through final scalar permutations.
/// This storage prototype retains stable logical ranges to avoid ever compacting
/// untouched initial lists into a complete table. Dynamic storage are allocated
/// only on writes made by the original elimination operations.
pub fn order<'a>(pattern: &'a YbusPattern<'a>, control: &Control) -> Result<Ordering, Status> {
    Workspace::default().order(pattern, control)
}
/// Reuses address metadata and dynamic storage. Each call clears occupancy and
/// recomputes the permutation; neither initial neighbors nor a prior order are reused.
#[derive(Default)]
pub struct Workspace {
    starts: Vec<usize>,
    offsets: Vec<isize>,
    lengths: Vec<usize>,
    data: Vec<MaybeUninit<isize>>,
    written: Vec<u64>,
    touched: Vec<Cell<bool>>,
}
impl Workspace {
    pub fn order<'a>(
        &mut self,
        pattern: &'a YbusPattern<'a>,
        control: &Control,
    ) -> Result<Ordering, Status> {
        self.execute::<false>(pattern, control)
    }
    pub fn order_counted<'a>(
        &mut self,
        pattern: &'a YbusPattern<'a>,
        control: &Control,
    ) -> Result<Ordering, Status> {
        self.execute::<true>(pattern, control)
    }
    fn execute<'a, const COUNT: bool>(
        &mut self,
        pattern: &'a YbusPattern<'a>,
        control: &Control,
    ) -> Result<Ordering, Status> {
        let n = pattern.scalar_dimension();
        let a = pattern.active_bus_count();
        let mut starts = std::mem::take(&mut self.starts);
        starts.clear();
        starts.reserve(n + 1);
        starts.push(0usize);
        for scalar in 0..n {
            let bus = if scalar < a { scalar } else { scalar - a };
            let col = pattern.columns()[bus];
            starts.push(
                starts
                    .last()
                    .unwrap()
                    .checked_add(col.active_count + col.pq_count - 1)
                    .ok_or(Status::Invalid)?,
            );
        }
        let entries = *starts.last().unwrap();
        let virtual_limit = entries
            .checked_add(n.checked_mul(n).ok_or(Status::Invalid)?)
            .and_then(|x| x.checked_add(n + 1))
            .filter(|&x| x < isize::MAX as usize)
            .ok_or(Status::Invalid)?;
        self.offsets.clear();
        self.offsets.extend(starts[..n].iter().map(|&x| x as isize));
        self.lengths.clear();
        self.lengths.extend(starts.windows(2).map(|x| x[1] - x[0]));
        // Reserve raw, UNINITIALIZED value slots, not generated neighbor IDs.
        // Every read still requires a write by elimination before accessing them.
        let reserve_values = entries + entries / 5 + n;
        if self.data.len() < reserve_values {
            self.data.resize_with(reserve_values, MaybeUninit::uninit);
        }
        self.written.fill(0);
        if COUNT {
            self.touched.resize_with(n, || Cell::new(false));
            for v in &self.touched {
                v.set(false);
            }
        }
        let mut storage = NeighborStorage::<COUNT> {
            pattern,
            starts,
            initial_entries: entries,
            data: std::mem::take(&mut self.data),
            written: std::mem::take(&mut self.written),
            cursor: Cell::new(Cursor::default()),
            reads: Cell::new(0),
            touched: std::mem::take(&mut self.touched),
            writes: 0,
            distinct: 0,
        };
        let mut info = Info::new(n);
        info.nz = pattern.scalar_nnz();
        info.nz_diag = n;
        info.symmetry = true;
        info.nz_a_plus_at = entries;
        let (_, inverse, permutation, _) = crate::lazy_core::amd_lazy::<usize, true, COUNT>(
            n,
            &mut self.offsets,
            &mut storage,
            &mut self.lengths,
            virtual_limit,
            entries,
            control,
            &mut info,
            None,
        );
        assert_eq!(
            info.n_cmp_a, 0,
            "unexpected compaction: must never materialize untouched initial lists"
        );
        let counts = storage.counts();
        self.starts = storage.starts;
        self.data = storage.data;
        self.written = storage.written;
        self.touched = storage.touched;
        Ok(Ordering {
            permutation,
            inverse,
            info,
            counts,
        })
    }
}

/// Timing control: same scalar AMD with all initial neighbors stored explicitly.
/// Prepared metadata and buffers can be reused, but the table is rebuilt every call.
#[derive(Default)]
pub struct FullTableControl {
    offsets: Vec<isize>,
    lengths: Vec<usize>,
    neighbors: Vec<isize>,
}
impl FullTableControl {
    pub fn order<'a>(&mut self, pattern: &'a YbusPattern<'a>, control: &Control) -> Ordering {
        let n = pattern.scalar_dimension();
        let a = pattern.active_bus_count();
        let entries = pattern.scalar_nnz() - n;
        let capacity = entries + entries / 5 + n;
        self.offsets.resize(n, 0);
        self.lengths.resize(n, 0);
        self.neighbors.resize(capacity, 0);
        let mut pos = 0;
        for id in 0..n {
            let bus = if id < a { id } else { id - a };
            self.offsets[id] = pos as isize;
            let rows = pattern.active_rows(bus);
            let q = pattern.columns()[bus].pq_count;
            for v in rows.iter().copied().chain(rows[..q].iter().map(|&i| a + i)) {
                if v != id {
                    self.neighbors[pos] = v as isize;
                    pos += 1;
                }
            }
            self.lengths[id] = pos - self.offsets[id] as usize;
        }
        let mut info = Info::new(n);
        info.nz = pattern.scalar_nnz();
        info.nz_diag = n;
        info.symmetry = true;
        info.nz_a_plus_at = entries;
        let (_, inverse, permutation, _) = crate::amd_2::amd_2::<usize, true, false>(
            n,
            &mut self.offsets,
            &mut self.neighbors,
            &mut self.lengths,
            capacity,
            entries,
            control,
            &mut info,
            None,
            None,
        );
        Ordering {
            permutation,
            inverse,
            info,
            counts: Counts {
                generated_neighbor_ids_saved: entries,
                allocated_value_slots: self.neighbors.capacity(),
                ..Default::default()
            },
        }
    }
    /// Diagnostic copy only: coarse allocation/table/kernel timing, never used by order.
    #[cfg(feature = "profile_stages")]
    pub fn order_profiled<'a>(&mut self, pattern: &'a YbusPattern<'a>, control: &Control) -> (Ordering, [f64; 3]) {
        let clock = std::time::Instant::now();
        let n = pattern.scalar_dimension();
        let a = pattern.active_bus_count();
        let entries = pattern.scalar_nnz() - n;
        let capacity = entries + entries / 5 + n;
        self.offsets.resize(n, 0);
        self.lengths.resize(n, 0);
        self.neighbors.resize(capacity, 0);
        let allocated = clock.elapsed().as_secs_f64();
        let mut pos = 0;
        for id in 0..n {
            let bus = if id < a { id } else { id - a };
            self.offsets[id] = pos as isize;
            let rows = pattern.active_rows(bus);
            let q = pattern.columns()[bus].pq_count;
            for v in rows.iter().copied().chain(rows[..q].iter().map(|&i| a + i)) {
                if v != id {
                    self.neighbors[pos] = v as isize;
                    pos += 1;
                }
            }
            self.lengths[id] = pos - self.offsets[id] as usize;
        }
        let mut info = Info::new(n);
        info.nz = pattern.scalar_nnz();
        info.nz_diag = n;
        info.symmetry = true;
        info.nz_a_plus_at = entries;
        let built = clock.elapsed().as_secs_f64();
        let (_, inverse, permutation, _) = crate::amd_2::amd_2::<usize, true, false>(
            n,
            &mut self.offsets,
            &mut self.neighbors,
            &mut self.lengths,
            capacity,
            entries,
            control,
            &mut info,
            None,
            None,
        );
        let finished = clock.elapsed().as_secs_f64();
        (Ordering {
            permutation,
            inverse,
            info,
            counts: Counts {
                generated_neighbor_ids_saved: entries,
                allocated_value_slots: self.neighbors.capacity(),
                ..Default::default()
            },
        }, [allocated, built - allocated, finished - built])
    }
}
