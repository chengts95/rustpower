//! Experimental exact scalar-AMD input specialization and first-touch sharing.
//! Does not premerge PQ variables or change assembly layout.
use crate::{amd_2::amd_2, powerflow::YbusPattern, Control, Info};
#[derive(Default, Debug, Clone)]
pub struct Counters {
    pub pristine_updates: usize,
    pub pair_payloads: usize,
    pub pair_reuses: usize,
    pub own_twin_skips: usize,
    pub seed_entries_scanned: usize,
    pub entries_written: usize,
    pub suffix_scans_avoided: usize,
}
struct Payload {
    epoch: usize,
    begin: usize,
    count: usize,
    degree: usize,
    hash: u64,
}
pub(crate) struct FirstTouch<'a> {
    pattern: &'a YbusPattern<'a>,
    pristine: Vec<bool>,
    payloads: Vec<Option<Payload>>,
    data: Vec<isize>,
    pub counters: Counters,
}
impl FirstTouch<'_> {
    pub(crate) fn invalidate(&mut self, id: usize) {
        self.pristine[id] = false;
    }
    // Return false for original scalar processing. True supplies the EXACT ordered
    // suffix and its contributions, while leaving all scalar side effects in AMD.
    pub(crate) fn filter(
        &mut self,
        id: usize,
        epoch: usize,
        nv: &[isize],
        arena: &mut [isize],
        start: usize,
        pn: &mut isize,
        degree: &mut usize,
        hash: &mut u64,
    ) -> bool {
        let pristine = std::mem::replace(&mut self.pristine[id], false);
        if !pristine {
            return false;
        }
        self.counters.pristine_updates += 1;
        let a = self.pattern.active_bus_count();
        let q = self.pattern.pq_bus_count();
        let bus = if id < a { id } else { id - a };
        if bus >= q {
            return false;
        }
        let sibling = if id < a { a + bus } else { bus };
        // Own pristine twin is this pivot: every original neighbor is already
        // in this front or eliminated. Original scalar AMD mass-eliminates id.
        if sibling == epoch && nv[id] == -1 {
            self.counters.own_twin_skips += 1;
            self.counters.suffix_scans_avoided +=
                self.pattern.columns()[bus].active_count + self.pattern.columns()[bus].pq_count - 1;
            return true;
        }
        let hit = self.payloads[bus]
            .as_ref()
            .is_some_and(|p| p.epoch == epoch);
        if !hit {
            // Share only when both singleton seed lists enter the same front together.
            if !self.pristine[sibling] || nv[sibling] != -1 || nv[id] != -1 {
                return false;
            }
            let rows = self.pattern.active_rows(bus);
            let cut = self.pattern.columns()[bus].pq_count;
            let begin = self.data.len();
            let mut d = 0;
            let mut h = 0u64;
            for v in rows
                .iter()
                .copied()
                .chain(rows[..cut].iter().map(|&i| a + i))
            {
                self.counters.seed_entries_scanned += 1;
                if nv[v] > 0 {
                    self.data.push(v as isize);
                    d += nv[v] as usize;
                    h = h.wrapping_add(v as u64);
                }
            }
            self.payloads[bus] = Some(Payload {
                epoch,
                begin,
                count: self.data.len() - begin,
                degree: d,
                hash: h,
            });
            self.counters.pair_payloads += 1;
        } else {
            self.counters.pair_reuses += 1;
            self.counters.suffix_scans_avoided +=
                self.pattern.columns()[bus].active_count + self.pattern.columns()[bus].pq_count - 1;
        }
        let payload = self.payloads[bus].as_ref().unwrap();
        arena[start..start + payload.count]
            .copy_from_slice(&self.data[payload.begin..payload.begin + payload.count]);
        self.counters.entries_written += payload.count;
        *pn += payload.count as isize;
        *degree += payload.degree;
        *hash = hash.wrapping_add(payload.hash);
        true
    }
}
#[derive(Debug)]
pub struct ExactOrdering {
    pub permutation: Vec<usize>,
    pub inverse: Vec<usize>,
    pub info: Info,
    pub counters: Counters,
    pub input_us: f64,
    pub kernel_us: f64,
}
/// Exact scalar path; share=false is its control with identical input preparation.
/// The current prototype still materializes AMD's own mutable adjacency arena.
/// It avoids an intermediate J CSC/A+A' build, but is not yet a fully lazy arena.
pub fn order<'a>(pattern: &'a YbusPattern<'a>, control: &Control, share: bool) -> ExactOrdering {
    let input_started = std::time::Instant::now();
    let n = pattern.scalar_dimension();
    let a = pattern.active_bus_count();
    let q = pattern.pq_bus_count();
    let edges = pattern.scalar_nnz() - n;
    let capacity = edges + edges / 5 + n;
    let mut offsets = vec![0isize; n];
    let mut lengths = vec![0usize; n];
    let mut arena = vec![0isize; capacity];
    let mut pos = 0;
    for id in 0..n {
        let bus = if id < a { id } else { id - a };
        offsets[id] = pos as isize;
        let rows = pattern.active_rows(bus);
        let pq = pattern.columns()[bus].pq_count;
        for v in rows
            .iter()
            .copied()
            .chain(rows[..pq].iter().map(|&i| a + i))
        {
            if v != id {
                arena[pos] = v as isize;
                pos += 1;
            }
        }
        lengths[id] = pos - offsets[id] as usize;
    }
    assert_eq!(pos, edges);
    let mut sharing = FirstTouch {
        pattern,
        pristine: vec![true; n],
        payloads: (0..q).map(|_| None).collect(),
        data: Vec::with_capacity(edges / 2),
        counters: Counters::default(),
    };
    let mut info = Info::new(n);
    info.nz = pattern.scalar_nnz();
    info.nz_diag = n;
    info.symmetry = true;
    info.nz_a_plus_at = edges;
    let input_us = input_started.elapsed().as_secs_f64() * 1e6;
    let kernel_started = std::time::Instant::now();
    let (_, inverse, permutation, _) = if share {
        amd_2::<usize, true, true>(
            n,
            &mut offsets,
            &mut arena,
            &mut lengths,
            capacity,
            edges,
            control,
            &mut info,
            None,
            Some(&mut sharing),
        )
    } else {
        amd_2::<usize, true, false>(
            n,
            &mut offsets,
            &mut arena,
            &mut lengths,
            capacity,
            edges,
            control,
            &mut info,
            None,
            None,
        )
    };
    let kernel_us = kernel_started.elapsed().as_secs_f64() * 1e6;
    ExactOrdering {
        permutation,
        inverse,
        info,
        counters: sharing.counters,
        input_us,
        kernel_us,
    }
}
