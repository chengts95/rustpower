//! `KktPattern`: the symbolic pattern of the exact-LM augmented system
//! (architecture doc §1.5), reduced **polar** convention:
//!
//! ```text
//! ┌ μI + H(r)    Jᵀ ┐ ┌ δ ┐   ┌  0 ┐
//! │                 │ │   │ = │    │
//! └ J            −I ┘ └ s ┘   └ −r ┘
//! ```
//!
//! States  `δ = [θ_0..θ_{n_act}, |V|_0..|V|_{n_pq})`
//! Equations `s = [P_0..P_{n_act}, Q_0..Q_{n_pq})`
//! (PV buses lose the Q equation and the |V| column; slack is eliminated and
//! enters through the `scalc`/`yv` channel only.)
//!
//! Key fact: **J, H(r) and Jᵀ share one column pattern**. Each column of bus
//! `k` is `[active neighbours][PQ neighbours, shifted by n_act]`:
//!
//! * J:  θ col = `[J11 | J21]`, |V| col = `[J12 | J22]` — the layout of
//!   `JacobianPattern2` (new_dsdvbus2.rs) itself;
//! * H:  θ col = `[aa | va]`, |V| col = `[av | vv]` — the polar Hessian
//!   quadrants of MATPOWER TN2 (`d2sbus_dv2.rs`, v4 Node Power Balance);
//! * Jᵀ: P col = `[θ rows | |V| rows]`, Q col likewise.
//!
//! So the symbolic phase builds the pattern **once** (`graph`) and places
//! the blocks by base offsets `[J | H | Jᵀ | −I]` (§3.4). `graph.col_starts`
//! are block-local (graph base = 0); a block's global start of column `c` is
//! `block_base + graph.col_starts[c]`.

use nalgebra_sparse::CscMatrix;
use num_complex::Complex64;

use super::block::BlockDesc;
use super::cache::YbusAnalysisCache;

pub struct KktPattern {
    pub cache: YbusAnalysisCache,
    /// Shared column pattern of J, H(r) and Jᵀ. Identical to
    /// `JacobianPattern2`'s layout; `col_starts` are block-local (base 0).
    pub graph: BlockDesc,
    /// Base of the J block in the global value array (= 0).
    pub j_base: usize,
    /// Base of the H(r) block.
    pub h_base: usize,
    /// Base of the Jᵀ block.
    pub jt_base: usize,
    /// Base of the `−I` block; the entry of equation `i` is at `d_base + i`.
    pub d_base: usize,
    /// Total length of the global value array, `−I` block included.
    pub nnz_total: usize,
}

impl KktPattern {
    pub fn build(ybus: &CscMatrix<Complex64>, n_pv: usize, n_pq: usize) -> Self {
        let cache = YbusAnalysisCache::build(ybus, n_pv, n_pq);
        let n_act = cache.n_active();
        let y_cp = cache.col_ptrs();
        let y_ri = cache.row_indices();
        let pq_ends = cache.pq_ends();
        let active_ends = cache.active_ends();

        // ── The one graph pattern ───────────────────────────────────────
        // Columns [θ_0..θ_{n_act}, |V|_0..|V|_{n_pq}); the column of bus k is
        //   [active neighbours][PQ neighbours, shifted by n_act]
        let mut graph = BlockDesc::empty(0);
        for c in 0..n_act + n_pq {
            let k = c % n_act;
            let rows = &y_ri[y_cp[k]..y_cp[k + 1]];
            graph.begin_column();
            for &r in rows.iter().take(active_ends[k]) {
                graph.push_row(r);
            }
            for &r in rows.iter().take(pq_ends[k]) {
                graph.push_row(n_act + r);
            }
        }

        // ── Block bases: [J | H | Jᵀ | −I] ──────────────────────────────
        let j_base = 0;
        let h_base = graph.nnz;
        let jt_base = 2 * graph.nnz;
        let d_base = 3 * graph.nnz;
        let nnz_total = d_base + n_act + n_pq; // −I: one entry per equation

        Self {
            cache,
            graph,
            j_base,
            h_base,
            jt_base,
            d_base,
            nnz_total,
        }
    }

    /// Global start of matrix column `c` inside a block.
    pub fn col_start(&self, block_base: usize, c: usize) -> usize {
        block_base + self.graph.col_starts[c]
    }

    /// Global position of `bus`'s diagonal inside a block column `col`,
    /// when the diagonal lies in the column's leading (active) segment:
    /// `block_base + col_starts[col] + diag_off[bus]`.
    ///
    /// Later-segment diagonals follow by adding the leading segment length
    /// `active_ends[bus]` — e.g. the `va`/`vv` diagonal quadrants, which are
    /// the slots `apply_mu_delta` touches together with `aa`.
    pub fn col_diag(&self, block_base: usize, col: usize, bus: usize) -> usize {
        block_base + self.graph.col_starts[col] + self.cache.diag_off()[bus]
    }
}
