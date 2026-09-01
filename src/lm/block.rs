//! Layer 1 — `BlockDesc`: one block matrix inside the global value array.
//!
//! A block is identified by **one integer**: its `base`. Everything else is
//! derived from Layer 0 ([`YbusAnalysisCache`]):
//!
//! * `col_starts[k]` — where matrix column `k` starts (includes `base`, so
//!   numeric kernels index the global values directly);
//! * `row_indices` — the global row numbers, flat, in emission order.
//!
//! There is deliberately **no diagonal table and no per-edge map**: the
//! diagonal slot of column `k` is `col_starts[k] + diag_off[k]`, and a later
//! segment's diagonal follows by adding the segment length from the cache.

use std::ops::Range;

pub struct BlockDesc {
    /// The single integer marking the block's start in the global values.
    pub base: usize,
    /// Per-column starts: `col_starts[k] = base + prefix-sum of the
    /// per-column lengths`.
    pub col_starts: Vec<usize>,
    /// Global row numbers, flat, in emission order.
    pub row_indices: Vec<usize>,
    /// Number of matrix columns owned by this block.
    pub n_cols: usize,
    /// Total number of nonzeros in this block.
    pub nnz: usize,
}

impl BlockDesc {
    /// Start building a block whose values begin at `base`.
    pub fn empty(base: usize) -> Self {
        Self {
            base,
            col_starts: Vec::new(),
            row_indices: Vec::new(),
            n_cols: 0,
            nnz: 0,
        }
    }

    /// Mark the start of the next matrix column.
    pub fn begin_column(&mut self) {
        self.col_starts.push(self.base + self.row_indices.len());
        self.n_cols += 1;
    }

    /// Emit one row entry into the current column.
    pub fn push_row(&mut self, row: usize) {
        self.row_indices.push(row);
        self.nnz += 1;
    }

    /// Absolute offset just past this block: the next block's `base`.
    pub fn end(&self) -> usize {
        self.base + self.nnz
    }

    /// Value-array range owned by matrix column `col` (includes `base`).
    pub fn col_range(&self, col: usize) -> Range<usize> {
        let end = if col + 1 < self.n_cols {
            self.col_starts[col + 1]
        } else {
            self.end()
        };
        self.col_starts[col]..end
    }

    /// Row numbers of matrix column `col`.
    pub fn col_rows(&self, col: usize) -> &[usize] {
        let range = self.col_range(col);
        &self.row_indices[range.start - self.base..range.end - self.base]
    }
}
