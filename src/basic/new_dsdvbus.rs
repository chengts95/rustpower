pub struct JacobianPattern {
    pub nnz_j: usize,
    pub j_col_ptrs: Vec<usize>,    // 长度: n_active + npq + 1
    pub j_row_indices: Vec<usize>, // 长度: nnz_j

    // Ybus 拓扑切刀缓存 (热循环的“路标”，消灭运行时计算)
    pub pv_ends: Vec<usize>, // 长度: npv + npq
    pub pq_ends: Vec<usize>, // 长度: npv + npq

    // 区块一维内存绝对起点
    pub j11_starts: Vec<usize>, // 长度: npv + npq
    pub j21_starts: Vec<usize>, // 长度: npv + npq
    pub j12_starts: Vec<usize>, // 长度: npq (极度压缩，仅PQ)
    pub j22_starts: Vec<usize>, // 长度: npq (极度压缩，仅PQ)

    // 对角线元素在 Ybus 原始 values 数组中的绝对指针
    pub diag_ptrs: Vec<usize>, // 长度: npv + npq
}

impl JacobianPattern {
    pub fn build_from_permuted(
        y_col_ptrs: &[usize],
        y_row_indices: &[usize],
        npv: usize,
        npq: usize,
    ) -> Self {
        let n_active = npv + npq;
        let n_j_cols = n_active + npq; // 相角列 (PV+PQ) + 幅值列 (仅PQ)

        let mut j_col_ptrs = Vec::with_capacity(n_j_cols + 1);
        let mut j_row_indices = Vec::new();
        j_col_ptrs.push(0);

        let mut pv_ends = vec![0; n_active];
        let mut pq_ends = vec![0; n_active];

        let mut j11_starts = vec![0; n_active];
        let mut j21_starts = vec![0; n_active];
        // 绝杀：内存严格对齐物理语义，拒绝浪费！
        let mut j12_starts = vec![0; npq];
        let mut j22_starts = vec![0; npq];
        let mut diag_ptrs = vec![0; n_active];

        let pq_boundary = npv + npq;
        let mut current_nnz = 0;

        // ==========================================
        // 阶段 1: 推演所有的 相角(θ) 列 (共 n_active 列)
        // ==========================================
        for k in 0..n_active {
            let start = y_col_ptrs[k];
            let end = y_col_ptrs[k + 1];
            let row_slice = &y_row_indices[start..end];

            // 你的绝技：利用预排列，二分直接切出物理边界！
            let idx_pv_end = row_slice.partition_point(|&r| r < npv);
            let idx_pq_end = row_slice.partition_point(|&r| r < pq_boundary);

            pv_ends[k] = idx_pv_end;
            pq_ends[k] = idx_pq_end;

            // O(log N) 顺手锁定对角线指针
            if let Ok(diag_idx) = row_slice.binary_search(&k) {
                diag_ptrs[k] = start + diag_idx;
            }

            // --- 写入 J11 结构 (对应 P 方程，涵盖 PV 和 PQ 目标行) ---
            j11_starts[k] = current_nnz;
            for offset in 0..idx_pq_end {
                // 行号不变，P 方程的行索引就是 0..n_active
                j_row_indices.push(row_slice[offset]);
            }
            current_nnz += idx_pq_end;

            // --- 写入 J21 结构 (对应 Q 方程，仅涵盖 PQ 目标行) ---
            j21_starts[k] = current_nnz;
            for offset in idx_pv_end..idx_pq_end {
                let r = row_slice[offset];
                // 核心映射：Q 方程所在的行被统一推移到了 P 方程下方
                j_row_indices.push(n_active + r - npv);
            }
            current_nnz += idx_pq_end - idx_pv_end;

            j_col_ptrs.push(current_nnz);
        }

        // ==========================================
        // 阶段 2: 推演所有的 幅值(Vm) 列 (仅 npq 列)
        // ==========================================
        for k in npv..n_active {
            let pq_idx = k - npv; // 压缩寻址坐标

            let start = y_col_ptrs[k];
            let row_slice = &y_row_indices[start..y_col_ptrs[k + 1]];

            // 直接复用阶段 1 切好的物理边界，极致压榨算力！
            let idx_pv_end = pv_ends[k];
            let idx_pq_end = pq_ends[k];

            // --- 写入 J12 结构 ---
            j12_starts[pq_idx] = current_nnz;
            for offset in 0..idx_pq_end {
                j_row_indices.push(row_slice[offset]);
            }
            current_nnz += idx_pq_end;

            // --- 写入 J22 结构 ---
            j22_starts[pq_idx] = current_nnz;
            for offset in idx_pv_end..idx_pq_end {
                let r = row_slice[offset];
                j_row_indices.push(n_active + r - npv);
            }
            current_nnz += idx_pq_end - idx_pv_end;

            j_col_ptrs.push(current_nnz);
        }

        Self {
            nnz_j: current_nnz,
            j_col_ptrs,
            j_row_indices,
            pv_ends,
            pq_ends,
            j11_starts,
            j21_starts,
            j12_starts,
            j22_starts,
            diag_ptrs,
        }
    }
}
