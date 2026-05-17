use nalgebra_sparse::CscMatrix;
use num_complex::Complex64;

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

#[allow(non_snake_case)]
#[inline(never)] // 保持独立函数，方便 LLVM 专注优化硬件寄存器分配与 AVX 向量化
pub fn fill_jacobian_ultimate(
    Ybus: &CscMatrix<Complex64>,
    v: &[Complex64],
    Vnorm: &[Complex64],
    ibus: &[Complex64],
    pattern: &JacobianPattern,
    npv: usize,
    npq: usize,
    j_values: &mut [f64],
) {
    let y_col_offsets = Ybus.col_offsets();
    let y_row_indices = Ybus.row_indices();
    let y_vals = Ybus.values();
    let n_active = npv + npq;

    // 彻底重置目标数组（保证未触及区域如 PV 的 Q 项物理清零）
    //j_values.fill(0.0);

    for k in 0..n_active {
        let y_start = y_col_offsets[k];
        let pv_end = pattern.pv_ends[k];
        let pq_end = pattern.pq_ends[k];

        // --- 循环不变量外提：解包当前列节点 k 的电学状态到寄存器 ---
        let ek = v[k].re;
        let fk = v[k].im;
        let enk = Vnorm[k].re;
        let fnk = Vnorm[k].im;
        let Ire_k = ibus[k].re;
        let Iim_k = ibus[k].im;

        // 静态锁定对角线元在当前列一维 Ybus values 中的相对偏移量
        let diag_offset = pattern.diag_ptrs[k] - y_start;

        // =====================================================================
        // 第一幕：相角 (Va) 导数盲填流 —— 覆盖 0..pq_end (所有活跃行)
        // =====================================================================
        // 直接用裸指针切片，避开借用检查器，保证不出界即可
        let j_ptr = j_values.as_mut_ptr();
        let out_j11 =
            unsafe { std::slice::from_raw_parts_mut(j_ptr.add(pattern.j11_starts[k]), pq_end) };
        let out_j21 = unsafe {
            std::slice::from_raw_parts_mut(j_ptr.add(pattern.j21_starts[k]), pq_end - pv_end)
        };

        // 核心劈裂 1A：只命中 PV 目标行 (0..pv_end)
        for offset in 0..pv_end {
            let y_ptr = y_start + offset;
            let i = y_row_indices[y_ptr];
            let Y_ik = y_vals[y_ptr];

            let Va_re = Y_ik.re * ek - Y_ik.im * fk;
            let Va_im = Y_ik.re * fk + Y_ik.im * ek;

            out_j11[offset] = v[i].im * Va_re - v[i].re * Va_im;
        }

        // 核心劈裂 1B：命中 PQ 目标行 (pv_end..pq_end) -> 双路并发直写
        for offset in pv_end..pq_end {
            let y_ptr = y_start + offset;
            let i = y_row_indices[y_ptr];
            let Y_ik = y_vals[y_ptr];

            let Va_re = Y_ik.re * ek - Y_ik.im * fk;
            let Va_im = Y_ik.re * fk + Y_ik.im * ek;

            let ei = v[i].re;
            let fi = v[i].im;

            out_j11[offset] = fi * Va_re - ei * Va_im;
            out_j21[offset - pv_end] = -(ei * Va_re + fi * Va_im);
        }

        // =====================================================================
        // 第二幕：电压幅值 (Vm) 导数盲填流 —— 仅当源节点 k 拥有 Vm 变量 (k >= npv) 时触发
        // =====================================================================
        if k >= npv {
            let pq_idx = k - npv;

            let start_12 = pattern.j12_starts[pq_idx];
            let start_22 = pattern.j22_starts[pq_idx];

            let j_ptr = j_values.as_mut_ptr();
            let out_j12 = unsafe { std::slice::from_raw_parts_mut(j_ptr.add(start_12), pq_end) };
            let out_j22 =
                unsafe { std::slice::from_raw_parts_mut(j_ptr.add(start_22), pq_end - pv_end) };

            // 核心劈裂 2A：只命中 PV 目标行 (0..pv_end)
            for offset in 0..pv_end {
                let y_ptr = y_start + offset;
                let i = y_row_indices[y_ptr];
                let Y_ik = y_vals[y_ptr];

                let Vm_re = Y_ik.re * enk - Y_ik.im * fnk;
                let Vm_im = Y_ik.re * fnk + Y_ik.im * enk;

                out_j12[offset] = v[i].re * Vm_re + v[i].im * Vm_im;
            }

            // 核心劈裂 2B：命中 PQ 目标行 (pv_end..pq_end) -> 双路并发直写
            for offset in pv_end..pq_end {
                let y_ptr = y_start + offset;
                let i = y_row_indices[y_ptr];
                let Y_ik = y_vals[y_ptr];

                let Vm_re = Y_ik.re * enk - Y_ik.im * fnk;
                let Vm_im = Y_ik.re * fnk + Y_ik.im * enk;

                let ei = v[i].re;
                let fi = v[i].im;

                out_j12[offset] = ei * Vm_re + fi * Vm_im;
                out_j22[offset - pv_end] = fi * Vm_re - ei * Vm_im;
            }
        }

        // =====================================================================
        // 第三幕：对角线单点爆破修正 (O(1) 寄存器级增量追加)
        // =====================================================================
        unsafe {
            // 修正 J11 对角线: 显式追加 -Q_k = ek * Iim_k - fk * Ire_k
            *j_values.get_unchecked_mut(pattern.j11_starts[k] + diag_offset) +=
                ek * Iim_k - fk * Ire_k;

            if k >= npv {
                let pq_idx = k - npv;
                let start_21 = pattern.j21_starts[k];
                let start_12 = pattern.j12_starts[pq_idx];
                let start_22 = pattern.j22_starts[pq_idx];

                // 修正 J21 对角线: 显式追加 +P_k = ek * Ire_k + fk * Iim_k
                *j_values.get_unchecked_mut(start_21 + diag_offset - pv_end) +=
                    ek * Ire_k + fk * Iim_k;

                // 修正 J12 对角线: 显式追加 +enk * Ire_k + fnk * Iim_k (上一版已正确)
                *j_values.get_unchecked_mut(start_12 + diag_offset) += enk * Ire_k + fnk * Iim_k;

                // 修正 J22 对角线: 显式追加 +fnk * Ire_k - enk * Iim_k (上一版已正确)
                *j_values.get_unchecked_mut(start_22 + diag_offset - pv_end) +=
                    fnk * Ire_k - enk * Iim_k;
            }
        }
    }
}
