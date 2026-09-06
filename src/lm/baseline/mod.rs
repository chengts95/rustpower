//! 历史装配基线：COO 重建与完整 Jacobian 后切片。
//! 保留实现供研究参考；当前公平性能对照统一放在 `lm::comparison`。
//! 注意这些历史驱动器可能每次试步创建求解器，不能直接归因于装配成本。

#[cfg(feature = "qdldl")]
pub mod aug_coo;
#[cfg(feature = "qdldl")]
pub mod full_slice;
#[cfg(feature = "qdldl")]
mod tests;
