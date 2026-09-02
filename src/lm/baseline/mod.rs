//! Ablation baselines for the LM assembly paper — the "write-it-like-a-
//! stranger" floors that [`super::gn_flat`] (AUG-SDF) is measured against.
//!
//! * [`aug_coo`] — **AUG-COO**: the same GN-LM augmented system
//!   `[μI Jᵀ; J −I]`, but assembled the way a general-purpose sparse stack
//!   makes you do it: COO push + sort/convert **every μ try**, and a fresh
//!   solver (full symbolic + numeric factorization) per solve. The J values
//!   come from the shared offset kernel (`fill_jacobian_v4`) so the only
//!   variables under test are assembly strategy and symbolic reuse.
//!
//! The NE-COO floor lives in [`super::normal_eq`] (`dumb_mode = true`).

#[cfg(feature = "qdldl")]
pub mod aug_coo;
#[cfg(feature = "qdldl")]
mod bench;
