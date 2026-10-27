use crate::{
    powerflow::{OrderingWorkspace, YbusPattern},
    Control, Info, Status,
};
/// Compatibility predicate; specialized implementation lives in powerflow.
pub fn ybus_is_eligible(p: &[usize], rows: &[usize], npv: usize, npq: usize) -> bool {
    crate::powerflow::eligible(p, rows, npv, npq)
}

/// Exact supported J structure, with same-bus PQ variables premerged before AMD elimination.
/// This does not promise the same permutation as ordinary scalar AMD.
pub fn order_ybus(
    p: &[usize],
    i: &[usize],
    npv: usize,
    npq: usize,
    control: &Control,
) -> Result<(Vec<usize>, Vec<usize>, Info), Status> {
    let pattern = YbusPattern::analyze(p, i, npv, npq)?;
    Ok(OrderingWorkspace::new()
        .order(&pattern, control)?
        .into_parts())
}
/// # Safety
/// Caller must uphold all AMD workspace index invariants. Experimental unchecked kernel;
/// use order_ybus for checked diagnostics. Structural eligibility is validated in both variants.
pub unsafe fn order_ybus_unchecked(
    p: &[usize],
    i: &[usize],
    npv: usize,
    npq: usize,
    control: &Control,
) -> Result<(Vec<usize>, Vec<usize>, Info), Status> {
    let pattern = YbusPattern::analyze(p, i, npv, npq)?;
    Ok(OrderingWorkspace::new()
        .order_unchecked(&pattern, control)?
        .into_parts())
}
