//! Read-only branch data prepared once by the owning World.
//!
//! Workers receive solver-order voltages and a scenario-local outage index.
//! Device entities and names remain in the source World.
use bevy_ecs::{prelude::*, world::WorldId};
use std::sync::{
    OnceLock,
    atomic::{AtomicU64, Ordering},
};
static NEXT_CATALOG_ID: AtomicU64 = AtomicU64::new(1);
use num_complex::Complex64;
use soa_rs::{Soa, Soars};

use crate::basic::{
    branch::{Branch2Port, BranchFlowResult},
    ecs::post_processing::{BusPrecalc, TwoPortFlow, compute_two_port_flow_portable},
    ecs::{
        elements::{
            BusMatrixIndex, FromBusMatrixIndex, PFCommonData, ToBusMatrixIndex,
            bus::{OutOfService, VNominal},
            line::{LineID, LineParams},
            switch::{NodeAggRes, OpenLineModel, OpenTransformerModel},
            trans::{Port4MatPatch, TransformerDevice, TransformerID},
        },
        powerflow::systems::PowerFlowMat,
    },
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum BranchKind {
    Line,
    Transformer,
}

/// Identity belongs to the source case, never to a worker's Entity allocator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct BranchIdentity {
    pub kind: BranchKind,
    pub id: i64,
}

/// The existing AVX2 kernel vectorizes the real/imaginary components of
/// the two terminal currents of one branch. It is not cross-scenario SIMD.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PostprocessKernel {
    Portable,
    Avx2Fma,
}
type FlowKernel = fn(&[Complex64; 4], &BusPrecalc, &BusPrecalc, f64) -> TwoPortFlow;
static SELECTED_KERNEL: OnceLock<(PostprocessKernel, FlowKernel)> = OnceLock::new();
fn selected_kernel() -> (PostprocessKernel, FlowKernel) {
    *SELECTED_KERNEL.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            return (PostprocessKernel::Avx2Fma, avx2_flow);
        }
        (PostprocessKernel::Portable, compute_two_port_flow_portable)
    })
}
#[cfg(target_arch = "x86_64")]
fn avx2_flow(g: &[Complex64; 4], f: &BusPrecalc, t: &BusPrecalc, scale: f64) -> TwoPortFlow {
    // Selected only after the once-per-process AVX2/FMA capability check.
    unsafe {
        crate::basic::ecs::post_processing::compute_two_port_flow_avx2(
            g.as_ptr().cast::<f64>(),
            f,
            t,
            scale,
        )
    }
}
impl PostprocessKernel {
    pub fn available(self) -> bool {
        self == Self::Portable || selected_kernel().0 == self
    }
    fn function(self) -> Result<FlowKernel, String> {
        match self {
            Self::Portable => Ok(compute_two_port_flow_portable),
            Self::Avx2Fma if self.available() => Ok(selected_kernel().1),
            _ => Err("AVX2/FMA postprocessing is unavailable on this host".into()),
        }
    }
}

/// One branch definition; Soars generates the column storage and field views.
#[derive(Soars)]
struct BranchData {
    source_entity: Entity,
    phase_injection: f64,
    identity: BranchIdentity,
    from: usize,
    to: usize,
    y_ff: Complex64,
    y_tf: Complex64,
    y_ft: Complex64,
    y_tt: Complex64,
    base_i_from_ka: f64,
    base_i_to_ka: f64,
    in_service: bool,
    inv_line_rating: f64,
    transformer_loading_scale: f64,
}

/// Squared threshold metrics used by status-only screening.
pub(super) struct BranchMetrics {
    pub current_squared: f64,
    pub loading_squared: f64,
}

/// Immutable branch columns, owned together and borrowed by every worker.
pub struct BranchCatalog {
    id: u64,
    source_world: WorldId,
    branches: Soa<BranchData>,
    kernel_kind: PostprocessKernel,
    kernel: FlowKernel,
    n_bus: usize,
    sbase: f64,
}

impl BranchCatalog {
    /// Capture prepared ordinary two-port models in the final solver ordering.
    /// Open-terminal reconstruction is a separate model and is explicitly rejected.
    pub fn from_world(world: &mut World) -> Result<Self, String> {
        let sbase = world
            .get_resource::<PFCommonData>()
            .ok_or("N1 branch catalog requires PFCommonData")?
            .sbase;
        if !sbase.is_finite() || sbase <= 0.0 {
            return Err("N1 branch catalog requires a positive finite power base".into());
        }
        let mat = world
            .get_resource::<PowerFlowMat>()
            .ok_or("N1 branch catalog requires a prepared PowerFlowMat")?;
        let n_bus = mat.y_bus.nrows();
        let to_perm = mat.to_perm.clone();
        let merged = world
            .get_resource::<NodeAggRes>()
            .map(|a| a.original_to_merged.clone());
        let mut nominal = Vec::<Option<f64>>::new();
        for (index, vn) in world.query::<(&BusMatrixIndex, &VNominal)>().iter(world) {
            nominal.resize(nominal.len().max(index.0 + 1), None);
            nominal[index.0] = Some(vn.0.0);
        }
        let solver_index = |physical: usize| -> Result<usize, String> {
            let group = if let Some(map) = &merged {
                *map.get(physical)
                    .ok_or_else(|| format!("missing merged bus mapping for {physical}"))?
            } else {
                physical
            };
            let perm = *to_perm
                .get(group)
                .ok_or_else(|| format!("missing solver bus mapping for {physical}"))?;
            if perm >= n_bus {
                return Err(format!(
                    "solver bus mapping for {physical} is out of bounds"
                ));
            }
            Ok(perm)
        };
        let voltage_base = |physical: usize| -> Result<f64, String> {
            nominal
                .get(physical)
                .copied()
                .flatten()
                .filter(|v| v.is_finite() && *v > 0.0)
                .ok_or_else(|| format!("missing positive nominal voltage for bus {physical}"))
        };
        let expected_lines = world.query::<&LineParams>().iter(world).count();
        let expected_trafos = world.query::<&TransformerDevice>().iter(world).count();
        let mut records = Vec::new();
        for (entity, id, params, from, to, patch, off, open) in world
            .query::<(
                Entity,
                Option<&LineID>,
                &LineParams,
                &FromBusMatrixIndex,
                &ToBusMatrixIndex,
                Option<&Port4MatPatch>,
                Has<OutOfService>,
                Has<OpenLineModel>,
            )>()
            .iter(world)
        {
            let id = id
                .ok_or_else(|| format!("line {entity:?} has no persistent LineID"))?
                .0;
            if open && !off {
                return Err(format!(
                    "N1 branch catalog does not yet support open-terminal line {id}"
                ));
            }
            let from_vn = voltage_base(from.0)?;
            let to_vn = voltage_base(to.0)?;
            // The existing ordinary-line postprocessor uses from-side Vbase on
            // both terminals. Restrict this model to consistent line voltage bases.
            if (from_vn - to_vn).abs() > 1e-9 * from_vn.max(to_vn) {
                return Err(format!("line {id} has unequal terminal voltage bases"));
            }
            let y_mat = match patch {
                Some(p) => p.0.scale(from_vn * from_vn / sbase),
                None if off => nalgebra::Matrix2::zeros(),
                None => return Err(format!("line {id} has no prepared admittance patch")),
            };
            records.push((
                BranchIdentity {
                    kind: BranchKind::Line,
                    id,
                },
                Branch2Port {
                    from_bus: solver_index(from.0)?,
                    to_bus: solver_index(to.0)?,
                    y_mat,
                    base_i_from_ka: sbase / (3.0_f64.sqrt() * from_vn),
                    base_i_to_ka: sbase / (3.0_f64.sqrt() * to_vn),
                    max_i_ka: params.max_i_ka,
                    sn_trafo_mva: 0.0,
                    in_service: !off,
                },
                None,
                entity,
                0.0,
            ));
        }
        for (entity, id, dev, from, to, patch, off, open) in world
            .query::<(
                Entity,
                Option<&TransformerID>,
                &TransformerDevice,
                &FromBusMatrixIndex,
                &ToBusMatrixIndex,
                Option<&Port4MatPatch>,
                Has<OutOfService>,
                Has<OpenTransformerModel>,
            )>()
            .iter(world)
        {
            let id = id
                .ok_or_else(|| format!("transformer {entity:?} has no persistent TransformerID"))?
                .0;
            if open && !off {
                return Err(format!(
                    "N1 branch catalog does not yet support open-terminal transformer {id}"
                ));
            }
            if !dev.vn_hv_kv.is_finite()
                || !dev.vn_lv_kv.is_finite()
                || dev.vn_hv_kv <= 0.0
                || dev.vn_lv_kv <= 0.0
            {
                return Err(format!("transformer {id} has invalid rated voltages"));
            }
            let y_mat = match patch {
                Some(p) => p.0.scale(dev.vn_lv_kv * dev.vn_lv_kv / sbase),
                None if off => nalgebra::Matrix2::zeros(),
                None => return Err(format!("transformer {id} has no prepared admittance patch")),
            };
            let rating = dev.sn_mva * dev.parallel as f64;
            let shift = dev.effective_shift_degree();
            let phase = if !off && shift.abs() > 1e-6 {
                y_mat[(0, 1)].norm() * -shift.to_radians()
            } else {
                0.0
            };
            records.push((
                BranchIdentity {
                    kind: BranchKind::Transformer,
                    id,
                },
                Branch2Port {
                    from_bus: solver_index(from.0)?,
                    to_bus: solver_index(to.0)?,
                    y_mat,
                    base_i_from_ka: sbase / (3.0_f64.sqrt() * dev.vn_hv_kv),
                    base_i_to_ka: sbase / (3.0_f64.sqrt() * dev.vn_lv_kv),
                    max_i_ka: 0.0,
                    sn_trafo_mva: rating,
                    in_service: !off,
                },
                Some(rating),
                entity,
                phase,
            ));
        }
        if records
            .iter()
            .filter(|r| r.0.kind == BranchKind::Line)
            .count()
            != expected_lines
            || records
                .iter()
                .filter(|r| r.0.kind == BranchKind::Transformer)
                .count()
                != expected_trafos
        {
            return Err(
                "N1 branch catalog requires resolved endpoint indices on every device".into(),
            );
        }
        records.sort_unstable_by_key(|r| r.0);
        if records.windows(2).any(|r| r[0].0 == r[1].0) {
            return Err("N1 branch catalog has duplicate source device identities".into());
        }
        let mut branches = Soa::<BranchData>::with_capacity(records.len());
        for (identity, branch, rating, source_entity, phase_injection) in records {
            branches.push(BranchData {
                source_entity,
                phase_injection,
                identity,
                from: branch.from_bus,
                to: branch.to_bus,
                y_ff: branch.y_mat[(0, 0)],
                y_tf: branch.y_mat[(1, 0)],
                y_ft: branch.y_mat[(0, 1)],
                y_tt: branch.y_mat[(1, 1)],
                base_i_from_ka: branch.base_i_from_ka,
                base_i_to_ka: branch.base_i_to_ka,
                in_service: branch.in_service,
                inv_line_rating: if branch.max_i_ka > 0.0 {
                    100.0 / branch.max_i_ka
                } else {
                    0.0
                },
                transformer_loading_scale: match rating {
                    Some(r) if r > 0.0 => 100.0 * sbase / r,
                    _ => 0.0,
                },
            });
        }
        Ok(Self {
            id: NEXT_CATALOG_ID.fetch_add(1, Ordering::Relaxed),
            source_world: world.id(),
            branches,
            kernel_kind: selected_kernel().0,
            kernel: selected_kernel().1,
            n_bus,
            sbase,
        })
    }

    pub fn capacity(&self) -> usize {
        self.branches.capacity()
    }
    pub fn source_entities(&self) -> &[Entity] {
        self.branches.source_entity()
    }
    pub fn phase_injections(&self) -> &[f64] {
        self.branches.phase_injection()
    }
    pub(super) fn catalog_id(&self) -> u64 {
        self.id
    }
    pub(super) fn source_world_id(&self) -> WorldId {
        self.source_world
    }

    pub fn identities(&self) -> &[BranchIdentity] {
        self.branches.identity()
    }
    pub fn from_buses(&self) -> &[usize] {
        self.branches.from()
    }
    pub fn to_buses(&self) -> &[usize] {
        self.branches.to()
    }
    pub fn in_service(&self) -> &[bool] {
        self.branches.in_service()
    }
    /// Column-major 2x2 patch components (ff, tf, ft, tt), each contiguous by branch.
    pub fn admittances(&self) -> [&[Complex64]; 4] {
        [
            self.branches.y_ff(),
            self.branches.y_tf(),
            self.branches.y_ft(),
            self.branches.y_tt(),
        ]
    }
    pub fn kernel(&self) -> PostprocessKernel {
        self.kernel_kind
    }
    /// Preparation/diagnostic view, reconstructed on demand. No AoS catalog is kept.
    /// The hot numerical paths use the SoA fields and precomputed multipliers.
    pub fn branch(&self, index: usize) -> Option<Branch2Port> {
        let branch = self.branches.get(index)?;
        Some(Branch2Port {
            from_bus: *branch.from,
            to_bus: *branch.to,
            y_mat: nalgebra::Matrix2::new(*branch.y_ff, *branch.y_ft, *branch.y_tf, *branch.y_tt),
            base_i_from_ka: *branch.base_i_from_ka,
            base_i_to_ka: *branch.base_i_to_ka,
            max_i_ka: if *branch.inv_line_rating > 0.0 {
                100.0 / *branch.inv_line_rating
            } else {
                0.0
            },
            sn_trafo_mva: if *branch.transformer_loading_scale > 0.0 {
                100.0 * self.sbase / *branch.transformer_loading_scale
            } else {
                0.0
            },
            in_service: *branch.in_service,
        })
    }
    /// Terminal current multipliers, shared read-only with native backends.
    pub fn base_currents_ka(&self) -> [&[f64]; 2] {
        [self.branches.base_i_from_ka(), self.branches.base_i_to_ka()]
    }
    /// Precomputed line and transformer loading multipliers.
    pub fn loading_scales(&self) -> [&[f64]; 2] {
        [
            self.branches.inv_line_rating(),
            self.branches.transformer_loading_scale(),
        ]
    }
    pub fn solver_bus_count(&self) -> usize {
        self.n_bus
    }
    pub fn sbase(&self) -> f64 {
        self.sbase
    }
    pub fn len(&self) -> usize {
        self.branches.len()
    }
    pub fn is_empty(&self) -> bool {
        self.branches.is_empty()
    }

    /// Computes into a worker-owned buffer. `outage` is local to this scenario;
    /// it does not edit the shared catalog or allocate a modified branch array.
    pub fn compute(
        &self,
        voltage: &[Complex64],
        outage: Option<usize>,
        sbase: f64,
        out: &mut Vec<BranchFlowResult>,
    ) -> Result<(), String> {
        if voltage.len() != self.n_bus {
            return Err(format!(
                "N1 branch voltage length {} differs from {}",
                voltage.len(),
                self.n_bus
            ));
        }
        if sbase != self.sbase {
            return Err("N1 branch postprocessing power base differs from its catalog".into());
        }
        if outage.is_some_and(|i| i >= self.len()) {
            return Err("N1 branch outage index is out of bounds".into());
        }
        out.clear();
        out.reserve(self.len());
        for index in 0..self.len() {
            out.push(self.compute_validated(index, voltage, outage));
        }
        Ok(())
    }

    /// Compute one branch through the once-selected portable/AVX2 kernel.
    pub fn compute_one(
        &self,
        index: usize,
        voltage: &[Complex64],
        outage: Option<usize>,
    ) -> Result<BranchFlowResult, String> {
        if voltage.len() != self.n_bus {
            return Err("N1 branch voltage length differs from its catalog".into());
        }
        if index >= self.len() || outage.is_some_and(|i| i >= self.len()) {
            return Err("N1 branch index is out of bounds".into());
        }
        Ok(self.compute_validated(index, voltage, outage))
    }

    /// Threshold-only path: currents and precomputed scales, with no full flow output.
    pub(super) fn check_metrics(
        &self,
        index: usize,
        voltage: &[Complex64],
        outage: Option<usize>,
    ) -> Result<BranchMetrics, String> {
        if voltage.len() != self.n_bus
            || index >= self.len()
            || outage.is_some_and(|i| i >= self.len())
        {
            return Err("N1 branch voltage length or index differs from its catalog".into());
        }
        let b = self.branches.idx(index);
        if outage == Some(index) || !*b.in_service {
            return Ok(BranchMetrics {
                current_squared: 0.,
                loading_squared: 0.,
            });
        }
        let from = voltage[*b.from];
        let to = voltage[*b.to];
        let i0 = (*b.y_ff * from + *b.y_ft * to).norm_sqr();
        let i1 = (*b.y_tf * from + *b.y_tt * to).norm_sqr();
        let current_squared = (i0 * b.base_i_from_ka.powi(2)).max(i1 * b.base_i_to_ka.powi(2));
        let loading_squared = match b.identity.kind {
            BranchKind::Line => current_squared * b.inv_line_rating.powi(2),
            BranchKind::Transformer => i0.max(i1) * b.transformer_loading_scale.powi(2),
        };
        Ok(BranchMetrics {
            current_squared,
            loading_squared,
        })
    }

    /// Explicit kernel selection for numerical comparison and reproducibility.
    /// Feature detection is cached; this never invokes unsupported instructions.
    pub fn compute_with_kernel(
        &self,
        voltage: &[Complex64],
        outage: Option<usize>,
        kind: PostprocessKernel,
        out: &mut Vec<BranchFlowResult>,
    ) -> Result<(), String> {
        if voltage.len() != self.n_bus || outage.is_some_and(|i| i >= self.len()) {
            return Err("N1 branch voltage length or outage index is invalid".into());
        }
        let kernel = kind.function()?;
        out.clear();
        out.reserve(self.len());
        for index in 0..self.len() {
            out.push(self.compute_with(index, voltage, outage, kernel));
        }
        Ok(())
    }
    fn compute_validated(
        &self,
        index: usize,
        voltage: &[Complex64],
        outage: Option<usize>,
    ) -> BranchFlowResult {
        self.compute_with(index, voltage, outage, self.kernel)
    }
    fn compute_with(
        &self,
        index: usize,
        voltage: &[Complex64],
        outage: Option<usize>,
        kernel: FlowKernel,
    ) -> BranchFlowResult {
        let branch = self.branches.idx(index);
        if outage == Some(index) || !*branch.in_service {
            return BranchFlowResult::default();
        }
        let vf = voltage[*branch.from];
        let vt = voltage[*branch.to];
        const RAD_TO_DEG: f64 = 57.29577951308232;
        let f = BusPrecalc {
            vr: vf.re,
            vi: vf.im,
            vm: vf.norm(),
            va_deg: vf.arg() * RAD_TO_DEG,
            p_scale: self.sbase,
            i_scale: *branch.base_i_from_ka,
        };
        let t = BusPrecalc {
            vr: vt.re,
            vi: vt.im,
            vm: vt.norm(),
            va_deg: vt.arg() * RAD_TO_DEG,
            p_scale: self.sbase,
            i_scale: *branch.base_i_to_ka,
        };
        // Four SoA loads form the contiguous single-branch kernel input.
        let patch = [*branch.y_ff, *branch.y_tf, *branch.y_ft, *branch.y_tt];
        let flow = kernel(&patch, &f, &t, self.sbase);
        let i_from_ka = flow.i0_norm * *branch.base_i_from_ka;
        let i_to_ka = flow.i1_norm * *branch.base_i_to_ka;
        let i_ka = i_from_ka.max(i_to_ka);
        let loading_percent = match branch.identity.kind {
            BranchKind::Line => i_ka * *branch.inv_line_rating,
            BranchKind::Transformer => {
                flow.i0_norm.max(flow.i1_norm) * *branch.transformer_loading_scale
            }
        };
        BranchFlowResult {
            p_from_mw: flow.p0_mw,
            q_from_mvar: flow.q0_mvar,
            p_to_mw: flow.p1_mw,
            q_to_mvar: flow.q1_mvar,
            pl_mw: flow.pl_mw,
            ql_mvar: flow.ql_mvar,
            i_from_ka,
            i_to_ka,
            i_ka,
            vm_from_pu: f.vm,
            vm_to_pu: t.vm,
            va_from_degree: f.va_deg,
            va_to_degree: t.va_deg,
            loading_percent,
        }
    }
}
