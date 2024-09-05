use bevy_app::prelude::*;
use bevy_ecs::{prelude::*, system::RunSystemOnce};
use bevy_hierarchy::prelude::*;
use csv::{Writer, WriterBuilder};
use nalgebra::*;
use nalgebra_sparse::*;
use num_complex::{Complex64, ComplexFloat};
use num_traits::Zero;
mod res_display;
use res_display::*;
use serde::{Deserialize, Serialize};
use tabled::{settings::Style, Table};

use crate::basic::{
    newton_pf,
    solver::RSparseSolver,
    system::{PFNetwork, RunPF},
};

use super::{elements::*, network::*};

/// Data structure for storing results of power flow calculations for a line.
#[derive(Component, Debug, Default, Serialize, Deserialize)]
struct LineResultData {
    p_from_mw: f64,       // Active power from the 'from' bus (MW)
    q_from_mvar: f64,     // Reactive power from the 'from' bus (MVAr)
    p_to_mw: f64,         // Active power to the 'to' bus (MW)
    q_to_mvar: f64,       // Reactive power to the 'to' bus (MVAr)
    pl_mw: f64,           // Line active power loss (MW)
    ql_mvar: f64,         // Line reactive power loss (MVAr)
    i_from_ka: f64,       // Current from the 'from' bus (kA)
    i_to_ka: f64,         // Current to the 'to' bus (kA)
    i_ka: f64,            // Line current (kA)
    vm_from_pu: f64,      // Voltage magnitude at the 'from' bus (p.u.)
    va_from_degree: f64,  // Voltage angle at the 'from' bus (degrees)
    vm_to_pu: f64,        // Voltage magnitude at the 'to' bus (p.u.)
    va_to_degree: f64,    // Voltage angle at the 'to' bus (degrees)
    loading_percent: f64, // Line loading percentage (%)
}
impl Into<LineResTable> for &LineResultData {
    fn into(self) -> LineResTable {
        LineResTable {
            from: 0,
            to: 0,
            p_from_mw: FloatWrapper::new(self.p_from_mw, 3),
            q_from_mvar: FloatWrapper::new(self.q_from_mvar, 3),
            p_to_mw: FloatWrapper::new(self.p_to_mw, 3),
            q_to_mvar: FloatWrapper::new(self.q_to_mvar, 3),
            pl_mw: FloatWrapper::new(self.pl_mw, 3),
            ql_mvar: FloatWrapper::new(self.ql_mvar, 3),
            i_from_ka: FloatWrapper::new(self.i_from_ka, 3),
            i_to_ka: FloatWrapper::new(self.i_to_ka, 3),
            i_ka: FloatWrapper::new(self.i_ka, 3),
            vm_from_pu: FloatWrapper::new(self.vm_from_pu, 2),
            va_from_degree: FloatWrapper::new(self.va_from_degree, 2),
            vm_to_pu: FloatWrapper::new(self.vm_to_pu, 2),
            va_to_degree: FloatWrapper::new(self.va_to_degree, 2),
            loading_percent: FloatWrapper::new(self.loading_percent, 1),
        }
    }
}
/// Extracts bus results after power flow calculation.
fn extract_res_bus(
    mut cmd: Commands,
    nodes: Res<NodeLookup>,
    mat: Res<PowerFlowMat>,
    res: Res<PowerFlowResult>,
) {
    let cv = &mat.reorder * &res.v;
    let mis = &cv.component_mul(&(&mat.y_bus * &cv).conjugate());
    let mut sbus_res = -mis.clone();

    sbus_res = &mat.reorder.transpose() * sbus_res;
    for (idx, entity) in nodes.0.iter() {
        cmd.entity(*entity).insert((
            SBusResult(sbus_res[*idx as usize]),
            VBusResult(res.v[*idx as usize]),
        ));
    }
}

/// Prints the results of the power flow for each bus.
fn print_res_bus(q: Query<(&PFNode, &VBusResult, &SBusResult)>, common: Res<PFCommonData>) {
    let bus_res_table = q
        .iter()
        .sort_by::<&PFNode>(|value_1, value_2| value_1.cmp(&value_2))
        .map(|(node, v, s)| {
            let vm = v.0.modulus();
            let angle = v.0.argument().to_degrees();
            let p = s.0.re();
            let q = s.0.im();
            BusResTable {
                Bus: node.0 as i32,
                Vm: FloatWrapper::new(vm, 3),
                Va: FloatWrapper::new(angle, 3),
                P_mw: FloatWrapper::new(p, 5),
                Q_mvar: FloatWrapper::new(q, 5),
            }
        });
    let table = Table::new(bus_res_table)
        .with(Style::markdown())
        .to_string();
    println!("{table}");
}

/// Enumeration for the type of admittance in a power grid branch.
enum AdmittanceType {
    FromToGround,
    ToToGround,
    BetweenBus,
}

/// Determines the type of admittance between two nodes.
fn determine_branch(parent: &Port2, child: &Port2) -> AdmittanceType {
    if parent[0] == child[0] && child[1] == GND {
        AdmittanceType::FromToGround
    } else if parent[1] == child[0] && child[1] == GND {
        AdmittanceType::ToToGround
    } else {
        AdmittanceType::BetweenBus
    }
}

/// Extracts line results after power flow calculation.

#[allow(unused_assignments)]
fn extract_res_line(
    mut cmd: Commands,
    q: Query<(Entity, &Children, &Port2), With<Line>>,
    admit: Query<(&Admittance, &VBase, &Port2), With<Parent>>,
    results: Res<PowerFlowResult>,
    common: Res<PFCommonData>,
) {
    q.iter().for_each(|(e, children, p)| {
        let mut data = LineResultData::default();
        let v_from = results.v[p[0] as usize];
        let v_to = results.v[p[1] as usize];

        data.vm_from_pu = v_from.modulus();
        data.va_from_degree = v_from.argument().to_degrees();
        data.vm_to_pu = v_to.modulus();
        data.va_to_degree = v_to.argument().to_degrees();

        let _s_base = common.sbase;
        let (mut i_f, mut i_t, mut i_l) = (Complex64::zero(), Complex64::zero(), Complex64::zero());
        let mut v_base = 0.0;

        for child in children {
            let (a, vbase, pins) = admit.get(*child).unwrap();
            match determine_branch(p, pins) {
                AdmittanceType::FromToGround => {
                    i_f += (v_from * vbase.0) * a.0;
                }
                AdmittanceType::ToToGround => {
                    i_t -= (v_to * vbase.0) * a.0;
                }
                AdmittanceType::BetweenBus => {
                    let v = v_from - v_to;
                    i_l = (v * vbase.0) * a.0;
                    let s = (v * vbase.0) * i_l.conj();
                    data.pl_mw = s.re();
                    data.ql_mvar += s.im();
                    i_f += i_l;
                    i_t += i_l;
                    v_base = vbase.0;
                }
            }
        }

        let s_f = v_from * v_base * i_f.conj();
        let s_t = -v_to * v_base * i_t.conj();
        data.p_from_mw = s_f.real();
        data.q_from_mvar = s_f.im();
        data.p_to_mw = s_t.real();
        data.q_to_mvar = s_t.im();
        data.pl_mw = data.p_to_mw + data.p_from_mw;
        data.ql_mvar = data.q_to_mvar + data.q_from_mvar;
        data.i_from_ka = i_f.modulus();
        data.i_to_ka = i_t.modulus();
        data.i_ka = data.i_from_ka.max(data.i_to_ka);

        cmd.entity(e).insert(data);
    });
}

/// Prints the results of the power flow for each line.
fn print_res_line(q: Query<(&Port2, &LineResultData)>) {
    let table = q.iter().map(|(p, record)| {
        let mut row_display: LineResTable = record.into();
        row_display.from = p[0];
        row_display.to = p[1];
        row_display
    });

    let table = Table::new(table).with(Style::markdown()).to_string();
    println!("{table}");
}

/// Trait for post-processing after a power flow simulation.
pub trait PostProcessing {
    /// Runs all post-processing steps.
    fn post_process(&mut self);

    /// Processes and prints the bus results.
    fn print_res_bus(&mut self);

    /// Processes and prints the line results.
    fn print_res_line(&mut self);
}

impl PostProcessing for PowerGrid {
    fn print_res_bus(&mut self) {
        self.world_mut().run_system_once(print_res_bus);
    }

    fn print_res_line(&mut self) {
        self.world_mut().run_system_once(print_res_line);
    }

    fn post_process(&mut self) {
        self.world_mut().run_system_once(extract_res_bus);
        self.world_mut().run_system_once(extract_res_line);
    }
}
#[cfg(test)]
#[allow(unused_imports)]
mod tests {
    use super::*;
    use crate::basic::new_ecs::network::PowerFlow;
    use crate::{
        basic::{
            self,
            system::{PFNetwork, RunPF},
        },
        io::pandapower::load_csv_zip,
    };
    use bevy_ecs::system::RunSystemOnce;
    use nalgebra::ComplexField;
    use std::env;

    /// Tests the ECS results for power flow.
    #[test]
    fn test_ecs_results() {
        let dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let folder = format!("{}/cases/IEEE118", dir);
        let name = folder.to_owned() + "/data.zip";
        let net = load_csv_zip(&name).unwrap();

        let mut pf_net = PowerGrid::default();
        pf_net.world_mut().insert_resource(PPNetwork(net));
        pf_net.init_pf_net();
        pf_net.run_pf();

        assert_eq!(
            pf_net
                .world()
                .get_resource::<PowerFlowResult>()
                .unwrap()
                .converged,
            true
        );

        pf_net.post_process();
        pf_net.print_res_bus();
        pf_net.print_res_line();
    }
}
