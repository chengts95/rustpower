#![allow(deprecated)]
#[allow(unused_imports)]
use std::{f64::consts::PI, str::FromStr};

use super::{admittance, test_ieee39};
use crate::basic::newtonpf::newton_pf;
#[allow(unused_imports)]
use crate::basic::solver::RSparseSolver;
use crate::io::pandapower::*;
use bevy_ecs::component::Component;
use nalgebra::*;
use nalgebra_sparse::*;
use num_complex::Complex64;
use num_traits::One;

#[cfg(feature = "klu")]
use crate::basic::solver::KLUSolver;

/// Represents the ground node in the network.
pub const GND: i32 = -1;

/// Represents a branch with admittance and port information.
#[derive(Debug, Default, Clone, Component)]
#[cfg_attr(feature = "archive", derive(serde::Serialize, serde::Deserialize))]
pub struct AdmittanceBranch {
    /// The admittance value of the branch.
    pub y: admittance::Admittance,
    /// The port information of the branch.
    pub port: admittance::Port2,
    /// base voltage for per-unit values
    pub v_base: f64,
}

/// Represents a node with specified power and bus information in a power system.
#[derive(Debug, Clone, Copy, Default, Component)]
#[cfg_attr(feature = "archive", derive(serde::Serialize, serde::Deserialize))]
pub struct PQNode {
    /// The complex power injected at the node.
    pub s: Complex<f64>,
    /// The bus identifier of the node.
    pub bus: i64,
}

/// Represents a node with specified active power, voltage, and bus information in a power system.
#[derive(Debug, Clone, Copy, Default, Component)]
#[cfg_attr(feature = "archive", derive(serde::Serialize, serde::Deserialize))]
pub struct PVNode {
    /// The active power injected at the node.
    pub p: f64,
    /// The voltage magnitude at the node.
    pub v: f64,
    /// The bus identifier of the node.
    pub bus: i64,
}

/// Represents an external grid node with voltage, phase, and bus information.
#[derive(Debug, Clone, Copy, Component)]
#[cfg_attr(feature = "archive", derive(serde::Serialize, serde::Deserialize))]
pub struct ExtGridNode {
    /// The voltage magnitude at the external grid node.
    pub v: f64,
    /// The phase angle at the external grid node.
    pub phase: f64,
    /// The bus identifier of the external grid node.
    pub bus: i64,
}

impl Default for ExtGridNode {
    /// Creates a default external grid node with voltage set to 1.0 and other properties set to default.
    fn default() -> Self {
        Self {
            v: 1.0,
            phase: Default::default(),
            bus: Default::default(),
        }
    }
}
#[deprecated(
    since = "0.2.0",
    note = "This struct is deprecated.
     Use `default_app()` to create a bevy App with default plugins or `PowerGrid` instead."
)]
/// Represents a power flow network with base voltage and power, bus, load, PV node, external grid node, and branch information.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "archive", derive(serde::Serialize, serde::Deserialize))]
pub struct PFNetwork {
    /// The base voltage of the network.
    pub v_base: f64,
    /// The base power of the network.
    pub s_base: f64,
    /// The list of buses in the network.
    pub buses: Vec<Bus>,
    /// The list of PQ nodes in the network.
    pub pq_loads: Vec<PQNode>,
    /// The list of PV nodes in the network.
    pub pv_nodes: Vec<PVNode>,
    /// The external grid node in the network.
    pub ext: ExtGridNode,
    /// The list of branches with admittance and port information in the network.
    pub y_br: Vec<AdmittanceBranch>,
}

/// Creates the nodal admittance matrix (Ybus) of the power flow network.
///
/// This function calculates the nodal admittance matrix (Ybus) of the power flow network using the provided parameters.
///
/// # Arguments
///
/// * `pf` - A reference to the power flow network.
/// * `incidence_matrix` - A COO (Coordinate) matrix representing the network's incidence matrix.
/// * `admits` - An array of complex numbers representing the admittance values.
///
/// # Returns
///
/// The nodal admittance matrix (Ybus) of the power flow network as a CSR (Compressed Sparse Row) matrix.
fn create_ybus(
    pf: &PFNetwork,
    incidence_matrix: &CooMatrix<Complex<f64>>,
    admits: &[AdmittanceBranch],
) -> CsrMatrix<Complex<f64>> {
    let mut diag_admit = CsrMatrix::identity(pf.y_br.len());
    let y: Vec<_> = admits.iter().map(|x| x.y.0).collect();
    let base: Vec<_> = admits.iter().map(|x| x.v_base).collect();
    diag_admit.values_mut().clone_from_slice(y.as_slice());
    diag_admit
        .values_mut()
        .iter_mut()
        .zip(base)
        .for_each(|(x, vbase)| (*x) *= (vbase * vbase) / pf.s_base);

    let incidence_matrix = CsrMatrix::from(incidence_matrix);
    let ybus = &incidence_matrix * (diag_admit * incidence_matrix.transpose());

    ybus
}

/// Creates the incidence matrix of the power flow network.
///
/// This function creates the incidence matrix of the power flow network based on the provided number of nodes and admittance branch information.
///
/// # Arguments
///
/// * `nodes` - The number of nodes in the power flow network.
/// * `y_br` - A reference to a vector containing the admittance branch information.
///
/// # Returns
///
/// The incidence matrix of the power flow network as a COO (Coordinate) matrix.
fn create_incidence_mat(nodes: usize, y_br: &Vec<AdmittanceBranch>) -> CooMatrix<Complex<f64>> {
    let mut incidence_matrix = CooMatrix::new(nodes, y_br.len());
    for (idx, i) in y_br.iter().enumerate() {
        if i.port.0[0] >= 0 {
            incidence_matrix.push(i.port.0[0] as usize, idx as usize, Complex::one());
        }
        if i.port.0[1] >= 0 {
            incidence_matrix.push(i.port.0[1] as usize, idx as usize, -Complex::one());
        }
    }
    incidence_matrix
}

/// Creates the permutation matrix for reordering buses in the power flow network.
///
/// This function creates the permutation matrix for reordering buses in the power flow network based on PV nodes, PQ nodes, and external grid nodes.
///
/// # Arguments
///
/// * `pv` - A reference to a vector containing PV node indices.
/// * `pq` - A reference to a vector containing PQ node indices.
/// * `ext` - A reference to a vector containing external grid node indices.
/// * `nodes` - The total number of nodes in the power flow network.
///
/// # Returns
///
/// The permutation matrix for reordering buses in the power flow network as a COO (Coordinate) matrix.
fn create_premute_mat(
    pv: &Vec<i64>,
    pq: &Vec<i64>,
    ext: &Vec<i64>,
    nodes: usize,
) -> CooMatrix<i32> {
    let row_indices = DVector::from_fn(nodes, |i, _| i);
    let mut col_indices = DVector::from_fn(nodes, |i, _| i);
    let values = DVector::from_element(nodes, 1);

    let n_bus = pv.len() + pq.len();
    for i in 0..pv.len() {
        //let temp = col_indices[i];
        col_indices[i] = pv[i] as usize;
        //col_indices[pv[i] as usize] = temp;
    }
    for i in pv.len()..n_bus {
        //let temp = col_indices[i];
        col_indices[i] = pq[i - pv.len()] as usize;
        //col_indices[pv[i] as usize] = temp;
    }
    for i in n_bus..nodes {
        col_indices[i] = ext[i - n_bus] as usize;
    }
    let t = unsafe {
        CooMatrix::try_from_triplets(
            nodes,
            nodes,
            row_indices.data.into(),
            col_indices.data.into(),
            values.data.into(),
        )
        .unwrap_unchecked()
    };
    t
}

/// A trait for running power flow analysis.
pub trait RunPF {
    /// Creates the nodal admittance matrix (Ybus) of the power flow network.
    fn create_y_bus(&self) -> CsrMatrix<Complex64>;

    /// Creates the nodal power injection vector (Sbus) of the power flow network.
    fn create_s_bus(&self) -> DVector<Complex64>;

    /// Creates the initial voltage vector (V_init) of the power flow network.
    fn create_v_init(&self) -> DVector<Complex64>;

    /// Runs the power flow analysis.
    ///
    /// # Arguments
    ///
    /// * `v_init` - The initial voltage vector.
    /// * `max_it` - The maximum number of iterations (optional).
    /// * `tol` - The convergence tolerance (optional).
    ///
    /// # Returns
    ///
    /// The converged voltage vector and iterations.
    fn run_pf(
        &self,
        v_init: DVector<Complex64>,
        max_it: Option<usize>,
        tol: Option<f64>,
    ) -> (DVector<Complex64>, usize);
}

impl RunPF for PFNetwork {
    fn create_y_bus(&self) -> CsrMatrix<Complex64> {
        let nodes = self.buses.len();
        let incidence_matrix = create_incidence_mat(nodes, &self.y_br);
        let ybus = create_ybus(self, &incidence_matrix, self.y_br.as_slice());

        ybus
    }

    fn create_s_bus(&self) -> DVector<Complex64> {
        let nodes = self.buses.len();
        let mut sbus = DVector::zeros(nodes);
        for i in &self.pq_loads {
            sbus[i.bus as usize] -= i.s;
        }
        for i in &self.pv_nodes {
            sbus[i.bus as usize] += i.p;
        }

        let divider = 1.0 / self.s_base;
        sbus.apply(|x| (*x) *= divider);

        sbus
    }

    fn create_v_init(&self) -> DVector<Complex64> {
        let nodes = self.buses.len();
        let mut vbus = DVector::from_element(nodes, Complex64::one());
        for i in &self.pv_nodes {
            vbus[i.bus as usize] = Complex64::new(i.v, 0.0);
        }
        vbus[self.ext.bus as usize] = Complex64::from_polar(self.ext.v, self.ext.phase);

        vbus
    }
    #[allow(non_snake_case)]
    fn run_pf(
        &self,
        v_init: DVector<Complex64>,
        max_it: Option<usize>,
        tol: Option<f64>,
    ) -> (DVector<Complex64>, usize) {
        let (reorder, Ybus, Sbus, v_init, npv, npq) = self.prepare_matrices(v_init);

        #[cfg(feature = "klu")]
        let mut solver = KLUSolver::default();
        #[cfg(not(feature = "klu"))]
        let mut solver = RSparseSolver {};
        let v = newton_pf(&Ybus, &Sbus, &v_init, npv, npq, tol, max_it, &mut solver);
        let (v, iter) = v.unwrap();
        let x = reorder.transpose() * &v;

        (x, iter)
    }
}

impl PFNetwork {
    /// Prepares matrices for power flow analysis.
    #[allow(non_snake_case)]
    pub fn prepare_matrices(
        &self,
        v_init: Matrix<Complex<f64>, Dyn, Const<1>, VecStorage<Complex<f64>, Dyn, Const<1>>>,
    ) -> (
        CsrMatrix<Complex<f64>>,
        CscMatrix<Complex<f64>>,
        Matrix<Complex<f64>, Dyn, Const<1>, VecStorage<Complex<f64>, Dyn, Const<1>>>,
        Matrix<Complex<f64>, Dyn, Const<1>, VecStorage<Complex<f64>, Dyn, Const<1>>>,
        usize,
        usize,
    ) {
        let Sbus = self.create_s_bus();
        let Ybus = self.create_y_bus();
        let pv: Vec<_> = self.pv_nodes.iter().map(|x| x.bus).collect();
        let ext: Vec<_> = vec![self.ext.bus];
        let pq: Vec<_> = self
            .buses
            .iter()
            .flat_map(|x| {
                if pv.contains(&x.index) || ext.contains(&x.index) {
                    None
                } else {
                    Some(x.index)
                }
            })
            .collect();

        let reorder = create_premute_mat(&pv, &pq, &ext, self.buses.len());
        let from = CsrMatrix::from(&reorder);
        let reorder: CsrMatrix<Complex64> = CsrMatrix::try_from_pattern_and_values(
            from.pattern().clone(),
            Vec::from_iter(from.values().iter().map(|x| Complex64::new(*x as f64, 0.0))),
        )
        .unwrap();
        // Transform Ybus and Sbus according to the permutation
        let Ybus: CscMatrix<_> = (&reorder * Ybus * &reorder.transpose()).transpose_as_csc();

        let Sbus = &reorder * Sbus;
        let v_init = &reorder * v_init;
        let npv = pv.len();
        let npq = pq.len();
        (reorder, Ybus, Sbus, v_init, npv, npq)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_pf() {
        let (pf, _pv, _, _) = test_system();
        let v_init = pf.create_v_init();

        let v_actual = vec![
            "1.01051018-0.24328488j",
            "1.03324024-0.17819797j",
            "1.00713882-0.21915695j",
            "0.98016697-0.21957353j",
            "0.98687314-0.19526902j",
            "0.99163544-0.18214819j",
            "0.97375748-0.22043933j",
            "0.97096446-0.23016773j",
            "1.00670186-0.25433173j",
            "1.0075106 -0.14466175j",
            "1.00108312-0.15742722j",
            "0.98849655-0.15654167j",
            "1.002621  -0.15754284j",
            "0.99466736-0.18821936j",
            "0.99632818-0.19990713j",
            "1.01672943-0.17988707j",
            "1.01483154-0.19940437j",
            "1.00908201-0.2142324j",
            "1.04542896-0.09900757j",
            "0.98399585-0.1177033j",
            "1.02318219-0.13704417j",
            "1.04852254-0.05831164j",
            "1.04332564-0.06164282j",
            "1.02250159-0.17870812j",
            "1.04641909-0.15394765j",
            "1.03831111-0.17261321j",
            "1.01799494-0.20456408j",
            "1.04475607-0.1084877j",
            "1.0485082 -0.0580677j",
            "1.0412251 -0.13468596j",
            "0.982     +0.j        ",
            "0.98409468-0.00323655j",
            "0.99719433-0.00336208j",
            "1.01188982-0.02881467j",
            "1.04889561+0.03253237j",
            "1.06036708+0.08286498j",
            "1.02710791-0.02838292j",
            "1.02413166+0.06968932j",
            "0.99703319-0.25850496j",
        ];
        let v_actual = DVector::from_iterator(
            v_actual.len(),
            v_actual
                .iter()
                .map(|x| Complex64::from_str(&x.replace(" ", "")).unwrap()),
        );
        let (v, _) = pf.run_pf(v_init, Some(10), Some(1e-6));
        for i in 0..v.len() {
            assert!(
                (v[i] - v_actual[i]).norm() < 1e-3,
                "Mismatch at {} norm({}-{})={}!",
                i,
                v[i],
                v_actual[i],
                (v[i] - v_actual[i]).norm()
            );
        }
    }
    #[test]
    fn test_ybus() {
        let (pf, _pv, nodes, _) = test_system();

        let incidence_matrix = create_incidence_mat(nodes, &pf.y_br);
        let ybus = create_ybus(&pf, &incidence_matrix, &pf.y_br);
        let nan = ybus.values().iter().fold(false, |a, b| a | b.is_nan());
        assert_eq!(nan, false, "invalid parameters {:?}", ybus.values());
    }

    #[test]
    fn test_node_reordering() {
        let (pf, pv, nodes, _) = test_system();
        let pq: Vec<_> = pf
            .buses
            .iter()
            .flat_map(|x| {
                if pv.contains(&x.index) || pf.ext.bus == x.index {
                    None
                } else {
                    Some(x.index)
                }
            })
            .collect();
        let ext = vec![pf.ext.bus];
        let t = create_premute_mat(&pv, &pq, &ext, pf.buses.len());
        let o = CsrMatrix::from(&t);
        let v = DVector::from_fn(nodes, |i, _| i as i32);
        let ybus = o * v;
        println!("{:?} {:?}", pv, pq);
        println!("{}", ybus);
    }

    #[test]
    fn test_ybus_values() {
        let (pf, _pv, nodes, _) = test_system();
        let incidence_matrix = create_incidence_mat(nodes, &pf.y_br);
        let ybus = create_ybus(&pf, &incidence_matrix, &pf.y_br);
        //pandapower IEEE39 case for validation
        let data = vec![
            "3.65450097 -63.36747732j",
            "-2.05705688 +24.15572508j",
            "-1.59744409 +39.93610224j",
            "  -2.05705688 +24.15572508j",
            "64.64569545-211.87056921j",
            "  -5.65955594 +65.73791902j",
            "-56.92908263 +69.94144437j",
            "   0.         +53.9010915j ",
            "-5.65955594 +65.73791902j",
            "  14.69062005-186.84298941j",
            "-2.85475866 +46.77412271j",
            "  -6.17630545 +74.67714767j",
            "-2.85475866 +46.77412271j",
            "  12.50755723-201.57062289j",
            "-4.86381323 +77.82101167j",
            "  -4.78898533 +77.22238851j",
            "-4.86381323 +77.82101167j",
            "  40.6207556 -548.84384016j",
            "-29.41176471+382.35294118j",
            "  -6.34517766 +88.83248731j",
            "-29.41176471+382.35294118j",
            "  46.80574252-646.44708467j",
            "-7.05882353+108.23529412j",
            " -10.33515429+121.06895024j",
            " 0.         +37.38317757j",
            "  -7.05882353+108.23529412j",
            "25.82054961-323.89964402j",
            " -18.76172608+215.75984991j",
            "-6.34517766 +88.83248731j",
            " -18.76172608+215.75984991j",
            "26.84540319-331.72739372j",
            "  -1.73849945 +27.43805651j",
            "-1.73849945 +27.43805651j",
            "   3.33594354 -66.58395875j",
            "-1.59744409 +39.93610224j",
            "  42.89544236-504.72504178j",
            "-21.44772118+230.56300268j",
            " -21.44772118+230.56300268j",
            " 0.         +46.72897196j",
            " -10.33515429+121.06895024j",
            "-21.44772118+230.56300268j",
            "  32.62728731-374.48349985j",
            "-0.83937559 +22.82052378j",
            "  -0.83937559 +22.82052378j",
            " 1.66873874 -45.36883455j",
            "  -0.83937559 +22.82052378j",
            "-21.44772118+230.56300268j",
            "  -0.83937559 +22.82052378j",
            "31.04529388-351.62776596j",
            "  -8.75316086 +98.22991636j",
            "-4.78898533 +77.22238851j",
            "  -8.75316086 +98.22991636j",
            "17.33857334-220.88209317j",
            "  -3.79642714 +45.7680383j ",
            "-3.79642714 +45.7680383j ",
            "  13.88950777-150.91615824j",
            "-10.09308063+105.41661994j",
            " -10.09308063+105.41661994j",
            "36.02583832-510.42778861j",
            "  -8.78293601+111.66875784j",
            "-4.17961913 +50.93910817j",
            "  -4.37421401 +73.81486139j",
            "-8.59598854+169.05444126j",
            "  -8.78293601+111.66875784j",
            "23.43731417-289.92276034j",
            " -10.33515429+121.06895024j",
            "-4.31922387 +57.47890225j",
            "  -6.17630545 +74.67714767j",
            "-10.33515429+121.06895024j",
            "  16.51145974-195.57324791j",
            "-4.17961913 +50.93910817j",
            "  10.46740324-176.47474991j",
            "-3.45874068 +68.18660202j",
            "  -3.23655869 +65.6559048j ",
            "-3.45874068 +68.18660202j",
            "   6.38790579-126.71061147j",
            "-2.74613543 +54.92270865j",
            "  -4.37421401 +73.81486139j",
            " 8.44256226-144.75530578j",
            "  -4.06834825 +71.19609439j",
            "-4.06834825 +71.19609439j",
            "  10.55343256-241.29734057j",
            "-6.48508431+103.7613489j ",
            "   0.         +68.22445847j",
            "-6.48508431+103.7613489j ",
            "   8.9495284 -168.69982232j",
            "-1.78885058 +28.45898653j",
            "  -0.67559351 +36.75228688j",
            "-8.59598854+169.05444126j",
            "  -1.78885058 +28.45898653j",
            "10.38483912-197.2989278j ",
            " -56.92908263 +69.94144437j",
            "61.02681073-141.26083757j",
            "  -3.03740757 +30.65883269j",
            "-1.08682854 +42.02403702j",
            "  -3.03740757 +30.65883269j",
            "12.80336185-133.57752446j",
            "  -6.42054575 +67.41573034j",
            "-1.89824523 +20.92484273j",
            "  -1.44716331 +15.86801871j",
            "-4.31922387 +57.47890225j",
            "  -6.42054575 +67.41573034j",
            "10.73976962-124.61403259j",
            "  -1.89824523 +20.92484273j",
            " 7.9859958  -86.07098109j",
            "  -6.08775058 +65.66073836j",
            "-1.44716331 +15.86801871j",
            "  -6.08775058 +65.66073836j",
            "10.65561682-141.74346436j",
            "  -3.19872051 +62.37504998j",
            " 0.         +53.9010915j ",
            "   0.         -55.24861878j",
            " 0.         +37.38317757j",
            "   0.         -40.j        ",
            " 0.         +46.72897196j",
            "   0.         -50.j        ",
            "-3.23655869 +65.6559048j ",
            "   3.4631178  -70.25181814j",
            "-2.74613543 +54.92270865j",
            "   2.77085065 -55.41701302j",
            " 0.         +68.22445847j",
            "   0.         -69.93006993j",
            "-0.67559351 +36.75228688j",
            "   0.67559351 -36.75228688j",
            "-1.08682854 +42.02403702j",
            "   1.11399926 -43.07463795j",
            "-3.19872051 +62.37504998j",
            "   3.27868852 -63.93442623j",
            "-1.59744409 +39.93610224j",
            "  -1.59744409 +39.93610224j",
            " 3.19488818 -78.89720447j",
        ];
        let standard: CsrMatrix<_> = CsrMatrix::try_from_csr_data(
            39,
            39,
            vec![
                0, 3, 8, 12, 16, 20, 25, 28, 32, 35, 39, 43, 46, 50, 54, 57, 63, 67, 70, 74, 77,
                80, 84, 88, 91, 95, 100, 103, 106, 110, 112, 114, 116, 118, 120, 122, 124, 126,
                128, 131,
            ],
            vec![
                0, 1, 38, 0, 1, 2, 24, 29, 1, 2, 3, 17, 2, 3, 4, 13, 3, 4, 5, 7, 4, 5, 6, 10, 30,
                5, 6, 7, 4, 6, 7, 8, 7, 8, 38, 9, 10, 12, 31, 5, 9, 10, 11, 10, 11, 12, 9, 11, 12,
                13, 3, 12, 13, 14, 13, 14, 15, 14, 15, 16, 18, 20, 23, 15, 16, 17, 26, 2, 16, 17,
                15, 18, 19, 32, 18, 19, 33, 15, 20, 21, 20, 21, 22, 34, 21, 22, 23, 35, 15, 22, 23,
                1, 24, 25, 36, 24, 25, 26, 27, 28, 16, 25, 26, 25, 27, 28, 25, 27, 28, 37, 1, 29,
                5, 30, 9, 31, 18, 32, 19, 33, 21, 34, 22, 35, 24, 36, 28, 37, 0, 8, 38,
            ],
            data.iter()
                .map(|x| Complex64::from_str(&x.replace(" ", "")).unwrap())
                .collect(),
        )
        .unwrap();

        assert_eq!(
            ybus.pattern(),
            standard.pattern(),
            "The pattern doesn't match!"
        );
        println!("{:?}", ybus.values().iter().collect::<Vec<_>>());
        let error = ybus
            .values()
            .iter()
            .zip(standard.values().iter())
            .map(|(x, y)| (x - y).norm_squared());
        for (idx, i) in error.enumerate() {
            assert!(
                i < 0.0006,
                "The values doesn't match! ours={} standard={} at {:?}, error: {:?}",
                ybus.values()[idx],
                standard.values()[idx],
                ybus.col_indices()[idx],
                i
            );
        }
    }
}

pub fn test_system() -> (PFNetwork, Vec<i64>, usize, Vec<Complex<f64>>) {
    let file_path = test_ieee39::IEEE_39;
    let net: Network = serde_json::from_str(file_path).unwrap();
    let pf = PFNetwork::from(net);
    let pv: Vec<_> = pf.pv_nodes.iter().map(|x| x.bus).collect();
    let nodes = pf.buses.len();
    let admits: Vec<_> = pf.y_br.iter().map(|x| x.y.0).collect();
    (pf, pv, nodes, admits)
}
