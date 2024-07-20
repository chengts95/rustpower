use nalgebra::{Complex, ComplexField};
use nalgebra_sparse::{CooMatrix, CscMatrix, CsrMatrix};
use num_traits::One;

use rustpower::{io::pandapower::Network, prelude::*};

#[macro_export]
macro_rules! timeit {
    ($name:ident, $times:expr, $block:expr) => {{
        use std::time::{Duration, Instant};
        let mut total_duration = Duration::new(0, 0);
        let mut max_duration = Duration::new(0, 0);
        let mut min_duration = Duration::new(u64::MAX, 999_999_999);

        for _ in 0..$times {
            let start_time = Instant::now();
            let _result = $block();
            let end_time = Instant::now();
            let duration = end_time - start_time;

            total_duration += duration;
            if duration > max_duration {
                max_duration = duration;
            }
            if duration < min_duration {
                min_duration = duration;
            }
        }

        let avg_duration = total_duration / $times;
        println!(
            " {} loops, {} - Average: {:?}, Max: {:?}, Min: {:?}",
            $times,
            stringify!($name),
            avg_duration,
            max_duration,
            min_duration
        );
    }};
}

fn create_incidence_mat(nodes: usize, y_br: &[Port2]) -> CooMatrix<Complex<f64>> {
    let mut incidence_matrix = CooMatrix::new(nodes, y_br.len());
    for (idx, i) in y_br.iter().enumerate() {
        if i.0[0] >= 0 {
            incidence_matrix.push(i.0[0] as usize, idx as usize, Complex::one());
        }
        if i.0[1] >= 0 {
            incidence_matrix.push(i.0[1] as usize, idx as usize, -Complex::one());
        }
    }
    incidence_matrix
}
#[allow(non_snake_case)]
fn main() {
    let file_path = test_ieee39::IEEE_39;
    let net: Network = serde_json::from_str(file_path).unwrap();
    let pf = PFNetwork::from(net);
    let v_init = pf.create_v_init();
    let tol = Some(1e-8);
    let max_it = Some(10);
    let (reorder, Ybus, mut Sbus, _, npv, npq) = pf.prepare_matrices(v_init.clone());
    let reverse_order = reorder.transpose();
    let (v, _) = pf.run_pf(v_init.clone(), max_it, tol);
    let cv = &reorder * &v;
    let mis = &cv.component_mul(&(Ybus * &cv).conjugate()) - &Sbus;
    Sbus.rows_range_mut(0..npv)
        .zip_apply(&mis.rows_range(0..npv), |x, y| (*x).im = y.im);
    Sbus[npv + npq] = mis[npv + npq];
    Sbus.scale_mut(-pf.s_base);

    let Sbus = reverse_order * Sbus;
    let lines: Vec<_> = pf.y_br.iter().map(|x| x).collect();
    let scale = 1.0 / pf.s_base;
    let y_lines: Vec<_> = lines
        .iter()
        .map(|x| x.v_base * x.v_base * scale * x.y.0)
        .collect();
    let indcies: Vec<_> = lines.iter().map(|x| x.port.clone()).collect();
    let mut diagline = CsrMatrix::identity(y_lines.len());
    diagline.values_mut().clone_from_slice(y_lines.as_slice());
    let imat =
        CscMatrix::from(&create_incidence_mat(v.len(), indcies.as_slice())).transpose_as_csr();
    let imat_f = imat.filter(|_, _, x| x.re > 0.0);
    let imat_t = imat.filter(|_, _, x| x.re < 0.0);
    let i_f = imat_f.transpose() * diagline * &imat_f * &v;
    let from_s = (&imat_f * &v).component_mul(&i_f);
    println!("{}", from_s.scale(100.0));
    println!("Vm,\t angle, P,\t Q");
    for (i, s) in v.iter().zip(Sbus.iter()) {
        print!("{:.5}, {:.5} ,", i.modulus(), i.argument().to_degrees());
        println!("{:.5}, {:.5}", s.re, s.im);
    }
    // println!("f,\t t,\t From P,\t From Q,\t To P,\t To Q");
    // for (i, p) in ibranch.iter().zip(indcies.iter()) {
    //     let from_s = pf.s_base * v[p.0[0] as usize] * i.conj();
    //     let to_s = pf.s_base * -v[p.0[1] as usize] * i.conj();
    //     print!("{},\t {},\t  ", p.0[0], p.0[1]);
    //     println!(
    //         "{:.5},\t{:.5},\t{:.5},\t{:.5},",
    //         from_s.re, from_s.im, to_s.re, to_s.im
    //     );
    // }
    timeit!(pf_ieee39, 100, || _ =
        (&pf).run_pf(v_init.clone(), max_it, tol));
}
