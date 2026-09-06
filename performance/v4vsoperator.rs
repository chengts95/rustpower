//! 运行LM已有的原填充/operator切换，直接读取驱动器内置计时器。
//! taskset -c 1 cargo bench --features benchmark --bench comparison -- v4vsoperator
use super::lm_comparison::{Method, load_case, run_with_options};
use num_complex::Complex64;
use rustpower::lm::LmOptions;
use serde_json::json;

pub fn run() {
    let output = std::env::var("RUSTPOWER_V4_OPERATOR_OUTPUT").unwrap();
    let options = LmOptions::default();
    let methods = [
        (Method::FullBaseline, "完整增广/KLU：原填充"),
        (Method::FullOperator, "完整增广/KLU：operator"),
        (Method::UpperBaseline, "上三角/QDLDL：原填充"),
        (Method::UpperOperator, "上三角/QDLDL：operator"),
    ];
    let mut records = Vec::new();
    for name in ["IEEE39", "IEEE118", "pegase9241"] {
        let case = load_case(name);
        let mut references: [Option<(usize, u64, Vec<Complex64>)>; 2] = [None, None];
        // 第0轮预热，后7轮记录；每轮交换顺序，同一潮流内部复用求解器。
        for round in 0..8 {
            for slot in 0..methods.len() {
                let index = if round % 2 == 0 { slot } else { 3 - slot };
                let (method, label) = methods[index];
                let (m, voltage) = run_with_options(&case, method, 300, &options);
                assert!(m.converged, "{name} {label}未收敛，残差{}", m.residual_inf);
                let reference = &mut references[index / 2];
                if let Some((iterations, solves, v)) = reference {
                    assert_eq!(m.iterations, *iterations, "{name}：切换后迭代数不同");
                    assert_eq!(m.linear_solves, *solves, "{name}：切换后求解次数不同");
                    let error = voltage
                        .iter()
                        .zip(v.iter())
                        .map(
                            |(a, b): (&num_complex::Complex64, &num_complex::Complex64)| {
                                (*a - *b).norm()
                            },
                        )
                        .fold(0.0_f64, f64::max);
                    assert!(error < 1e-8, "{name}：切换后电压差{error}");
                } else {
                    *reference = Some((m.iterations, m.linear_solves, voltage));
                }
                // fill_ms和linear_solve_ms来自prof_fill_ns、prof_solve_ns。
                println!(
                    "{name} | {label} | 第{round}轮 | {}步/{}次线性求解 | 填充{:.3}ms（{:.1}µs/次） | 线性求解{:.3}ms | 潮流总计{:.3}ms | 残差{:.2e}",
                    m.iterations,
                    m.linear_solves,
                    m.fill_ms,
                    m.fill_ms * 1000.0 / m.iterations as f64,
                    m.linear_solve_ms,
                    m.solve_ms,
                    m.residual_inf,
                );
                records.push(json!({
                    "case":name, "method":method, "label":label, "round":round,
                    "options":options, "tolerance_inf":1e-8, "max_iterations":300,
                    "measurement":m,
                }));
            }
        }
    }
    std::fs::write(
        format!("{output}/lm_measurements.json"),
        serde_json::to_vec_pretty(&records).unwrap(),
    )
    .unwrap();
}
