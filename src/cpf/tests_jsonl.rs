//! CPF 大算例验证：回放 robustness 导出的 pypower 内部 reduced 输入
//! （Ybus CSC 已按 [PQ|PV|slack] 置换，与 v4 填充约定同源），
//! 鼻点对照文献值：
//!
//! | 算例       | λ_c 文献 | 来源 |
//! |------------|----------|------|
//! | IEEE 118   | 3.1871   | FLF 论文 Table II |
//! | IEEE 300   | 1.4293   | 同上 |
//! | 2383wp     | 1.8936   | 同上 |
//!
//! case39.jsonl 同时作为导入器自检：与 ECS 夹具同一电网，鼻点必须落在
//! 真值 2.1357 的 ±1% 内。
//!
//! 数据文件在 target/（不入库），缺失时测试打印 skip 直接返回，不失败。

use super::*;
use crate::basic::solver::KLUSolver;
use nalgebra::DVector;
use nalgebra_sparse::CscMatrix;
use num_complex::Complex64;

struct JsonlCase {
    ybus: CscMatrix<Complex64>,
    s_base: DVector<Complex64>,
    v_flat: DVector<Complex64>,
    npv: usize,
    npq: usize,
}

fn load_jsonl_base(case: &str) -> Option<JsonlCase> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/target/research/lm_robustness/");
    let text = std::fs::read_to_string(format!("{path}{case}.jsonl")).ok()?;
    let first = text.lines().next()?;
    let root: serde_json::Value = serde_json::from_str(first).ok()?;
    let base = root.get("base")?;

    let usizes = |key: &str| -> Vec<usize> {
        base[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| e.as_u64().unwrap() as usize)
            .collect()
    };
    let f64s = |key: &str| -> Vec<f64> {
        base[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| e.as_f64().unwrap())
            .collect()
    };

    let nb = base["nb"].as_u64().unwrap() as usize;
    let (cp, ri) = (usizes("cp"), usizes("ri"));
    let (y_re, y_im) = (f64s("y_re"), f64s("y_im"));
    let values: Vec<Complex64> = y_re
        .iter()
        .zip(y_im.iter())
        .map(|(re, im)| Complex64::new(*re, *im))
        .collect();
    let ybus = CscMatrix::try_from_csc_data(nb, nb, cp, ri, values)
        .expect("jsonl Ybus CSC data invalid");

    let cx = |re_key: &str, im_key: &str| -> DVector<Complex64> {
        let (re, im) = (f64s(re_key), f64s(im_key));
        DVector::from_vec(
            re.iter()
                .zip(im.iter())
                .map(|(r, i)| Complex64::new(*r, *i))
                .collect(),
        )
    };

    Some(JsonlCase {
        ybus,
        s_base: cx("s_re", "s_im"),
        v_flat: cx("v_flat_re", "v_flat_im"),
        npv: base["npv"].as_u64().unwrap() as usize,
        npq: base["npq"].as_u64().unwrap() as usize,
    })
}

/// 跑一个算例的 CPF 并对照文献鼻点。`band` 为允许相对偏差。
fn check_nose(case: &str, lambda_lit: f64, band: f64) {
    let Some(c) = load_jsonl_base(case) else {
        eprintln!("{case}: jsonl not found, skip");
        return;
    };
    let solver = KLUSolver::default();
    let mut cpf = CpfSolver::new(&c.ybus, c.npv, c.npq, c.s_base, c.v_flat, solver);
    let rep = cpf.run(&c.ybus, &CpfOptions::default()).unwrap();
    let err = (rep.lambda_nose - lambda_lit).abs() / lambda_lit;
    println!(
        "{case}: λ_nose={:.4} (lit {lambda_lit:.4}, err {:.2}%) | passed_nose={} \
         points={} fact={} resolve={} wall={:?}",
        rep.lambda_nose,
        err * 100.0,
        rep.passed_nose,
        rep.points.len(),
        rep.factorizations,
        rep.resolves,
        rep.wall
    );
    assert!(rep.passed_nose, "{case}: did not pass the nose ({})", rep.stop_reason);
    assert!(
        err < band,
        "{case}: λ_nose={:.4} deviates {:.2}% from literature {lambda_lit:.4} (band {:.1}%)",
        rep.lambda_nose,
        err * 100.0,
        band * 100.0
    );
}

/// 导入器自检：case39 jsonl 与 ECS 夹具同一电网，鼻点真值 2.1357。
#[test]
fn jsonl_case39_cross_check() {
    check_nose("case39", 2.1357, 0.01);
}

#[test]
fn jsonl_case118_nose() {
    check_nose("case118", 3.1871, 0.02);
}

#[test]
fn jsonl_case300_nose() {
    check_nose("case300", 1.4293, 0.02);
}

#[test]
fn jsonl_case2383wp_nose() {
    check_nose("case2383wp", 1.8936, 0.02);
}
