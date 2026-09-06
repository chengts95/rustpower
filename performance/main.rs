//! 独立性能程序；不作为库的单元测试编译或运行。
// Existing benchmark imports resolve to the library's public API.
pub use rustpower::*;
mod acpf;
mod ecs;
mod jacobian;
mod kkt;
#[path = "lm.rs"]
mod lm_comparison;
#[path = "opf.rs"]
mod opf_comparison;
mod pf_builder;
mod v4;
mod v4vsoperator;

fn main() {
    // Some existing data loaders use the runtime variable; Cargo's root is fixed at build time.
    unsafe {
        std::env::set_var("CARGO_MANIFEST_DIR", env!("CARGO_MANIFEST_DIR"));
    }
    let name = std::env::args().nth(1).unwrap_or_else(|| "help".into());
    let output_key = match name.as_str() {
        "lm-assembly" | "lm-ablation" => Some("RUSTPOWER_NE_AUDIT_DIR"),
        "lm-solvers" => Some("RUSTPOWER_CHOLESKY_OUTPUT"),
        "v4vsoperator" => Some("RUSTPOWER_V4_OPERATOR_OUTPUT"),
        _ => None,
    };
    if let Some(key) = output_key {
        let output = std::env::var(key).unwrap_or_else(|_| {
            format!(
                "{}/target/research/performance/{name}",
                env!("CARGO_MANIFEST_DIR")
            )
        });
        std::fs::create_dir_all(&output).unwrap();
        unsafe {
            std::env::set_var(key, &output);
        }
        let command = |program: &str, args: &[&str]| {
            std::process::Command::new(program)
                .args(args)
                .output()
                .ok()
                .map(|x| String::from_utf8_lossy(&x.stdout).trim().to_owned())
        };
        let environment = serde_json::json!({
            "command": std::env::args().collect::<Vec<_>>(),
            "git_commit": command("git", &["rev-parse", "HEAD"]),
            "working_tree_changes": command("git", &["status", "--porcelain"]),
            "source_diff": command("git", &["diff", "HEAD", "--", "src/lm", "src/basic", "performance", "Cargo.toml", "Cargo.lock"]),
            "input_sha256": command("sha256sum", &[
                "cases/IEEE39/data.zip", "cases/IEEE118/data.zip", "cases/pegase9241/data.zip",
                "target/research/lm_audit/6515rte_dc.json", "target/research/lm_audit/6515rte_flat.json",
            ]),
            "rustc": command("rustc", &["--version"]),
            "os": std::env::consts::OS, "arch": std::env::consts::ARCH,
            "cpu_info": std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default().lines().find(|l| l.starts_with("model name")),
            "cpu_affinity": std::fs::read_to_string("/proc/self/status").unwrap_or_default().lines().find(|l| l.starts_with("Cpus_allowed_list")),
            "omp_threads": std::env::var("OMP_NUM_THREADS").ok(),
            "blas_preload": std::env::var("LD_PRELOAD").ok(),
            "cholmod_library": std::env::var("RUSTPOWER_CHOLMOD_LIBRARY").ok(),
            "cholmod_threads": std::env::var("RUSTPOWER_CHOLMOD_THREADS").ok(),
            "tolerance_inf": 1e-8, "max_iterations": 300,
            "warmup_rounds": 1, "measured_rounds": 7
        });
        std::fs::write(
            format!("{output}/environment.json"),
            serde_json::to_vec_pretty(&environment).unwrap(),
        )
        .unwrap();
    }
    match name.as_str() {
        "lm-check" => lm_comparison::operator_lm_matches_original_drivers(),
        "lm-assembly" => lm_comparison::benchmark_cached_normal_equations(),
        "lm-ablation" => lm_comparison::benchmark_coo_ablation(),
        "lm-solvers" => lm_comparison::linear_solvers::benchmark_linear_solvers(),
        "acpf" => acpf::acpf_v3_vs_v4_fill(),
        "jacobian" => jacobian::bench_jacobian_fill(),
        "ecs" => ecs::perf_pf_app_api(),
        "ecs-klu" => ecs::perf_pf_klu_breakdown(),
        "ecs-lm" => ecs::perf_pf_augmented_vs_normal(),
        "opf" => opf_comparison::bench_full_opf_all_cases(),
        "opf-assembly" => opf_comparison::bench_ablation_breakdown(),
        "opf-v4-v5" => opf_comparison::bench_v4_vs_v5_endtoend(),
        "kkt" => kkt::bench_v5_2_kkt_prep(),
        "pf-builder" => pf_builder::compare_new_pf_performance(),
        "v3-v4" => v4::v4_vs_v3_perf_ieee118(),
        "v4-fused" => v4::fused_vs_two_pass_perf_ieee118(),
        "v4vsoperator" => v4vsoperator::run(),
        "help" | "--help" => println!(
            "cargo bench --bench comparison --features benchmark -- <入口>\nLM: lm-check, lm-assembly, lm-ablation, lm-solvers\nACPF: acpf, jacobian, ecs, ecs-klu, ecs-lm, pf-builder, v3-v4, v4-fused, v4vsoperator\nOPF: opf, opf-assembly, opf-v4-v5, kkt"
        ),
        _ => panic!("未知性能入口：{name}；使用 --help 查看"),
    }
}
