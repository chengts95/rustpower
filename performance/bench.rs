//! 性能程序共用：整段计时、轮次顺序、原始记录与终端汇总。
//! 求解器内部的分项时间仍来自其已有计时器。
use serde_json::Value;
use std::path::PathBuf;

/// 返回(结果, 毫秒)。不依赖probe；不会重新计量求解器已有分项。
macro_rules! timeit {
    ($body:expr) => {{
        let started = std::time::Instant::now();
        let result = $body;
        (result, started.elapsed().as_secs_f64() * 1000.0)
    }};
}
pub(crate) use timeit;

pub fn option(name: &str) -> Option<String> {
    let args: Vec<_> = std::env::args().collect();
    args.windows(2).find(|a| a[0] == name).map(|a| a[1].clone())
}
pub fn repeats() -> usize {
    let n = option("--repeats")
        .map(|s| s.parse().expect("--repeats需要正整数"))
        .unwrap_or(7);
    assert!(n > 0);
    n
}
pub fn selected(case: &str) -> bool {
    option("--case").is_none_or(|name| name == case)
}
/// 第0轮预热；后续轮次交替反转方法顺序。
pub fn runs<T: Copy>(methods: &[T], repeats: usize) -> impl Iterator<Item = (usize, T)> + '_ {
    (0..=repeats).flat_map(move |round| {
        (0..methods.len()).map(move |slot| {
            let index = if round % 2 == 0 {
                slot
            } else {
                methods.len() - 1 - slot
            };
            (round, methods[index])
        })
    })
}

pub type Columns = &'static [(&'static str, &'static str)];

/// 同一份字段定义用于逐轮输出和汇总；JSON始终保留完整原始记录。
pub struct Report {
    rows: Vec<Value>,
    path: PathBuf,
    groups: &'static [Columns],
}
impl Report {
    pub fn new(directory: impl Into<PathBuf>, groups: &'static [Columns]) -> Self {
        let directory = directory.into();
        std::fs::create_dir_all(&directory).unwrap();
        println!(
            "原始数据：{}",
            directory.join("measurements.json").display()
        );
        Self {
            rows: Vec::new(),
            path: directory.join("measurements.json"),
            groups,
        }
    }
    pub fn push(&mut self, mut row: Value, description: &str) {
        row["description"] = description.into();
        println!(
            "\n{} / {description} / 第{}轮（0为预热）",
            row["case"].as_str().unwrap(),
            row["round"]
        );
        for group in self.groups {
            let values: Vec<_> = group
                .iter()
                .map(|(key, label)| format!("{label}={}", display(key, &row[key])))
                .collect();
            println!("  {}", values.join("，"));
        }
        self.rows.push(row);
        std::fs::write(&self.path, serde_json::to_vec_pretty(&self.rows).unwrap()).unwrap();
    }
    pub fn summary(&self, case: &str) {
        let rows: Vec<_> = self
            .rows
            .iter()
            .filter(|r| r["case"] == case && r["round"].as_u64().unwrap() > 0)
            .collect();
        let mut methods = Vec::new();
        for row in self.rows.iter().filter(|r| r["case"] == case) {
            let method = row["method"].as_str().unwrap();
            if !methods.contains(&method) {
                methods.push(method);
            }
        }
        println!(
            "\n{case} 汇总：时间取中位数，总执行另列最小–最大；计数有变化时列范围。失败运行仍保留，不计算失败加速比。"
        );
        for group in self.groups {
            print!("| 方法 |");
            for (_, label) in *group {
                print!(" {label} |");
            }
            println!();
            println!("|---|{}", "---:|".repeat(group.len()));
            for method in &methods {
                let samples: Vec<_> = rows
                    .iter()
                    .copied()
                    .filter(|r| r["method"] == *method)
                    .collect();
                print!("| {} |", samples[0]["description"].as_str().unwrap());
                for (key, _) in *group {
                    print!(" {} |", statistic(&samples, key));
                }
                println!();
            }
        }
    }
}
fn display(key: &str, value: &Value) -> String {
    if value.is_null() {
        return "—".into();
    }
    if key.ends_with("_ms") {
        return format!("{:.6}", value.as_f64().unwrap());
    }
    if value.is_f64() {
        return format!("{:.6e}", value.as_f64().unwrap());
    }
    value
        .as_str()
        .map(str::to_owned)
        .unwrap_or_else(|| value.to_string())
}
fn statistic(rows: &[&Value], key: &str) -> String {
    if key == "converged" {
        return format!(
            "{}/{}",
            rows.iter().filter(|r| r[key] == true).count(),
            rows.len()
        );
    }
    let mut numbers: Vec<_> = rows.iter().filter_map(|r| r[key].as_f64()).collect();
    if numbers.len() != rows.len() {
        return if rows.iter().all(|r| r[key] == rows[0][key]) {
            display(key, &rows[0][key])
        } else {
            "多种（见逐轮）".into()
        };
    }
    numbers.sort_by(f64::total_cmp);
    let (lo, hi) = (numbers[0], numbers[numbers.len() - 1]);
    if key.ends_with("_ms") {
        let middle = numbers.len() / 2;
        let median = (numbers[(numbers.len() - 1) / 2] + numbers[middle]) / 2.0;
        return if matches!(key, "total_ms" | "total_execution_ms") {
            format!("{median:.6} [{lo:.6}, {hi:.6}]")
        } else {
            format!("{median:.6}")
        };
    }
    if rows[0][key].is_f64() {
        return format!("{hi:.6e}");
    }
    if lo == hi {
        format!("{lo}")
    } else {
        format!("{lo}–{hi}")
    }
}
