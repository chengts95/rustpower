//! Controlled per-thread App versus plain solver allocation probe (Linux/glibc).
#[cfg(all(target_os = "linux", target_env = "gnu"))]
mod probe {
    use bevy_app::{App, Plugin, Update};
    use bevy_ecs::prelude::*;
    use rustpower::{
        n1::*,
        prelude::{ecs::elements::line::LineID, *},
    };
    use serde_json::{Value, json};
    use std::{
        sync::{
            atomic::{AtomicUsize, Ordering},
            mpsc,
        },
        time::Duration,
    };
    #[repr(C)]
    struct MallInfo {
        arena: usize,
        ordblks: usize,
        smblks: usize,
        hblks: usize,
        hblkhd: usize,
        usmblks: usize,
        fsmblks: usize,
        uordblks: usize,
        fordblks: usize,
        keepcost: usize,
    }
    unsafe extern "C" {
        fn mallinfo2() -> MallInfo;
        fn malloc_trim(pad: usize) -> i32;
    }
    fn snapshot(label: &str) -> Value {
        let status = std::fs::read_to_string("/proc/self/status").unwrap();
        let number = |key: &str| {
            status
                .lines()
                .find(|l| l.starts_with(key))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|n| n.parse::<usize>().ok())
                .unwrap_or(0)
        };
        let m = unsafe { mallinfo2() };
        json!({"stage":label,"rss_kib":number("VmRSS:"),"hwm_kib":number("VmHWM:"),"virtual_kib":number("VmSize:"),"threads":number("Threads:"),"allocator_live_bytes":m.uordblks+m.hblkhd,"allocator_free_heap_bytes":m.fordblks,"allocator_arena_bytes":m.arena,"allocator_mmap_bytes":m.hblkhd})
    }
    #[derive(Resource)]
    struct Body {
        input: *const PreparedN1,
        engine: N1Worker,
        solutions: Vec<CaseSolution>,
        hits: Vec<WatchHits>,
    }
    // Scope below owns immutable PreparedN1 until all worker Apps are destroyed.
    unsafe impl Send for Body {}
    unsafe impl Sync for Body {}
    impl Body {
        fn new(p: &PreparedN1) -> Self {
            Self {
                input: p,
                engine: N1Worker::new(p.base().baseline()),
                solutions: vec![],
                hits: vec![],
            }
        }
        fn solve(&mut self, count: usize) {
            self.solutions.clear();
            self.hits.clear();
            let p = unsafe { &*self.input };
            for _ in 0..count {
                let s = self.engine.solve_base(p.base().baseline(), 50, 1e-10);
                assert!(s.converged);
                self.solutions.push(s);
            }
        }
        fn postprocess(&mut self) {
            let p = unsafe { &*self.input };
            for s in &self.solutions {
                self.hits.push(
                    p.watch()
                        .evaluate(p.base().catalog(), &s.voltage, None, s.converged)
                        .unwrap(),
                );
            }
        }
        fn stats(&self) -> Value {
            let w = self.engine.allocation_stats();
            json!({"vectors":w.vector_bytes,"dc_live":w.dc.current_bytes,"ac_live":w.ac.current_bytes,"dc_peak":w.dc.peak_bytes,"ac_peak":w.ac.peak_bytes,"voltage_capacity":self.solutions.iter().map(|s|s.voltage.capacity()*std::mem::size_of::<num_complex::Complex64>()).sum::<usize>(),"hit_record_capacity":self.hits.iter().map(|h|h.capacity()).sum::<usize>(),"unique_branch_results":self.hits.iter().map(|h|h.branch_results().len()).sum::<usize>()})
        }
    }
    #[derive(Message)]
    struct Go(usize);
    fn solve(mut messages: MessageReader<Go>, mut state: ResMut<Body>) {
        for message in messages.read() {
            state.solve(message.0);
        }
    }
    fn post(mut state: ResMut<Body>) {
        state.postprocess();
    }
    struct DiagPlugin;
    impl Plugin for DiagPlugin {
        fn build(&self, app: &mut App) {
            app.add_message::<Go>();
            app.add_systems(Update, (solve, post).chain());
        }
    }
    enum Runtime {
        Plain(Body),
        Bevy(App),
    }
    impl Runtime {
        fn new(p: &PreparedN1, apps: bool) -> Self {
            if apps {
                let mut a = App::new();
                a.insert_resource(Body::new(p));
                a.add_plugins(DiagPlugin);
                Self::Bevy(a)
            } else {
                Self::Plain(Body::new(p))
            }
        }
        fn step(&mut self, n: usize) {
            match self {
                Self::Plain(b) => {
                    b.solve(n);
                    b.postprocess();
                }
                Self::Bevy(a) => {
                    a.world_mut().write_message(Go(n));
                    a.update();
                }
            }
        }
        fn stats(&self) -> Value {
            match self {
                Self::Plain(b) => b.stats(),
                Self::Bevy(a) => a.world().resource::<Body>().stats(),
            }
        }
    }
    fn wait_stage(stage: &AtomicUsize, target: usize) {
        while stage.load(Ordering::Acquire) < target {
            std::thread::sleep(Duration::from_millis(1));
        }
    }
    pub fn run() -> Result<(), String> {
        let args: Vec<_> = std::env::args().collect();
        let apps = args.get(1).map(|s| s == "app").unwrap_or(false);
        let workers = args
            .get(2)
            .map(|s| s.parse())
            .transpose()
            .map_err(|e| format!("workers: {e}"))?
            .unwrap_or(16);
        if workers == 0 {
            return Err("workers>0 required".into());
        }
        let net = rustpower::io::pandapower::load_csv_zip(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/cases/pegase9241/data.zip"
        ))
        .map_err(|e| e.to_string())?;
        let mut world = PowerGrid::default();
        world.world_mut().insert_resource(PPNetwork(net));
        world.init_pf_net();
        let entities: Vec<_> = world
            .world_mut()
            .query_filtered::<Entity, With<LineID>>()
            .iter(world.world())
            .collect();
        for e in entities {
            world.world_mut().entity_mut(e).insert(N1Watch {
                rules: vec![WatchRule {
                    rule_id: 1,
                    condition: WatchCondition::BranchLoadingAbove(100.),
                }],
            });
        }
        let p = prepare_n1(world.world_mut())?;
        let mut samples = vec![snapshot("shared_ready")];
        let stage = AtomicUsize::new(0);
        let ready = AtomicUsize::new(0);
        let solved = AtomicUsize::new(0);
        let grouped = AtomicUsize::new(0);
        let (tx, rx) = mpsc::channel();
        std::thread::scope(|scope| -> Result<(), String> {
            let mut handles = vec![];
            for _ in 0..workers {
                let p = &p;
                let stage = &stage;
                let ready = &ready;
                let solved = &solved;
                let grouped = &grouped;
                let tx = tx.clone();
                match std::thread::Builder::new().spawn_scoped(scope, move || {
                    let mut runtime = Runtime::new(p, apps);
                    ready.fetch_add(1, Ordering::Release);
                    wait_stage(stage, 1);
                    if stage.load(Ordering::Acquire) == usize::MAX {
                        return;
                    }
                    runtime.step(1);
                    solved.fetch_add(1, Ordering::Release);
                    wait_stage(stage, 2);
                    if stage.load(Ordering::Acquire) == usize::MAX {
                        return;
                    }
                    runtime.step(8);
                    tx.send(runtime.stats()).unwrap();
                    grouped.fetch_add(1, Ordering::Release);
                    wait_stage(stage, 3);
                }) {
                    Ok(h) => handles.push(h),
                    Err(e) => {
                        stage.store(usize::MAX, Ordering::Release);
                        return Err(format!("spawn: {e}"));
                    }
                }
            }
            for (counter, label, next) in [
                (&ready, "workers_initialized", 1),
                (&solved, "single_base_with_factors", 2),
                (&grouped, "batch8_results_held", 3),
            ] {
                while counter.load(Ordering::Acquire) < workers {
                    if handles.iter().any(|h| h.is_finished()) {
                        stage.store(usize::MAX, Ordering::Release);
                        return Err("diagnostic worker failed".into());
                    }
                    std::thread::sleep(Duration::from_millis(1));
                }
                samples.push(snapshot(label));
                stage.store(next, Ordering::Release);
            }
            for h in handles {
                h.join().map_err(|_| "diagnostic worker panicked")?;
            }
            Ok(())
        })?;
        drop(tx);
        let detail: Vec<_> = rx.into_iter().collect();
        samples.push(snapshot("workers_dropped"));
        unsafe {
            malloc_trim(0);
        };
        samples.push(snapshot("allocator_trimmed"));
        println!("{}",serde_json::to_string_pretty(&json!({"mode":if apps{"app"}else{"plain"},"workers":workers,"snapshots":samples,"worker_allocations":detail,"scope":"Fresh process; identical shared base and worker numerical buffers. App variant adds a worker-local App, Resource, Message and two-stage schedule. Batch8 output is held in both variants. Not a full N-1 run or throughput benchmark. malloc_trim used only after workers drop in this diagnostic."})).unwrap());
        Ok(())
    }
}
fn main() -> Result<(), String> {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    {
        probe::run()
    }
    #[cfg(not(all(target_os = "linux", target_env = "gnu")))]
    {
        Err("probe requires Linux/glibc".into())
    }
}
