use std::{time::Instant,fs::File,io::Write,hint::black_box};
use rustpower::{prelude::*,io::pandapower::{Network,load_csv_zip},testcases::case_ieee39::IEEE_39};
use rustpower::prelude::{ecs::powerflow::systems::PowerFlowMat,jacobian_cache::JacobianCache};
use rustpower_sol_klu::KLUSolver as RawKlu;
use serde_json::json;
#[repr(C)]#[derive(Default,Debug)]struct Trace{amd_us:f64,prep_us:f64,amd_calls:u64}
#[link(name="rpamd")]unsafe extern "C"{
 fn rp_trace_reset();fn rp_trace_get(t:*mut Trace);
 fn rp_profile_analyze(n:i64,ap:*const usize,ai:*const usize,c:*mut std::ffi::c_void)->*mut std::ffi::c_void;
}
fn main(){
 let args:Vec<_>=std::env::args().collect();let reps:usize=args.get(1).map(|s|s.parse().unwrap()).unwrap_or(200);let mut out=File::create(args.get(2).map(|s|s.as_str()).unwrap_or("amd-cost.jsonl")).unwrap();
 for(case,net)in[("IEEE39",serde_json::from_str::<Network>(IEEE_39).unwrap()),("IEEE118",load_csv_zip("cases/IEEE118/data.zip").unwrap()),("PEGASE9241",load_csv_zip("cases/pegase9241/data.zip").unwrap())]{
 let mut grid=PowerGrid::default();grid.world_mut().insert_resource(PPNetwork(net));grid.init_pf_net();let m=grid.world().resource::<PowerFlowMat>();let mut expected:Option<Vec<i64>>=None;
 for pair in 0..(reps+10){for kind in if pair%2==0{[0,1]}else{[1,0]}{
 let t=Instant::now();let mut p=black_box(JacobianCache::build_from_permuted(m.y_bus.col_offsets(),m.y_bus.row_indices(),m.npv,m.npq));let jprep=t.elapsed().as_secs_f64()*1e6;
 let n=p.j_col_ptrs.len()-1;let mut k=RawKlu::default();unsafe{(*k.common).btf=0;rp_trace_reset();}
 let t=Instant::now();unsafe{if kind==0{k.solve_sym(p.j_col_ptrs.as_mut_ptr().cast(),p.j_row_indices.as_mut_ptr().cast(),n as i64);}else{k.symbolic=rp_profile_analyze(n as i64,p.j_col_ptrs.as_ptr(),p.j_row_indices.as_ptr(),k.common.cast()).cast();}}
 let total=t.elapsed().as_secs_f64()*1e6;let mut trace=Trace::default();unsafe{rp_trace_get(&mut trace);assert!(!k.symbolic.is_null());assert_eq!((*k.common).status,0);assert_eq!((*k.symbolic).do_btf,0);assert_eq!(trace.amd_calls,1);let q=std::slice::from_raw_parts((*k.symbolic).Q,n).to_vec();let pp=std::slice::from_raw_parts((*k.symbolic).P,n);assert_eq!(pp,q);if let Some(ref e)=expected{assert_eq!(e,&q);}else{expected=Some(q);}}
 if pair>=10{writeln!(out,"{}",json!({"case":case,"pair":pair-10,"path":if kind==0{"system"}else{"profile_copy"},"j_structure_us":jprep,"analyze_us":total,"amd_us":trace.amd_us,"amd_input_prep_us":if kind==1{Some(trace.prep_us)}else{None},"other_symbolic_us":if kind==1{Some(total-trace.amd_us-trace.prep_us)}else{None},"non_amd_symbolic_us":total-trace.amd_us,"amd_calls":trace.amd_calls,"n":n,"nnz":p.j_row_indices.len()})).unwrap();}
 }}eprintln!("{case} AMD cost done");}
}
