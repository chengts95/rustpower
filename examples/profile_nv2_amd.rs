//! Original scalar J->AMD only. Network and J are prepared before the perf gate.
use rustpower::{io::pandapower::{Network,load_csv_zip},prelude::*,prelude::{ecs::powerflow::systems::PowerFlowMat,jacobian_cache::JacobianCache},testcases::case_ieee39::IEEE_39};
use rustpower_amd::{Control,powerflow::{YbusPattern,OrderingWorkspace}};
use std::{hint::black_box,fs::{File,OpenOptions},io::{Write,BufRead,BufReader},time::Instant};
unsafe extern "C" {fn sched_getcpu()->i32;}
fn gate(command:&str){
 if let Ok(path)=std::env::var("PERF_CTL"){
  let mut ctl=OpenOptions::new().write(true).open(path).unwrap();
  let mut ack=BufReader::new(File::open(std::env::var("PERF_ACK").unwrap()).unwrap());
  writeln!(ctl,"{command}").unwrap();ctl.flush().unwrap();let mut reply=String::new();ack.read_line(&mut reply).unwrap();assert!(reply.starts_with("ack"));
 }
}
fn main(){
 let a:Vec<_>=std::env::args().collect();let case=&a[1];let reps:usize=a[2].parse().unwrap();let out=&a[3];let variant=&a[4];assert!(variant=="scalar"||variant=="nv2");
 let net=match case.as_str(){"39"=>serde_json::from_str::<Network>(IEEE_39).unwrap(),"118"=>load_csv_zip("cases/IEEE118/data.zip").unwrap(),"9241"=>load_csv_zip("cases/pegase9241/data.zip").unwrap(),_=>panic!()};
 let mut grid=PowerGrid::default();grid.world_mut().insert_resource(PPNetwork(net));grid.init_pf_net();
 let m=grid.world().resource::<PowerFlowMat>();
 let j=JacobianCache::build_from_permuted(m.y_bus.col_offsets(),m.y_bus.row_indices(),m.npv,m.npq);
 let n=j.j_col_ptrs.len()-1;let ctl=Control::default();
 let pattern=YbusPattern::analyze(m.y_bus.col_offsets(),m.y_bus.row_indices(),m.npv,m.npq).unwrap();
 let call=||{if variant=="scalar" {unsafe{rustpower_amd::order_unchecked(n,&j.j_col_ptrs,&j.j_row_indices,&ctl)}.unwrap()}else{unsafe{OrderingWorkspace::new().order_unchecked(&pattern,&ctl)}.unwrap().into_parts()}};
 let reference=call();
 let scalar_reference=unsafe{rustpower_amd::order_unchecked(n,&j.j_col_ptrs,&j.j_row_indices,&ctl)}.unwrap();
 let permutation_differences=reference.0.iter().zip(&scalar_reference.0).filter(|(a,b)|a!=b).count();
 for (k,&i) in reference.0.iter().enumerate(){assert_eq!(reference.1[i],k);}
 let mut seconds=Vec::with_capacity(reps);#[allow(unused_mut)] let mut kernel_seconds:Vec<f64>=Vec::with_capacity(reps);let cpu=unsafe{sched_getcpu()};let mut last=None;
 for k in 0..100+reps{
  if k==100{gate("enable");}
  let t=Instant::now();
  let r=black_box(&call)();
  black_box(&r.0);black_box(&r.1);let elapsed=t.elapsed().as_secs_f64();if k>=100{seconds.push(elapsed);}
  #[cfg(kernel_timing)] {let kt=rustpower_amd::kernel_profile::take().unwrap();if k>=100{kernel_seconds.push(kt);}}
  last=Some(r);
 }
 gate("disable");assert_eq!(cpu,unsafe{sched_getcpu()});let r=last.unwrap();assert_eq!(r.0,reference.0);assert_eq!(r.1,reference.1);assert_eq!(r.2.lnz,reference.2.lnz);
 std::fs::write(out,serde_json::to_vec(&serde_json::json!({"case":case,"variant":variant,"function":if variant=="scalar"{"order_unchecked(J)"}else{"OrderingWorkspace::order_unchecked(analyzed Ybus)"},"input":if variant=="scalar"{"prebuilt J CSC"}else{"preanalyzed YbusPattern"},"npq":m.npq,"npv":m.npv,"initial_representatives":if variant=="scalar"{n}else{m.npq+m.npv},"initial_adjacency_entries":if variant=="scalar"{j.j_row_indices.len()-n}else{pattern.columns().iter().map(|c|c.active_count-1).sum::<usize>()},"kernel_seconds":kernel_seconds,"permutation_differences_from_scalar":permutation_differences,"scalar_lnz":scalar_reference.2.lnz,"scalar_variables":n,"j_nnz":j.j_row_indices.len(),"cpu":cpu,"reps":reps,"warmup":100,"seconds":seconds,"lnz":r.2.lnz,"permutation_exact":true,"scope":"input already prepared: full entry clock separately from amd_2 kernel-only diagnostic; fresh work arrays every call"})).unwrap()).unwrap();
}
