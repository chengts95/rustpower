//! Original scalar J->AMD only. Network and J are prepared before the perf gate.
use rustpower::{io::pandapower::{Network,load_csv_zip},prelude::*,prelude::{ecs::powerflow::systems::PowerFlowMat,jacobian_cache::JacobianCache},testcases::case_ieee39::IEEE_39};
use rustpower_amd::Control;
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
 let a:Vec<_>=std::env::args().collect();let case=&a[1];let reps:usize=a[2].parse().unwrap();let out=&a[3];
 let net=match case.as_str(){"39"=>serde_json::from_str::<Network>(IEEE_39).unwrap(),"118"=>load_csv_zip("cases/IEEE118/data.zip").unwrap(),"9241"=>load_csv_zip("cases/pegase9241/data.zip").unwrap(),_=>panic!()};
 let mut grid=PowerGrid::default();grid.world_mut().insert_resource(PPNetwork(net));grid.init_pf_net();
 let m=grid.world().resource::<PowerFlowMat>();
 let j=JacobianCache::build_from_permuted(m.y_bus.col_offsets(),m.y_bus.row_indices(),m.npv,m.npq);
 let n=j.j_col_ptrs.len()-1;let ctl=Control::default();
 let reference=unsafe{rustpower_amd::order_unchecked(n,&j.j_col_ptrs,&j.j_row_indices,&ctl)}.unwrap();
 let mut seconds=Vec::with_capacity(reps);let cpu=unsafe{sched_getcpu()};let mut last=None;
 for k in 0..100+reps{
  if k==100{gate("enable");}
  let t=Instant::now();
  let r=unsafe{rustpower_amd::order_unchecked(black_box(n),black_box(&j.j_col_ptrs),black_box(&j.j_row_indices),black_box(&ctl))}.unwrap();
  black_box(&r.0);black_box(&r.1);let elapsed=t.elapsed().as_secs_f64();if k>=100{seconds.push(elapsed);}last=Some(r);
 }
 gate("disable");assert_eq!(cpu,unsafe{sched_getcpu()});let r=last.unwrap();assert_eq!(r.0,reference.0);assert_eq!(r.1,reference.1);assert_eq!(r.2.lnz,reference.2.lnz);
 std::fs::write(out,serde_json::to_vec(&serde_json::json!({"case":case,"function":"rustpower_amd::order_unchecked","input":"prebuilt J CSC","scalar_variables":n,"j_nnz":j.j_row_indices.len(),"cpu":cpu,"reps":reps,"warmup":100,"seconds":seconds,"lnz":r.2.lnz,"permutation_exact":true,"scope":"validate_csc -> order_impl -> valid -> AAT -> amd_1 -> scalar amd_2 -> postorder -> P/Pinv"})).unwrap()).unwrap();
}
