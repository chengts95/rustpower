//! Diagnostic driver. Parsing/warmup/serialization are outside the enabled perf region.
use rustpower::{io::pandapower::{Network, load_csv_zip}, prelude::*, testcases::case_ieee39::IEEE_39};
use rustpower::prelude::ecs::powerflow::systems::PowerFlowMat;
use rustpower_amd::{Control,powerflow::YbusPattern,on_demand::{FullTableControl,Workspace}};
use std::{hint::black_box, time::Instant,io::{Write,BufRead,BufReader},fs::{File,OpenOptions}};
unsafe extern "C" {fn sched_getcpu()->i32;}
fn gate(command:&str) {
    if let Ok(path)=std::env::var("PERF_CTL") {
        let mut ctl=OpenOptions::new().write(true).open(path).unwrap();
        let mut ack=BufReader::new(File::open(std::env::var("PERF_ACK").unwrap()).unwrap());
        writeln!(ctl,"{command}").unwrap();ctl.flush().unwrap();
        let mut reply=String::new(); ack.read_line(&mut reply).unwrap();assert!(reply.starts_with("ack"),"{reply}");
    }
}
fn main(){
 let args:Vec<_>=std::env::args().collect();
 let case=&args[1];let variant=&args[2];let mode=&args[3];let reps:usize=args[4].parse().unwrap();let output=&args[5];
 let net=match case.as_str(){"39"=>serde_json::from_str::<Network>(IEEE_39).unwrap(),"118"=>load_csv_zip("cases/IEEE118/data.zip").unwrap(),"9241"=>load_csv_zip("cases/pegase9241/data.zip").unwrap(),_=>panic!()};
 let mut grid=PowerGrid::default();grid.world_mut().insert_resource(PPNetwork(net));grid.init_pf_net();
 let m=grid.world().resource::<PowerFlowMat>();let control=Control::default();
 let mut full=FullTableControl::default();let mut lazy=Workspace::default();
 let view=YbusPattern::analyze(m.y_bus.col_offsets(),m.y_bus.row_indices(),m.npv,m.npq).unwrap();
 let reference=full.order(&view,&control);let expected=reference.permutation;let inverse=reference.inverse;let lnz=reference.info.lnz;
 let mut times=Vec::with_capacity(reps);let mut phases:Vec<[f64;4]>=Vec::with_capacity(reps);let mut last=None;
 let cpu=unsafe{sched_getcpu()};
 for i in 0..100+reps {
   if i==100 {gate("enable");}
   let start=Instant::now();
   let pattern=YbusPattern::analyze(black_box(m.y_bus.col_offsets()),black_box(m.y_bus.row_indices()),m.npv,m.npq).unwrap();
   #[cfg(feature="experimental_amd_profile")]
   let analyzed=start.elapsed().as_secs_f64();
   let r=if variant=="full" {
       #[cfg(feature="experimental_amd_profile")]
       {let (r,p)=if mode=="fresh"{FullTableControl::default().order_profiled(&pattern,&control)}else{full.order_profiled(&pattern,&control)};if i>=100 {phases.push([analyzed,p[0],p[1],p[2]]);}r}
       #[cfg(not(feature="experimental_amd_profile"))]
       {if mode=="fresh"{FullTableControl::default().order(&pattern,&control)}else{full.order(&pattern,&control)}}
   }else{if mode=="fresh"{rustpower_amd::on_demand::order(&pattern,&control).unwrap()}else{lazy.order(&pattern,&control).unwrap()}};
   drop(pattern);
   black_box(&r.permutation);black_box(&r.inverse);
   let elapsed=start.elapsed().as_secs_f64();
   if i>=100{times.push(elapsed);} last=Some(r);
 }
 gate("disable");assert_eq!(cpu,unsafe{sched_getcpu()});
 let r=last.unwrap();assert_eq!(r.permutation,expected);assert_eq!(r.inverse,inverse);assert_eq!(r.info.lnz,lnz);
 let data=serde_json::json!({"case":case,"variant":variant,"mode":mode,"reps":reps,"cpu":cpu,"seconds":times,"phases_seconds":phases,"phase_names":["ybus_analysis","buffer_resize","initial_table","amd_kernel_and_postorder"],"lnz":lnz,"arena_compactions":r.info.n_cmp_a,"permutation_exact":true,"warmup":100});
 std::fs::write(output,serde_json::to_vec(&data).unwrap()).unwrap();
}
