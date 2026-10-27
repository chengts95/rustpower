//! Exact early J-pattern AMD from permuted Ybus; isolated BTF=0 experiment. No production code or defaults are changed.
use std::{hint::black_box, time::Instant, fs::File, io::{Write, BufWriter}};
use rustpower::{prelude::*, io::pandapower::{Network, load_csv_zip}, testcases::case_ieee39::IEEE_39};
use rustpower::prelude::{ecs::powerflow::systems::PowerFlowMat, newtonpf::NewtonCache, solver::Solve};
use rustpower_sol_klu::KLUSolver as RawKlu;
use serde_json::{json,Value};
const TOL:f64=1e-10;
const MAX_ITER:usize=100;
use rustpower::prelude::jacobian_cache::JacobianCache;
use nalgebra::DVector;
#[repr(C)]#[derive(Default)]struct Trace {amd_us:f64,prep_us:f64,amd_calls:u64,aat_us:f64,amd1_us:f64,elim_us:f64,direct_prep_us:f64,elim_calls:u64,super_calls:u64,edges:u64,workspace_bytes:u64}
#[link(name="rpsuperamd")]unsafe extern "C"{
 fn rp_early_analyze(n:i64,ap:*const usize,ai:*const usize,c:*mut std::ffi::c_void)->*mut std::ffi::c_void;
 fn rp_ybus_analyze(n:i64,ap:*const usize,ai:*const usize,c:*mut std::ffi::c_void,path:i32)->*mut std::ffi::c_void;
 fn rp_set_ybus(p:*const usize,i:*const usize,active:i64,npq:i64);
 fn rp_trace_reset();fn rp_trace_get(t:*mut Trace);
}
fn early_cache(m:&PowerFlowMat)->NewtonCache{
 let mut c=NewtonCache::default();let p=JacobianCache::build_from_permuted(m.y_bus.col_offsets(),m.y_bus.row_indices(),m.npv,m.npq);
 c.j_values=vec![0.;p.j_row_indices.len()];c.j_pattern=Some(p);c.npv=m.npv;c.npq=m.npq;
 c.ibus=DVector::zeros(m.v_bus_init.len());c.F=DVector::zeros(m.npv+2*m.npq);c.s_calc=DVector::zeros(m.v_bus_init.len());c
}
unsafe fn early(k:&mut RawKlu,c:&NewtonCache,path:usize){let p=c.j_pattern.as_ref().unwrap();k.symbolic=if path==0{rp_early_analyze((p.j_col_ptrs.len()-1) as i64,p.j_col_ptrs.as_ptr(),p.j_row_indices.as_ptr(),k.common.cast())}else{rp_ybus_analyze((p.j_col_ptrs.len()-1) as i64,p.j_col_ptrs.as_ptr(),p.j_row_indices.as_ptr(),k.common.cast(),path as i32)}.cast();assert!(!k.symbolic.is_null());}
fn audit(st:&mut Value,k:&RawKlu,expected:&[i64],cold:bool,path:usize){unsafe{
 let q=std::slice::from_raw_parts((*k.symbolic).Q,expected.len());let p=std::slice::from_raw_parts((*k.symbolic).P,expected.len());assert_eq!(p,q);if path<=1{assert_eq!(q,expected);}
 let mut t=Trace::default();rp_trace_get(&mut t);assert_eq!(t.amd_calls,if cold&&path==0{1}else{0});assert_eq!(t.elim_calls+t.super_calls,if cold{1}else{0});st["amd_calls"]=json!(t.amd_calls);st["amd_us"]=json!(t.amd_us);st["identical_permutation"]=json!(q==expected);st["aat_us"]=json!(t.aat_us);st["amd1_graph_us"]=if path==0{json!(t.amd1_us-t.elim_us)}else{Value::Null};st["elim_us"]=json!(t.elim_us);st["direct_prep_us"]=json!(t.direct_prep_us);st["elim_calls"]=json!(t.elim_calls);st["super_calls"]=json!(t.super_calls);st["graph_edges"]=json!(t.edges);st["amd_workspace_bytes"]=json!(t.workspace_bytes);
}}

fn configure(k:&mut RawKlu,_path:i32){unsafe{(*k.common).btf=0;}}
fn klu_stats(k:&RawKlu)->Value {unsafe{
    assert!(!k.symbolic.is_null());assert!(!k.numeric.is_null());
    let c=&*k.common;let s=&*k.symbolic;let n=&*k.numeric;
    json!({"common_btf":c.btf,"do_btf":s.do_btf,"ordering":c.ordering,"symbolic_ordering":s.ordering,
        "tol":c.tol,"scale":c.scale,"maxwork":c.maxwork,"mempeak":c.mempeak,"memusage":c.memusage,"status":c.status,"nblocks":s.nblocks,
        "maxblock":s.maxblock,"n":s.n,"jacobian_nnz":s.nz,"lnz":n.lnz,"unz":n.unz,"nzoff":n.nzoff,
        "fill_unique":n.lnz+n.unz-s.n+n.nzoff})}}
fn native(mat:&PowerFlowMat,btf:i32)->NewtonSolver {
    let mut s=NewtonSolver::new();s.mat=Some(mat.clone());configure(&mut s.linear_solver.0,btf);s
}
fn error(a:&[num_complex::Complex64],b:&[num_complex::Complex64])->f64 {
    a.iter().zip(b).map(|(a,b)|(*a-*b).norm()).fold(0.,f64::max)
}
#[derive(Default)]
struct Stages {analyze:f64,factor:f64,refactor:f64,triangular:f64,analyze_n:usize,factor_n:usize,refactor_n:usize,triangular_n:usize,fallback_n:usize}
struct TimedKlu {inner:RawKlu,t:Stages,path:usize}
impl TimedKlu {fn new(btf:i32)->Self{let mut inner=RawKlu::default();configure(&mut inner,btf);Self{inner,t:Stages::default(),path:btf as usize}}}
impl Solve for TimedKlu {
 fn solve(&mut self,ap:&mut[usize],ai:&mut[usize],ax:&mut[f64],b:&mut[f64],n:usize)->Result<(),&'static str>{unsafe{
    // Mirrors src/basic/solver/klu.rs exactly; only Instant timing is added.
    if self.inner.symbolic.is_null(){let t=Instant::now();self.inner.symbolic=if self.path==0{rp_early_analyze(n as i64,ap.as_ptr(),ai.as_ptr(),self.inner.common.cast())}else{rp_ybus_analyze(n as i64,ap.as_ptr(),ai.as_ptr(),self.inner.common.cast(),self.path as i32)}.cast();assert!(!self.inner.symbolic.is_null());self.t.analyze+=t.elapsed().as_secs_f64()*1e6;self.t.analyze_n+=1;}
    let mut ret=if self.inner.numeric.is_null(){let t=Instant::now();let r=self.inner.factor(ap.as_mut_ptr() as *mut i64,ai.as_mut_ptr() as *mut i64,ax.as_mut_ptr());self.t.factor+=t.elapsed().as_secs_f64()*1e6;self.t.factor_n+=1;r}
    else {let t=Instant::now();let status=self.inner.refactor(ap.as_mut_ptr() as *mut i64,ai.as_mut_ptr() as *mut i64,ax.as_mut_ptr(),n as i64);self.t.refactor+=t.elapsed().as_secs_f64()*1e6;self.t.refactor_n+=1;
      if status!=0 {self.t.fallback_n+=1;let t=Instant::now();let r=self.inner.factor(ap.as_mut_ptr() as *mut i64,ai.as_mut_ptr() as *mut i64,ax.as_mut_ptr());self.t.factor+=t.elapsed().as_secs_f64()*1e6;self.t.factor_n+=1;r}else{0}};
    let t=Instant::now();ret|=self.inner.solve(b.as_mut_ptr(),n as i64,1);self.t.triangular+=t.elapsed().as_secs_f64()*1e6;self.t.triangular_n+=1;
    if ret!=0{return Err("error occurred when calling KLU routines!");}
 }Ok(())}
 fn reset(&mut self){let btf=unsafe{(*self.inner.common).btf};self.inner.reset();configure(&mut self.inner,btf);}
}
fn run(case:&str,mat:&PowerFlowMat,reps:usize,w:&mut impl Write){
    let mut reference=native(mat,1);assert!(reference.solve(MAX_ITER,TOL).unwrap().0);let expected=reference.result.as_ref().unwrap().v.as_slice().to_vec();
    let expected_iter=reference.iterations().unwrap();
 unsafe{rp_set_ybus(mat.y_bus.col_offsets().as_ptr(),mat.y_bus.row_indices().as_ptr(),(mat.npq+mat.npv) as i64,mat.npq as i64);}
 let m=mat.npq+mat.npv;
 for j in 0..m {for p in mat.y_bus.col_offsets()[j]..mat.y_bus.col_offsets()[j+1]{let i=mat.y_bus.row_indices()[p];if i<m{assert!(mat.y_bus.row_indices()[mat.y_bus.col_offsets()[i]..mat.y_bus.col_offsets()[i+1]].binary_search(&j).is_ok(),"Ybus pattern must be symmetric");}}}
 let jp=reference.cache.j_pattern.as_ref().unwrap();
 for k in 0..mat.npq {assert_eq!(&jp.j_row_indices[jp.j_col_ptrs[k]..jp.j_col_ptrs[k+1]],&jp.j_row_indices[jp.j_col_ptrs[m+k]..jp.j_col_ptrs[m+k+1]],"true-twin closed neighborhoods");}
 eprintln!("Verified {} PQ true-twin pairs and symmetric Y/J input",mat.npq);
 let expected_perm=unsafe{std::slice::from_raw_parts((*reference.linear_solver.0.symbolic).Q,mat.npv+2*mat.npq).to_vec()};
 let base_pattern=reference.cache.j_pattern.as_ref().unwrap();let exact=early_cache(mat);let exact_pattern=exact.j_pattern.as_ref().unwrap();assert_eq!(base_pattern.j_col_ptrs,exact_pattern.j_col_ptrs);assert_eq!(base_pattern.j_row_indices,exact_pattern.j_row_indices);
 let pattern_bytes=(exact_pattern.j_col_ptrs.len()+exact_pattern.j_row_indices.len()+exact_pattern.pq_ends.len()+exact_pattern.active_ends.len()+exact_pattern.diag_ptrs.len())*8;
 eprintln!("{case}: exact symbolic J equality confirmed; shared cache bytes={pattern_bytes}; extra retained graph bytes=0");
    eprintln!("{case}: {} buses, {} NR iterations; benchmarking {reps} alternating pairs per mode",mat.v_bus_init.len(),expected_iter);
    // Each solve uses the same untouched mat.v_bus_init. Reuse means factor/cache reuse, not a converged-voltage start.
    for mode in ["cold","reuse"] {
        let mut solvers=[native(mat,0),native(mat,1),native(mat,2)];
        for p in 0..(reps+6){
            let order=match p%6{0=>[0usize,1,2],1=>[2,1,0],2=>[1,2,0],3=>[0,2,1],4=>[2,0,1],_=>[1,0,2]};
            for idx in order{
                if mode=="cold"{solvers[idx]=native(mat,idx as i32);}
                let s=&mut solvers[idx];unsafe{rp_trace_reset();}let t=Instant::now();if s.linear_solver.0.symbolic.is_null(){s.cache=early_cache(mat);unsafe{early(&mut s.linear_solver.0,&s.cache,idx);}}let prepared_ptr=if true{Some(s.cache.j_pattern.as_ref().unwrap().j_row_indices.as_ptr())}else{None};let (ok,residual)=black_box(s.solve(MAX_ITER,TOL).unwrap());let us=t.elapsed().as_secs_f64()*1e6;if let Some(ptr)=prepared_ptr{assert_eq!(ptr,s.cache.j_pattern.as_ref().unwrap().j_row_indices.as_ptr(),"J pattern must be reused without rebuilding");}
                let err=error(s.result.as_ref().unwrap().v.as_slice(),&expected);
                assert!(ok&&residual<TOL&&err<1e-7);assert_eq!(s.iterations().unwrap(),expected_iter);
                let mut st=klu_stats(&s.linear_solver.0);assert_eq!(st["do_btf"],0);if p>=6{audit(&mut st,&s.linear_solver.0,&expected_perm,mode=="cold",idx);}st["pattern_bytes"]=json!(pattern_bytes);st["extra_retained_graph_bytes"]=json!(0);
                if p>=6 {writeln!(w,"{}",json!({"kind":"full","case":case,"mode":mode,"pair":p-6,"position":order.iter().position(|&x|x==idx).unwrap(),"path":idx,"us":us,"iterations":s.iterations().unwrap(),"converged":ok,"residual":residual,"max_voltage_error":err,"stats":st})).unwrap();}
            }
        }
    }
    // Separate instrumented runs: stage timing overhead is excluded from the full-solve comparison above.
    for mode in ["cold","reuse"] {
      let mut solvers=[TimedKlu::new(0),TimedKlu::new(1),TimedKlu::new(2)];let mut caches=[NewtonCache::default(),NewtonCache::default(),NewtonCache::default()];
      for p in 0..(reps+6){let order=match p%6{0=>[0usize,1,2],1=>[2,1,0],2=>[1,2,0],3=>[0,2,1],4=>[2,0,1],_=>[1,0,2]};for idx in order{
        if mode=="cold"{solvers[idx]=TimedKlu::new(idx as i32);caches[idx]=NewtonCache::default();}
        let s=&mut solvers[idx];s.t=Stages::default();unsafe{rp_trace_reset();}
 let prep_start=Instant::now();if s.inner.symbolic.is_null(){caches[idx]=early_cache(mat);}let prepare_us=prep_start.elapsed().as_secs_f64()*1e6;if s.inner.symbolic.is_null(){let t=Instant::now();unsafe{early(&mut s.inner,&caches[idx],idx);}s.t.analyze+=t.elapsed().as_secs_f64()*1e6;s.t.analyze_n+=1;}
        let r=newton_pf(&mat.y_bus,&mat.s_bus,&mat.v_bus_init,mat.npv,mat.npq,Some(TOL),Some(MAX_ITER),s,Some(&mut caches[idx]));
        let(v,it)=r.unwrap();assert_eq!(it,expected_iter);let err=error(v.as_slice(),&expected);assert!(err<1e-7);
        let mut st=klu_stats(&s.inner);assert_eq!(st["do_btf"],0);if p>=6{audit(&mut st,&s.inner,&expected_perm,mode=="cold",idx);}
        if p>=6 {writeln!(w,"{}",json!({"kind":"stages","prepare_us":prepare_us,"case":case,"mode":mode,"pair":p-6,"path":idx,"analyze_us":s.t.analyze,"factor_us":s.t.factor,"refactor_us":s.t.refactor,"triangular_us":s.t.triangular,"analyze_n":s.t.analyze_n,"factor_n":s.t.factor_n,"refactor_n":s.t.refactor_n,"triangular_n":s.t.triangular_n,"fallback_n":s.t.fallback_n,"max_voltage_error":err,"stats":st})).unwrap();}
      }}
    }
    w.flush().unwrap();
}
fn main(){
    let args:Vec<_>=std::env::args().collect();let reps=args.get(1).map(|x|x.parse().unwrap()).unwrap_or(100usize);let path=args.get(2).map(|s|s.as_str()).unwrap_or("btf-results.jsonl");
    let mut w=BufWriter::new(File::create(path).unwrap());
    for (case,net) in [("PEGASE9241",load_csv_zip("cases/pegase9241/data.zip").unwrap())]{
        let mut grid=PowerGrid::default();grid.world_mut().insert_resource(PPNetwork(net));grid.init_pf_net();
        let mat=grid.world().resource::<PowerFlowMat>().clone();run(case,&mat,reps,&mut w);
    }
}
