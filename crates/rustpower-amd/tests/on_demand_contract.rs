use rustpower_amd::powerflow::YbusPattern;
#[test]
fn unsupported_structure_is_rejected_instead_of_falling_back(){
 let disconnected=(vec![0,1,2,3,4,5],vec![0,1,2,3,4]);
 let missing_diagonals=(vec![0,1,3,5,6,7],vec![1,0,2,1,3,2,4]);
 let asymmetric=(vec![0,2,4,7,9,10],vec![0,1,1,2,1,2,3,2,3,4]);
 let unsorted=(vec![0,2,4,7,9,10],vec![1,0,0,1,1,2,3,2,3,4]);
 for(p,r)in [disconnected,missing_diagonals,asymmetric,unsorted]{assert!(YbusPattern::analyze(&p,&r,4,0).is_err());}
}
