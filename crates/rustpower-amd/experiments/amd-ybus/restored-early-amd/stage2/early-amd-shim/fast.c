#include <stdint.h>
#include <amd.h>
#include <klu.h>
// Isolated BTF=0 specialization of SuiteSparse KLU2.3.6 symbolic setup.
// Uses the exact J CSC generated from Ybus, avoiding Pbtf/Qbtf/Pinv/Cp/Ci/Pblk copies.
extern klu_l_symbolic *klu_l_alloc_symbolic(int64_t,int64_t*,int64_t*,klu_l_common*);
klu_l_symbolic *rp_early_analyze(int64_t n,int64_t *ap,int64_t *ai,klu_l_common *c){
 if(c->btf!=0||c->ordering!=0||n<=3){c->status=KLU_INVALID;return NULL;}
 c->status=KLU_OK;c->structural_rank=-1;c->work=0;
 klu_l_symbolic *s=klu_l_alloc_symbolic(n,ap,ai,c);if(!s)return NULL;
 s->ordering=0;s->do_btf=0;s->structural_rank=-1;s->nblocks=1;s->maxblock=n;s->R[0]=0;s->R[1]=n;s->nzoff=0;
 double info[AMD_INFO];int result=amd_l_order(n,ap,ai,s->Q,NULL,info);
 if(result<AMD_OK){klu_l_free_symbolic(&s,c);c->status=result==AMD_OUT_OF_MEMORY?KLU_OUT_OF_MEMORY:KLU_INVALID;return NULL;}
 for(int64_t i=0;i<n;i++)s->P[i]=s->Q[i];
 s->lnz=s->unz=s->Lnz[0]=(int64_t)info[AMD_LNZ]+n;
 s->est_flops=2*info[AMD_NMULTSUBS_LU]+info[AMD_NDIV];s->symmetry=info[AMD_SYMMETRY];
 size_t peak=c->memusage+(size_t)info[AMD_MEMORY];if(peak>c->mempeak)c->mempeak=peak;
 return s;
}
