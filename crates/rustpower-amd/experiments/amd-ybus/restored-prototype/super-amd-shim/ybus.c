#include <stdint.h>
#include <stdlib.h>
#include <amd.h>
#include <klu.h>
#include "profile.h"
extern klu_l_symbolic *klu_l_alloc_symbolic(int64_t,int64_t*,int64_t*,klu_l_common*);
extern void rp_super2(int64_t,int64_t*,int64_t*,int64_t*,int64_t,int64_t,int64_t*,int64_t*,int64_t*,int64_t*,int64_t*,int64_t*,int64_t*,double*,double*,int64_t,int64_t);
static const int64_t *yp,*yi;static int64_t active,npq;
void rp_set_ybus(const int64_t *p,const int64_t *i,int64_t a,int64_t q){yp=p;yi=i;active=a;npq=q;}
klu_l_symbolic *rp_ybus_analyze(int64_t n,int64_t *ap,int64_t *ai,klu_l_common *c,int variant){
 if(c->btf||c->ordering||n!=active+npq){c->status=KLU_INVALID;return NULL;}
 c->status=KLU_OK;c->structural_rank=-1;c->work=0;klu_l_symbolic *s=klu_l_alloc_symbolic(n,ap,ai,c);if(!s)return NULL;
 s->ordering=0;s->do_btf=0;s->structural_rank=-1;s->nblocks=1;s->maxblock=n;s->R[0]=0;s->R[1]=n;s->nzoff=0;
 double begin=rp_now();int64_t edges=0;
 if(variant==1) edges=ap[n]-n;
 else for(int64_t j=0;j<active;j++)for(int64_t p=yp[j];p<yp[j+1]&&yi[p]<active;p++)if(yi[p]!=j)edges++;
 int64_t iwlen=edges+edges/5+n;int64_t *work=malloc((size_t)(7*n+iwlen)*8);
 if(!work){klu_l_free_symbolic(&s,c);c->status=KLU_OUT_OF_MEMORY;return NULL;}
 int64_t *pe=work,*len=pe+n,*nv=len+n,*head=nv+n,*elen=head+n,*deg=elen+n,*w=deg+n,*iw=w+n;
 int64_t pos=0;
 for(int64_t j=0;j<n;j++){
  pe[j]=pos;int64_t bus=j<active?j:j-active;
  if(variant==2&&j>=active){len[j]=0;deg[j]=0;continue;}
  int64_t extdeg=0;
  for(int64_t p=yp[bus];p<yp[bus+1]&&yi[p]<active;p++){int64_t i=yi[p];if(variant==1){if(i!=j)iw[pos++]=i;}else if(i!=bus){iw[pos++]=i;extdeg+=i<npq?2:1;}}
  if(variant==1)for(int64_t p=yp[bus];p<yp[bus+1]&&yi[p]<npq;p++){int64_t i=active+yi[p];if(i!=j)iw[pos++]=i;}
  len[j]=pos-pe[j];deg[j]=variant==2?extdeg:len[j];
 }
 if(pos!=edges){free(work);klu_l_free_symbolic(&s,c);c->status=KLU_INVALID;return NULL;}
 double prep=rp_now()-begin;rp_stats.direct_prep_us+=prep;rp_stats.edges=edges;rp_stats.workspace_bytes=(7*n+iwlen)*8;
 double info[AMD_INFO];for(int i=0;i<AMD_INFO;i++)info[i]=-1;
 info[AMD_N]=n;info[AMD_NZ]=ap[n];info[AMD_NZDIAG]=n;info[AMD_SYMMETRY]=1;info[AMD_NZ_A_PLUS_AT]=ap[n]-n;info[AMD_MEMORY]=(7*n+iwlen)*8;
 double t=rp_now();
 if(variant==1)amd_l2(n,pe,iw,len,iwlen,edges,nv,s->P,s->Q,head,elen,deg,w,NULL,info);
 else {rp_super2(n,pe,iw,len,iwlen,edges,nv,s->P,s->Q,head,elen,deg,w,NULL,info,active,npq);rp_stats.elim_us+=rp_now()-t;rp_stats.super_calls++;}
 rp_stats.amd_us+=prep+(rp_now()-t); // whole direct preparation+core time
 for(int64_t i=0;i<n;i++)s->P[i]=s->Q[i];
 // Validate a full scalar permutation produced by AMD's original postorder.
 for(int64_t i=0;i<n;i++)w[i]=0;for(int64_t i=0;i<n;i++){int64_t k=s->Q[i];if(k<0||k>=n||w[k]){free(work);klu_l_free_symbolic(&s,c);c->status=KLU_INVALID;return NULL;}w[k]=1;}
 s->lnz=s->unz=s->Lnz[0]=(int64_t)info[AMD_LNZ]+n;s->est_flops=2*info[AMD_NMULTSUBS_LU]+info[AMD_NDIV];s->symmetry=1;
 size_t peak=c->memusage+(size_t)info[AMD_MEMORY];if(peak>c->mempeak)c->mempeak=peak;
 free(work);return s;
}
