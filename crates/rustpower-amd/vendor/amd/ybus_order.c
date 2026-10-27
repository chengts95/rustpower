/* RustPower Ybus supervariable initialization. AMD core retains its BSD license.
 * No process-global state: each call owns its workspace and Symbolic object. */
#include <stdint.h>
#include <string.h>
#include <amd.h>
#include <klu.h>
#if AMD_MAIN_VERSION == 3 && KLU_MAIN_VERSION == 2
extern void rustpower_amd_super2(int64_t,int64_t*,int64_t*,int64_t*,int64_t,int64_t,int64_t*,int64_t*,int64_t*,int64_t*,int64_t*,int64_t*,int64_t*,double*,double*,int64_t,int64_t);
#endif
int rustpower_ybus_supported(void) {
#if AMD_MAIN_VERSION == 3 && KLU_MAIN_VERSION == 2
    return 1;
#else
    return 0;
#endif
}
klu_l_symbolic *rustpower_ybus_analyze(const int64_t *yp,const int64_t *yi,int64_t active,int64_t npq,klu_l_common *c) {
#if AMD_MAIN_VERSION == 3 && KLU_MAIN_VERSION == 2
    /* Rust has validated sorted, symmetric, diagonal-present, connected input. */
    int64_t n=active+npq,edges=0,nz=0;
    for(int64_t j=0;j<active;j++) {
        int64_t a=0,q=0;
        for(int64_t p=yp[j];p<yp[j+1]&&yi[p]<active;p++) {
            int64_t i=yi[p];a++;if(i<npq)q++;if(i!=j)edges++;
        }
        nz+=(a+q)*(j<npq?2:1);
    }
    c->status=KLU_OK;c->work=0;
    klu_l_symbolic *s=klu_l_malloc(1,sizeof(*s),c);
    if(!s)return NULL;
    memset(s,0,sizeof(*s));s->n=n;s->nz=nz;
    s->P=klu_l_malloc(n,sizeof(int64_t),c);s->Q=klu_l_malloc(n,sizeof(int64_t),c);
    s->R=klu_l_malloc(n+1,sizeof(int64_t),c);s->Lnz=klu_l_malloc(n,sizeof(double),c);
    if(!s->P||!s->Q||!s->R||!s->Lnz){klu_l_free_symbolic(&s,c);return NULL;}
    int64_t iwlen=edges+edges/5+n,worklen=7*n+iwlen;
    int64_t *work=klu_l_malloc(worklen,sizeof(int64_t),c);
    if(!work){klu_l_free_symbolic(&s,c);return NULL;}
    int64_t *pe=work,*len=pe+n,*nv=len+n,*head=nv+n,*elen=head+n,*degree=elen+n,*w=degree+n,*iw=w+n;
    int64_t pos=0;
    for(int64_t j=0;j<n;j++) {
        pe[j]=pos;len[j]=0;degree[j]=0;
        if(j>=active)continue;
        for(int64_t p=yp[j];p<yp[j+1]&&yi[p]<active;p++) {
            int64_t i=yi[p];if(i==j)continue;
            iw[pos++]=i;degree[j]+=i<npq?2:1;
        }
        len[j]=pos-pe[j];
    }
    double info[AMD_INFO];for(int i=0;i<AMD_INFO;i++)info[i]=-1;
    rustpower_amd_super2(n,pe,iw,len,iwlen,edges,nv,s->P,s->Q,head,elen,degree,w,NULL,info,active,npq);
    for(int64_t i=0;i<n;i++)w[i]=0;
    int valid=1;
    for(int64_t i=0;i<n;i++){int64_t k=s->Q[i];if(k<0||k>=n||w[k]){valid=0;break;}w[k]=1;s->P[i]=k;}
    klu_l_free(work,worklen,sizeof(int64_t),c);
    if(!valid||info[AMD_STATUS]!=AMD_OK){klu_l_free_symbolic(&s,c);c->status=KLU_INVALID;return NULL;}
    s->ordering=0;s->do_btf=c->btf!=0;s->structural_rank=s->do_btf?n:-1;
    c->structural_rank=s->structural_rank;s->nblocks=1;s->maxblock=n;s->R[0]=0;s->R[1]=n;s->nzoff=0;
    s->lnz=s->unz=s->Lnz[0]=(int64_t)info[AMD_LNZ]+n;
    s->est_flops=2*info[AMD_NMULTSUBS_LU]+info[AMD_NDIV];s->symmetry=1;
    return s;
#else
    (void)yp;(void)yi;(void)active;(void)npq;(void)c;return NULL;
#endif
}
