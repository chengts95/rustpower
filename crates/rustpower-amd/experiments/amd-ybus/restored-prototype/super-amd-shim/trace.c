#define _GNU_SOURCE
#include "profile.h"
#include <dlfcn.h>
#include <amd.h>
rp_profile_stats rp_stats={0};
void rp_trace_reset(void){rp_stats=(rp_profile_stats){0};}
void rp_trace_get(rp_profile_stats *s){*s=rp_stats;}
int amd_l_order(int64_t n,const int64_t *ap,const int64_t *ai,int64_t *p,double *control,double *info){
 typedef int(*fn)(int64_t,const int64_t*,const int64_t*,int64_t*,double*,double*);static fn real=NULL;if(!real)real=(fn)dlsym(RTLD_NEXT,"amd_l_order");double t=rp_now();int r=real(n,ap,ai,p,control,info);rp_stats.amd_us+=rp_now()-t;rp_stats.amd_calls++;return r;
}
size_t amd_l_aat(int64_t n,const int64_t *ap,const int64_t *ai,int64_t *len,int64_t *tp,double *info){
 typedef size_t(*fn)(int64_t,const int64_t*,const int64_t*,int64_t*,int64_t*,double*);static fn f=NULL;if(!f)f=(fn)dlsym(RTLD_NEXT,"amd_l_aat");double t=rp_now();size_t r=f(n,ap,ai,len,tp,info);rp_stats.aat_us+=rp_now()-t;return r;
}
void amd_l1(int64_t n,const int64_t *ap,const int64_t *ai,int64_t *p,int64_t *pinv,int64_t *len,int64_t slen,int64_t *s,double *control,double *info){
 typedef void(*fn)(int64_t,const int64_t*,const int64_t*,int64_t*,int64_t*,int64_t*,int64_t,int64_t*,double*,double*);static fn f=NULL;if(!f)f=(fn)dlsym(RTLD_NEXT,"amd_l1");double t=rp_now();f(n,ap,ai,p,pinv,len,slen,s,control,info);rp_stats.amd1_us+=rp_now()-t;
}
void amd_l2(int64_t n,int64_t *pe,int64_t *iw,int64_t *len,int64_t iwlen,int64_t pfree,int64_t *nv,int64_t *next,int64_t *last,int64_t *head,int64_t *elen,int64_t *degree,int64_t *w,double *control,double *info){
 typedef void(*fn)(int64_t,int64_t*,int64_t*,int64_t*,int64_t,int64_t,int64_t*,int64_t*,int64_t*,int64_t*,int64_t*,int64_t*,int64_t*,double*,double*);static fn f=NULL;if(!f)f=(fn)dlsym(RTLD_NEXT,"amd_l2");double t=rp_now();f(n,pe,iw,len,iwlen,pfree,nv,next,last,head,elen,degree,w,control,info);rp_stats.elim_us+=rp_now()-t;rp_stats.elim_calls++;
}
