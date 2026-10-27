#define _GNU_SOURCE
#include "profile.h"
#include <dlfcn.h>
#include <amd.h>
#include <klu.h>
rp_profile_stats rp_stats={0};
static double analysis_us;
void rp_trace_reset(void){rp_stats=(rp_profile_stats){0};analysis_us=0;}
void rp_trace_get(rp_profile_stats *s){*s=rp_stats;}
int amd_l_order(int64_t n,const int64_t *ap,const int64_t *ai,int64_t *p,double *control,double *info){
 typedef int(*fn)(int64_t,const int64_t*,const int64_t*,int64_t*,double*,double*);
 static fn real=NULL;if(!real)real=(fn)dlsym(RTLD_NEXT,"amd_l_order");double t=rp_now();int result=real(n,ap,ai,p,control,info);rp_stats.amd_us+=rp_now()-t;rp_stats.amd_calls++;return result;
}
double rp_klu_analysis_us(void){return analysis_us;}
klu_l_symbolic *klu_l_analyze(int64_t n,int64_t *ap,int64_t *ai,klu_l_common *c){
 typedef klu_l_symbolic*(*fn)(int64_t,int64_t*,int64_t*,klu_l_common*);static fn real=NULL;if(!real)real=(fn)dlsym(RTLD_NEXT,"klu_l_analyze");double t=rp_now();klu_l_symbolic*s=real(n,ap,ai,c);analysis_us+=rp_now()-t;return s;
}
