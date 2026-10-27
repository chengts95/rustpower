#define _GNU_SOURCE
#include "profile.h"
#include <dlfcn.h>
#include <amd.h>
#include <klu.h>
rp_profile_stats rp_stats={0};
void rp_trace_reset(void){rp_stats=(rp_profile_stats){0};}
void rp_trace_get(rp_profile_stats *s){*s=rp_stats;}
int amd_l_order(int64_t n,const int64_t *ap,const int64_t *ai,int64_t *p,double *control,double *info){
 typedef int(*fn)(int64_t,const int64_t*,const int64_t*,int64_t*,double*,double*);
 static fn real=NULL;if(!real)real=(fn)dlsym(RTLD_NEXT,"amd_l_order");double t=rp_now();int result=real(n,ap,ai,p,control,info);rp_stats.amd_us+=rp_now()-t;rp_stats.amd_calls++;return result;
}
