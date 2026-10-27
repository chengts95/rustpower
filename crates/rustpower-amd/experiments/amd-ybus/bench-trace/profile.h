#include <time.h>
#include <stdint.h>
typedef struct {double amd_us,prep_us;uint64_t amd_calls;} rp_profile_stats;
extern rp_profile_stats rp_stats;
static inline double rp_now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e6+t.tv_nsec*1e-3;}
