/* 仅供LM性能测试：保持CHOLMOD分析结果和回代工作区，不接入生产solver。 */
#include <cholmod.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

_Static_assert(sizeof(size_t) == sizeof(int64_t), "64-bit indices required");
typedef struct {
    cholmod_common common;
    cholmod_factor *factor;
    cholmod_dense *solution, *work_y, *work_e;
} LmCholesky;

static cholmod_sparse matrix(size_t n, size_t *cp, size_t *ri, double *values) {
    cholmod_sparse a = {0};
    a.nrow = a.ncol = n;
    a.nzmax = cp[n];
    a.p = cp;
    a.i = ri;
    a.x = values;
    a.stype = 1; /* 对称矩阵，只读取上三角。 */
    a.itype = CHOLMOD_LONG;
    a.xtype = CHOLMOD_REAL;
    a.dtype = CHOLMOD_DOUBLE;
    a.sorted = a.packed = 1;
    return a;
}

void lm_cholmod_free(LmCholesky *s) {
    if (!s) return;
    cholmod_l_free_factor(&s->factor, &s->common);
    cholmod_l_free_dense(&s->solution, &s->common);
    cholmod_l_free_dense(&s->work_y, &s->common);
    cholmod_l_free_dense(&s->work_e, &s->common);
    cholmod_l_finish(&s->common);
    free(s);
}

LmCholesky *lm_cholmod_analyze(size_t n, size_t *cp, size_t *ri, double *values,
                             int supernodal, int threads) {
    LmCholesky *s = calloc(1, sizeof(*s));
    if (!s) return NULL;
    if (!cholmod_l_start(&s->common)) { free(s); return NULL; }
    s->common.supernodal = supernodal ? CHOLMOD_SUPERNODAL : CHOLMOD_SIMPLICIAL;
    s->common.final_asis = 0;
    s->common.final_ll = 1; /* 明确要求LLᵀ，不能把默认的LDLᵀ叫作Cholesky测试。 */
    s->common.final_super = supernodal;
    s->common.final_pack = 0;
    s->common.quick_return_if_not_posdef = 1;
    s->common.nthreads_max = threads;
    cholmod_sparse a = matrix(n, cp, ri, values);
    s->factor = cholmod_l_analyze(&a, &s->common);
    if (!s->factor) { lm_cholmod_free(s); return NULL; }
    return s;
}

int lm_cholmod_factorize(LmCholesky *s, size_t n, size_t *cp, size_t *ri,
                        double *values) {
    cholmod_sparse a = matrix(n, cp, ri, values);
    int ok = cholmod_l_factorize(&a, s->factor, &s->common);
    return ok && s->common.status == CHOLMOD_OK && s->factor->minor == n
        && s->factor->is_ll;
}

int lm_cholmod_solve(LmCholesky *s, size_t n, double *rhs) {
    cholmod_dense b = {0};
    b.nrow = b.d = b.nzmax = n;
    b.ncol = 1;
    b.x = rhs;
    b.xtype = CHOLMOD_REAL;
    b.dtype = CHOLMOD_DOUBLE;
    int ok = cholmod_l_solve2(CHOLMOD_A, s->factor, &b, NULL, &s->solution,
                            NULL, &s->work_y, &s->work_e, &s->common);
    if (ok) memcpy(rhs, s->solution->x, n * sizeof(double));
    return ok;
}
