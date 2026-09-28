/* GeoTrust+ per-interval verification kernels for Cortex-M4F (float32, no heap).
 * Mirrors geotrust/verify.py (Verifier.stats edge construction + Verifier.exclude R/K rules),
 * the loss clamp (Verifier.clamp) and the watchdog test in geotrust/runner.py. */
#include <math.h>
#include "geotrust_verify.h"

#define MAXE (GT_VMAXN * (GT_VMAXN - 1) / 2 + GT_VMAXN)

static uint8_t ea[MAXE], eb[MAXE];      /* disagreement edges between neighbours a<b */
static int8_t eclaim[MAXE];             /* -1: two-sided disagreement, else index of the one-sided claimant */
static float edist[MAXE];               /* claimed distance of a one-sided claim */

int gt_reciprocity(int n, const float *D, const float *Dvi, const float *own,
                   float tau_r, float strong_d, uint8_t *blamed)
{
    int ne = 0, nb = 0;
    for (int v = 0; v < n; v++) {
        /* own link: a disagreement with the verifier's own measurement blames v */
        float x = Dvi[v];
        blamed[v] = (isnan(x) || fabsf(x - own[v]) > tau_r);
    }
    for (int v = 0; v < n; v++)
        for (int u = v + 1; u < n; u++) {
            float a = D[v * n + u], b = D[u * n + v];
            int ha = !isnan(a), hb = !isnan(b);
            if (ha && hb) {
                if (fabsf(a - b) > tau_r) { ea[ne] = v; eb[ne] = u; eclaim[ne] = -1; ne++; }
            } else if (ha || hb) {
                ea[ne] = v; eb[ne] = u; eclaim[ne] = ha ? v : u; edist[ne] = ha ? a : b; ne++;
            }
        }
    /* greedy vertex cover on edges whose endpoints are not yet blamed; only degree >= 2 convicts */
    static uint8_t alive[MAXE];
    int deg[GT_VMAXN];
    for (int e = 0; e < ne; e++) alive[e] = !blamed[ea[e]] && !blamed[eb[e]];
    for (;;) {
        for (int v = 0; v < n; v++) deg[v] = 0;
        int any = 0;
        for (int e = 0; e < ne; e++)
            if (alive[e]) { deg[ea[e]]++; deg[eb[e]]++; any = 1; }
        if (!any) break;
        int mx = 0, x = -1;
        for (int v = 0; v < n; v++) if (deg[v] > mx) mx = deg[v];
        if (mx < 2) break;
        /* ties: first node in edge order (Python's max() over an insertion-ordered dict) */
        for (int e = 0; e < ne && x < 0; e++)
            if (alive[e]) {
                if (deg[ea[e]] == mx) x = ea[e];
                else if (deg[eb[e]] == mx) x = eb[e];
            }
        blamed[x] = 1;
        for (int e = 0; e < ne; e++) if (ea[e] == x || eb[e] == x) alive[e] = 0;
    }
    /* one-sided claims well inside the range blame the claimant unless already explained */
    for (int e = 0; e < ne; e++)
        if (eclaim[e] >= 0 && !blamed[ea[e]] && !blamed[eb[e]] && edist[e] < strong_d + 1e-9f)
            blamed[eclaim[e]] = 1;
    for (int v = 0; v < n; v++) nb += blamed[v];
    return nb;
}

int gt_kinematic(int n, int m, const float *Dprev, const float *Dcur, float tau_k, uint8_t *flag)
{
    int nf = 0;
    for (int v = 0; v < n; v++) {
        float mx = -INFINITY;
        for (int u = 0; u < m; u++) {
            float p = Dprev[v * m + u], c = Dcur[v * m + u];
            if (!isnan(p) && !isnan(c) && p - c > mx) mx = p - c;
        }
        flag[v] = mx > tau_k;
        nf += flag[v];
    }
    return nf;
}

void gt_clamp(int cnt, const float *d, float *P, float a, float b, float c, float tau_p)
{
    for (int j = 0; j < cnt; j++) {
        if (isnan(d[j])) continue;
        float lo = 1.0f / (expf(a * d[j] + b) + c) - tau_p;     /* P(d) of Zhou et al. */
        if (P[j] < lo) P[j] = lo;
    }
}

int gt_watchdog(int n, const float *expect, const float *seen, const float *var, float z, uint8_t *flag)
{
    int nf = 0;
    for (int v = 0; v < n; v++) {
        flag[v] = expect[v] >= 10.0f && (seen[v] - expect[v]) / sqrtf(var[v] + 1.0f) < -z;
        nf += flag[v];
    }
    return nf;
}
