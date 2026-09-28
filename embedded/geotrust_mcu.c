/* GeoTrust+ on-board kernels (float32, no heap) - C port of geotrust/verify.py and geotrust/mpr.py
 *   gt_cm_resid     exact Cayley-Menger residual of one 5-point tuple (3 dets + 9 cofactors)
 *   gt_robust_greedy R-GRR for f <= 1 (greedy on the submodular G_f + reverse-delete prune)
 *   gt_cmac8        AES-128-CMAC truncated to 8 bytes (TESLA per-RU MAC)
 */
#include <math.h>
#include <string.h>
#include "geotrust_mcu.h"
#include "tiny-AES-c/aes.h"

/* ------------------------------------------------------------------ linear algebra */
static float det_n(float *a, int n)            /* in-place LU with partial pivoting, a is n x n */
{
    float d = 1.0f;
    for (int c = 0; c < n; c++) {
        int p = c;
        float best = fabsf(a[c * n + c]);
        for (int r = c + 1; r < n; r++)
            if (fabsf(a[r * n + c]) > best) { best = fabsf(a[r * n + c]); p = r; }
        if (best == 0.0f) return 0.0f;
        if (p != c) {
            for (int k = 0; k < n; k++) { float t = a[c * n + k]; a[c * n + k] = a[p * n + k]; a[p * n + k] = t; }
            d = -d;
        }
        float piv = a[c * n + c];
        d *= piv;
        for (int r = c + 1; r < n; r++) {
            float m = a[r * n + c] / piv;
            for (int k = c + 1; k < n; k++) a[r * n + k] -= m * a[c * n + k];
        }
    }
    return d;
}

static void bordered(const float d[5][5], float x, float B[36])
{
    for (int r = 0; r < 6; r++)
        for (int c = 0; c < 6; c++)
            B[r * 6 + c] = (r == 0 || c == 0) ? (r == c ? 0.0f : 1.0f) : d[r - 1][c - 1] * d[r - 1][c - 1];
    B[1 * 6 + 2] = B[2 * 6 + 1] = x;
}

static float det6_at(const float d[5][5], float x)
{
    float B[36];
    bordered(d, x, B);
    return det_n(B, 6);
}

static float minor5(const float B[36], int skip_r, int skip_c)
{
    float m[25];
    int k = 0;
    for (int r = 0; r < 6; r++) {
        if (r == skip_r) continue;
        for (int c = 0; c < 6; c++) {
            if (c == skip_c) continue;
            m[k++] = B[r * 6 + c];
        }
    }
    return det_n(m, 5);
}

/* d: symmetric 5x5 distances, index 0 = v, 1 = u (d[0][1] is the claim under test).
 * Returns 0 and fills e (claim - nearest consistent distance) and sd, or -1 if roots are complex. */
int gt_cm_resid(const float d[5][5], float sig, float *e, float *sd)
{
    /* distances are normalised by their mean so float32 determinants stay well scaled */
    float s = 0.0f, dn[5][5];
    for (int r = 0; r < 5; r++) for (int c = 0; c < 5; c++) s += d[r][c];
    s /= 20.0f;
    for (int r = 0; r < 5; r++) for (int c = 0; c < 5; c++) dn[r][c] = d[r][c] / s;
    sig /= s;
    float x0 = dn[0][1] * dn[0][1];
    float f0 = det6_at(dn, 0.0f), f1 = det6_at(dn, x0), f2 = det6_at(dn, 2.0f * x0);
    float a = (f2 - 2.0f * f1 + f0) / (2.0f * x0 * x0);
    float b = (f1 - f0) / x0 - a * x0;
    float disc = b * b - 4.0f * a * f0;
    if (disc <= 0.0f || a == 0.0f) return -1;
    float sq = sqrtf(disc);
    float r1 = (-b - sq) / (2.0f * a), r2 = (-b + sq) / (2.0f * a);
    float d1 = sqrtf(fmaxf(r1, 0.0f)), d2 = sqrtf(fmaxf(r2, 0.0f));
    float e1 = dn[0][1] - d1, e2 = dn[0][1] - d2;
    float xr = fabsf(e1) < fabsf(e2) ? r1 : r2;
    float ds = fabsf(e1) < fabsf(e2) ? d1 : d2;
    *e = (fabsf(e1) < fabsf(e2) ? e1 : e2) * s;
    float B[36], var = 0.0f;
    bordered(dn, xr, B);
    for (int j = 0; j < 5; j++)
        for (int l = j + 1; l < 5; l++) {
            if (j == 0 && l == 1) continue;
            float cof = (((j + l) & 1) ? -1.0f : 1.0f) * minor5(B, j + 1, l + 1);
            float g = 2.0f * cof * 2.0f * dn[j][l] * sig;
            var += g * g;
        }
    float slope = fabsf(2.0f * a * xr + b);
    *sd = sqrtf(var) / slope / (2.0f * fmaxf(ds, 0.05f)) * s;
    return 0;
}

/* ------------------------------------------------------------------ R-GRR (f <= 1) */
/* C: n x k contributions (row-major), t: k targets.  Fail sets: {} and {v} for f = 1.
 * S: output relay indices.  Returns |S|.  Work arrays live on the stack (n <= GT_MAXN, k <= GT_MAXK). */
int gt_robust_greedy(int n, int k, const float *C, const float *t, int f, uint8_t *S)
{
    static float T[(GT_MAXN + 1) * GT_MAXK], cur[(GT_MAXN + 1) * GT_MAXK];
    int nF = f ? n + 1 : 1;
    uint8_t chosen[GT_MAXN] = {0};
    float goal = 0.0f;
    for (int F = 0; F < nF; F++)
        for (int u = 0; u < k; u++) {
            float sum = 0.0f;
            for (int v = 0; v < n; v++) if (F == 0 || v != F - 1) sum += C[v * k + u];
            float tt = fminf(t[u], sum);
            T[F * k + u] = tt;
            cur[F * k + u] = 0.0f;
            goal += tt;
        }
    int m = 0;
    float have = 0.0f;
    /* lazy greedy (Minoux): G_f is submodular, so a stale marginal gain is an upper bound;
     * only the candidate on top of the bound list is re-evaluated.  Output = plain greedy. */
    float ub[GT_MAXN];
    for (int x = 0; x < n; x++) ub[x] = 3.4e38f;
    while (goal - have > 1e-5f * (goal + 1.0f)) {
        int best = -1;
        float bg = 1e-9f;
        for (;;) {
            int x = -1;
            float top = -1.0f, second = -1.0f;
            for (int y = 0; y < n; y++) {
                if (chosen[y]) continue;
                if (ub[y] > top) { second = top; top = ub[y]; x = y; }
                else if (ub[y] > second) second = ub[y];
            }
            if (x < 0 || top <= 1e-9f) break;
            float g = 0.0f;
            for (int F = 0; F < nF; F++) {
                if (F && F - 1 == x) continue;
                for (int u = 0; u < k; u++) {
                    float c0 = cur[F * k + u], tt = T[F * k + u];
                    if (c0 < tt) g += fminf(tt, c0 + C[x * k + u]) - c0;
                }
            }
            ub[x] = g;
            if (g >= second) { if (g > bg) { bg = g; best = x; } break; }
        }
        if (best < 0) break;
        chosen[best] = 1;
        S[m++] = (uint8_t)best;
        have = 0.0f;
        for (int F = 0; F < nF; F++)
            for (int u = 0; u < k; u++) {
                if (!(F && F - 1 == best)) cur[F * k + u] += C[best * k + u];
                have += fminf(cur[F * k + u], T[F * k + u]);
            }
    }
    /* reverse-delete prune */
    for (int j = m - 1; j >= 0; j--) {
        int x = S[j], ok = 1;
        for (int F = 0; F < nF && ok; F++)
            for (int u = 0; u < k && ok; u++) {
                float sum = 0.0f;
                for (int q = 0; q < m; q++) {
                    int v = S[q];
                    if (v == x || (F && F - 1 == v)) continue;
                    sum += C[v * k + u];
                }
                if (sum < T[F * k + u] - 1e-5f) ok = 0;
            }
        if (ok) { for (int q = j; q < m - 1; q++) S[q] = S[q + 1]; m--; }
    }
    return m;
}

/* ------------------------------------------------------------------ AES-CMAC (RFC 4493), 8-byte tag */
static void dbl(uint8_t b[16])
{
    uint8_t carry = b[0] & 0x80;
    for (int i = 0; i < 15; i++) b[i] = (uint8_t)((b[i] << 1) | (b[i + 1] >> 7));
    b[15] = (uint8_t)(b[15] << 1);
    if (carry) b[15] ^= 0x87;
}

void gt_cmac8(const uint8_t key[16], const uint8_t *msg, int len, uint8_t tag[8])
{
    struct AES_ctx ctx;
    uint8_t L[16] = {0}, x[16] = {0}, blk[16];
    AES_init_ctx(&ctx, key);
    AES_ECB_encrypt(&ctx, L);
    dbl(L);                                   /* K1 */
    int nb = (len + 15) / 16;
    if (nb == 0) nb = 1;
    int full = (len > 0 && len % 16 == 0);
    uint8_t K2[16];
    memcpy(K2, L, 16);
    dbl(K2);
    for (int i = 0; i < nb; i++) {
        int off = i * 16, last = (i == nb - 1);
        for (int j = 0; j < 16; j++) {
            uint8_t m;
            if (off + j < len) m = msg[off + j];
            else m = (off + j == len) ? 0x80 : 0x00;
            if (last) m ^= full ? L[j] : K2[j];
            blk[j] = x[j] ^ m;
        }
        memcpy(x, blk, 16);
        AES_ECB_encrypt(&ctx, x);
    }
    memcpy(tag, x, 8);
}
