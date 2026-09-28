/* Host cross-check: run gt_reciprocity on the cases produced by scripts/mcu_crosscheck_gen.py and
 * compare with the Python reference verifier.  Build: gcc -O2 crosscheck.c geotrust_verify.c -lm */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "geotrust_verify.h"

static float rd(FILE *f)
{
    char s[32];
    if (fscanf(f, "%31s", s) != 1) exit(2);
    return strcmp(s, "nan") == 0 ? NAN : strtof(s, NULL);
}

int main(int argc, char **argv)
{
    FILE *f = fopen(argc > 1 ? argv[1] : "crosscheck_cases.txt", "r");
    if (!f) return 1;
    int nc; float tau;
    if (fscanf(f, "%d %f", &nc, &tau) != 2) return 2;
    static float D[GT_VMAXN * GT_VMAXN], Dvi[GT_VMAXN], own[GT_VMAXN];
    uint8_t bl[GT_VMAXN];
    int bad = 0, flags = 0;
    for (int c = 0; c < nc; c++) {
        int n; float sd;
        if (fscanf(f, "%d %f", &n, &sd) != 2) return 2;
        for (int j = 0; j < n * n; j++) D[j] = rd(f);
        for (int j = 0; j < n; j++) Dvi[j] = rd(f);
        for (int j = 0; j < n; j++) own[j] = rd(f);
        gt_reciprocity(n, D, Dvi, own, tau, sd, bl);
        int mism = 0;
        for (int j = 0; j < n; j++) {
            int r; if (fscanf(f, "%d", &r) != 1) return 2;
            mism += (r != bl[j]); flags += r;
        }
        if (mism) { bad++; printf("case %d: %d mismatches (n=%d)\n", c, mism, n); }
    }
    printf("%d cases, %d reference convictions, %d cases with mismatches\n", nc, flags, bad);
    return bad != 0;
}
