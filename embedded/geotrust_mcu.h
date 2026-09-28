#ifndef GEOTRUST_MCU_H
#define GEOTRUST_MCU_H
#include <stdint.h>

#define GT_MAXN 32   /* max candidate relays */
#define GT_MAXK 48   /* max two-hop targets  */

int gt_cm_resid(const float d[5][5], float sig, float *e, float *sd);
int gt_robust_greedy(int n, int k, const float *C, const float *t, int f, uint8_t *S);
void gt_cmac8(const uint8_t key[16], const uint8_t *msg, int len, uint8_t tag[8]);

#endif
