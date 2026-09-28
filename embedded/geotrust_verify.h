#ifndef GEOTRUST_VERIFY_H
#define GEOTRUST_VERIFY_H
#include <stdint.h>

#define GT_VMAXN 32   /* max authenticated one-hop neighbours */

/* Reciprocity disagreement graph with own-link blame, degree>=2 greedy vertex cover and one-sided
 * claimant blame.  D[v*n+u]: distance neighbour v reports for neighbour u (NAN = not listed);
 * Dvi[v]: distance v reports for the verifier; own[v]: verifier's own (gated) distance to v.
 * strong_d = strong_frac * R_c.  Returns the number of blamed neighbours. */
int gt_reciprocity(int n, const float *D, const float *Dvi, const float *own,
                   float tau_r, float strong_d, uint8_t *blamed);
/* One-sided kinematic test over m reported links per neighbour (row-major n x m, NAN = missing). */
int gt_kinematic(int n, int m, const float *Dprev, const float *Dcur, float tau_k, uint8_t *flag);
/* Loss clamp P >= P(d) - tau_p over cnt reported links. */
void gt_clamp(int cnt, const float *d, float *P, float a, float b, float c, float tau_p);
/* Watchdog: one-sided binomial z-test (expected >= 10 packets). */
int gt_watchdog(int n, const float *expect, const float *seen, const float *var, float z, uint8_t *flag);

#endif
