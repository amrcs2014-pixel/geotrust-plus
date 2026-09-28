/* Bare-metal Cortex-M4F benchmark (QEMU netduinoplus2 = STM32F405).  Timing uses SysTick under
 * `-icount shift=0` (virtual time advances 1 ns per instruction), so tick deltas measure executed
 * instructions; a calibration loop of known length converts ticks -> instructions. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "geotrust_mcu.h"

#define SYST_CSR (*(volatile uint32_t *)0xE000E010)
#define SYST_RVR (*(volatile uint32_t *)0xE000E014)
#define SYST_CVR (*(volatile uint32_t *)0xE000E018)
#define CPACR (*(volatile uint32_t *)0xE000ED88)

static void sh_write0(const char *s)
{
    register int r0 __asm__("r0") = 0x04;
    register const char *r1 __asm__("r1") = s;
    __asm__ volatile("bkpt 0xAB" : "+r"(r0) : "r"(r1) : "memory");
}
static void sh_exit(void)
{
    register int r0 __asm__("r0") = 0x18;
    register int r1 __asm__("r1") = 0x20026;
    __asm__ volatile("bkpt 0xAB" : "+r"(r0) : "r"(r1) : "memory");
}
static char buf[160];
#define PRINT(...) do { snprintf(buf, sizeof buf, __VA_ARGS__); sh_write0(buf); } while (0)

static uint32_t lcg = 12345;
static float frand(void) { lcg = lcg * 1664525u + 1013904223u; return (lcg >> 8) * (1.0f / 16777216.0f); }

static inline uint32_t now(void) { return SYST_CVR; }
static uint32_t elapsed(uint32_t a, uint32_t b) { return (a - b) & 0x00FFFFFF; }   /* down-counter */

static void __attribute__((noinline)) spin(uint32_t n)
{
    __asm__ volatile("1: subs %0, %0, #1\n bne 1b" : "+r"(n));       /* 2 instructions / iter */
}

static float C[GT_MAXN * GT_MAXK], t[GT_MAXK];
static uint8_t S[GT_MAXN];

static void make_instance(int n, int k)
{
    for (int v = 0; v < n; v++)
        for (int u = 0; u < k; u++)
            C[v * k + u] = frand() < 0.35f ? 2.0f + 5.0f * frand() : 0.0f;
    for (int u = 0; u < k; u++) {
        float mx = 0.0f;
        for (int v = 0; v < n; v++) mx = fmaxf(mx, C[v * k + u]);
        if (mx == 0.0f) { C[u] = 3.0f; mx = 3.0f; }
        t[u] = mx - logf(1.2f);
    }
}

void Reset_C(void);
int main(void)
{
    CPACR |= (0xFu << 20);                                   /* enable FPU */
    __asm__ volatile("dsb\n isb");
    SYST_RVR = 0x00FFFFFF;
    SYST_CVR = 0;
    SYST_CSR = 0x5;                                          /* enable, core clock, no irq */

    /* calibration: ticks per instruction */
    uint32_t a = now();
    spin(200000);
    uint32_t tk = elapsed(a, now());
    float ipt = 400000.0f / (float)tk;                       /* instructions per tick */
    PRINT("CAL ticks=%lu insn_per_tick=%.4f\n", (unsigned long)tk, ipt);

    /* Cayley-Menger residual: average over 200 random tuples */
    float d[5][5], P[5][3], e, sd;
    uint32_t tot = 0; int ok = 0;
    for (int it = 0; it < 200; it++) {
        for (int p = 0; p < 5; p++) for (int q = 0; q < 3; q++) P[p][q] = 4.0f * frand();
        for (int p = 0; p < 5; p++) for (int q = 0; q < 5; q++) {
            float dx = P[p][0] - P[q][0], dy = P[p][1] - P[q][1], dz = P[p][2] - P[q][2];
            d[p][q] = sqrtf(dx * dx + dy * dy + dz * dz);
        }
        a = now();
        ok += gt_cm_resid(d, 0.035f, &e, &sd) == 0;
        tot += elapsed(a, now());
    }
    PRINT("CM insn_per_tuple=%.0f ok=%d\n", tot * ipt / 200.0f, ok);

    /* R-GRR */
    static const int sizes[][2] = {{12, 10}, {20, 25}, {30, 40}};
    for (int s = 0; s < 3; s++)
        for (int f = 0; f <= 1; f++) {
            make_instance(sizes[s][0], sizes[s][1]);
            a = now();
            int m = gt_robust_greedy(sizes[s][0], sizes[s][1], C, t, f, S);
            PRINT("RGRR n=%d k=%d f=%d insn=%.0f size=%d\n", sizes[s][0], sizes[s][1], f,
                  elapsed(a, now()) * ipt, m);
        }

    /* AES-CMAC-8 over a 20-neighbour ranging unit (229 B) */
    uint8_t key[16], msg[229], tag[8];
    for (int i = 0; i < 16; i++) key[i] = (uint8_t)i;
    for (int i = 0; i < 229; i++) msg[i] = (uint8_t)(i * 7);
    a = now();
    gt_cmac8(key, msg, 229, tag);
    PRINT("CMAC229 insn=%.0f tag=%02x%02x%02x%02x\n", elapsed(a, now()) * ipt, tag[0], tag[1], tag[2], tag[3]);
    /* RFC 4493 example 2 (16-byte message) for correctness */
    static const uint8_t k2[16] = {0x2b,0x7e,0x15,0x16,0x28,0xae,0xd2,0xa6,0xab,0xf7,0x15,0x88,0x09,0xcf,0x4f,0x3c};
    static const uint8_t m2[16] = {0x6b,0xc1,0xbe,0xe2,0x2e,0x40,0x9f,0x96,0xe9,0x3d,0x7e,0x11,0x73,0x93,0x17,0x2a};
    gt_cmac8(k2, m2, 16, tag);
    PRINT("RFC4493 ex2 tag=%02x%02x%02x%02x (expect 070a16b4)\n", tag[0], tag[1], tag[2], tag[3]);
    sh_exit();
    for (;;) {}
}

/* ---- minimal startup ---- */
extern uint32_t _estack, _sidata, _sdata, _edata, _sbss, _ebss;
void __attribute__((naked)) Reset_Handler(void)
{
    /* enable CP10/CP11 (FPU) before any compiler-generated FP instruction can run */
    __asm__ volatile("ldr r0, =0xE000ED88\n ldr r1, [r0]\n orr r1, r1, #(0xF << 20)\n str r1, [r0]\n"
                     "dsb\n isb\n b Reset_C");
}
void Reset_C(void)
{
    uint32_t *s = &_sidata, *p = &_sdata;
    while (p < &_edata) *p++ = *s++;
    for (p = &_sbss; p < &_ebss;) *p++ = 0;
    main();
}
void Default_Handler(void) { for (;;) {} }
__attribute__((section(".isr_vector"))) const void *vectors[16] = {
    &_estack, Reset_Handler, Default_Handler, Default_Handler, Default_Handler, Default_Handler,
    Default_Handler, 0, 0, 0, 0, Default_Handler, Default_Handler, 0, Default_Handler, Default_Handler};
