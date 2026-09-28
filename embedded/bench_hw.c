/* Cycle-accurate benchmark of the GeoTrust+ kernels on real STM32F405/F407 hardware
 * (Crazyflie 2.x, STM32F4DISCOVERY, Olimex STM32-H405, pyboard, ...).
 *
 * - Core clock 168 MHz from the internal HSI via the PLL (no board-specific crystal needed),
 *   flash with 5 wait states + prefetch + I/D caches (the production configuration).
 * - Timing with the DWT cycle counter (CYCCNT); every kernel is repeated and the median reported.
 * - Output via ARM semihosting (OpenOCD: "arm semihosting enable"); in addition all results are
 *   kept in the global struct `gt_results` so they can be read with a debugger if semihosting is
 *   unavailable (see HW_TIMING.md).
 * - Build with -DQEMU_TEST to run the same code on QEMU (SysTick under -icount instead of CYCCNT).
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "geotrust_mcu.h"
#include "geotrust_verify.h"

#define REG(a) (*(volatile uint32_t *)(a))
#define RCC_CR REG(0x40023800)
#define RCC_PLLCFGR REG(0x40023804)
#define RCC_CFGR REG(0x40023808)
#define FLASH_ACR REG(0x40023C00)
#define DEMCR REG(0xE000EDFC)
#define DWT_CTRL REG(0xE0001000)
#define DWT_CYCCNT REG(0xE0001004)
#define SYST_CSR REG(0xE000E010)
#define SYST_RVR REG(0xE000E014)
#define SYST_CVR REG(0xE000E018)

#define REPS 31
#define NKER 16

struct result { char name[24]; uint32_t cycles_med, cycles_min, cycles_max; int32_t check; };
volatile struct {
    uint32_t magic, clock_hz, flash_ws, timer;      /* timer: 1 = DWT CYCCNT, 2 = SysTick */
    uint32_t n;
    struct result r[NKER];
    uint32_t done;
} gt_results;

/* ------------------------------------------------------------------ semihosting output */
static int semihost_ok = 1;
static void sh_write0(const char *s)
{
    if (!semihost_ok) return;
    register int r0 __asm__("r0") = 0x04;
    register const char *r1 __asm__("r1") = s;
    __asm__ volatile("bkpt 0xAB" : "+r"(r0) : "r"(r1) : "memory");
}
static char buf[200];
#define PRINT(...) do { snprintf(buf, sizeof buf, __VA_ARGS__); sh_write0(buf); } while (0)

/* ------------------------------------------------------------------ clock and timer */
static uint32_t clock_init(void)
{
#ifdef QEMU_TEST
    return 168000000u;
#else
    RCC_CR |= 1u;                                            /* HSI on */
    for (int t = 0; t < 1000000 && !(RCC_CR & (1u << 1)); t++) {}
    /* PLL: HSI 16 MHz / M=16 * N=336 / P=2 = 168 MHz, Q=7 (48 MHz USB) */
    RCC_PLLCFGR = 16u | (336u << 6) | (0u << 16) | (0u << 22) | (7u << 24);
    RCC_CR |= (1u << 24);                                    /* PLL on */
    int t;
    for (t = 0; t < 1000000 && !(RCC_CR & (1u << 25)); t++) {}
    if (!(RCC_CR & (1u << 25))) return 16000000u;            /* PLL failed: stay on HSI */
    FLASH_ACR = 5u | (1u << 8) | (1u << 9) | (1u << 10);     /* 5 WS, prefetch, I-cache, D-cache */
    RCC_CFGR = (RCC_CFGR & ~0xFCF0u) | (0u << 4) | (5u << 10) | (4u << 13);   /* AHB/1, APB1/4, APB2/2 */
    RCC_CFGR = (RCC_CFGR & ~3u) | 2u;                        /* SYSCLK = PLL */
    for (t = 0; t < 1000000 && ((RCC_CFGR >> 2) & 3u) != 2u; t++) {}
    return ((RCC_CFGR >> 2) & 3u) == 2u ? 168000000u : 16000000u;
#endif
}

static int use_dwt;
static float tick_scale = 1.0f;                              /* QEMU: instructions per SysTick tick */
static void __attribute__((noinline)) spin(uint32_t n)
{
    __asm__ volatile("1: subs %0, %0, #1\n bne 1b" : "+r"(n));   /* 2 instructions / iteration */
}
static void timer_init(void)
{
    DEMCR |= (1u << 24);                                     /* TRCENA */
    DWT_CYCCNT = 0;
    DWT_CTRL |= 1u;                                          /* CYCCNTENA */
    uint32_t a = DWT_CYCCNT;
    for (volatile int i = 0; i < 1000; i++) {}
    use_dwt = (DWT_CYCCNT - a) > 1000;
#ifdef QEMU_TEST
    use_dwt = 0;
#endif
    if (!use_dwt) { SYST_RVR = 0x00FFFFFF; SYST_CVR = 0; SYST_CSR = 0x5; }
#ifdef QEMU_TEST
    uint32_t t0 = 0x00FFFFFFu - SYST_CVR;
    spin(200000);
    tick_scale = 400000.0f / (float)((((0x00FFFFFFu - SYST_CVR) - t0)) & 0x00FFFFFF);
#endif
}
static inline uint32_t now(void) { return use_dwt ? DWT_CYCCNT : (0x00FFFFFFu - SYST_CVR); }
static inline uint32_t since(uint32_t a) { return use_dwt ? DWT_CYCCNT - a : ((0x00FFFFFFu - SYST_CVR) - a) & 0x00FFFFFF; }

static void sort_u32(uint32_t *x, int n)
{
    for (int i = 1; i < n; i++) { uint32_t v = x[i]; int j = i - 1; while (j >= 0 && x[j] > v) { x[j + 1] = x[j]; j--; } x[j + 1] = v; }
}
static void record(const char *name, uint32_t *c, int check)
{
    sort_u32(c, REPS);
    if (tick_scale != 1.0f)                                  /* QEMU: report instructions, not ticks */
        for (int i = 0; i < REPS; i++) c[i] = (uint32_t)(c[i] * tick_scale + 0.5f);
    int k = gt_results.n++;
    struct result *r = (struct result *)&gt_results.r[k];
    strncpy(r->name, name, sizeof r->name - 1);
    r->cycles_med = c[REPS / 2]; r->cycles_min = c[0]; r->cycles_max = c[REPS - 1]; r->check = check;
    PRINT("KERNEL %-22s cycles_med=%lu min=%lu max=%lu us=%.1f check=%ld\n", name, (unsigned long)r->cycles_med,
          (unsigned long)r->cycles_min, (unsigned long)r->cycles_max,
          r->cycles_med * 1e6f / (float)gt_results.clock_hz, (long)check);
}

/* ------------------------------------------------------------------ synthetic neighbourhood */
static uint32_t lcg = 12345;
static float frand(void) { lcg = lcg * 1664525u + 1013904223u; return (lcg >> 8) * (1.0f / 16777216.0f); }
static float nrand(void) { float s = 0; for (int i = 0; i < 12; i++) s += frand(); return s - 6.0f; }

#define RC 4.0f
static float P3[GT_VMAXN + 1][3];
static float D[GT_VMAXN * GT_VMAXN], Dvi[GT_VMAXN], own[GT_VMAXN];
static float Dprev[GT_VMAXN * GT_VMAXN], Pl[GT_VMAXN * GT_VMAXN], Pw[GT_VMAXN * GT_VMAXN];
static float wexp[GT_VMAXN], wseen[GT_VMAXN], wvar[GT_VMAXN];
static uint8_t bl[GT_VMAXN], fk[GT_VMAXN], fw[GT_VMAXN];

static int make_neighbourhood(int n, int liars)
{
    for (int p = 0; p <= n; p++)                              /* p = n is the verifier, at the centre */
        for (int q = 0; q < 3; q++) P3[p][q] = (p == n) ? 0.0f : (2.0f * frand() - 1.0f) * RC * 0.55f;
    for (int v = 0; v < n; v++) {
        for (int u = 0; u < n; u++) {
            float dx = P3[v][0] - P3[u][0], dy = P3[v][1] - P3[u][1], dz = P3[v][2] - P3[u][2];
            float d = sqrtf(dx * dx + dy * dy + dz * dz) + 0.03f * nrand();
            D[v * n + u] = (u == v || d > RC) ? NAN : d;
        }
        float dx = P3[v][0], dy = P3[v][1], dz = P3[v][2];
        own[v] = sqrtf(dx * dx + dy * dy + dz * dz);
        Dvi[v] = own[v] + 0.014f * nrand();
    }
    for (int v = 0; v < n; v++)                               /* symmetric shared error */
        for (int u = v + 1; u < n; u++)
            if (!isnan(D[v * n + u]) && !isnan(D[u * n + v])) D[u * n + v] = D[v * n + u] + 0.014f * nrand();
    for (int l = 0; l < liars; l++)                           /* adaptive liars: shorten far links by 40 % */
        for (int u = 0; u < n; u++)
            if (!isnan(D[l * n + u]) && D[l * n + u] > RC / 2) D[l * n + u] *= 0.6f;
    for (int j = 0; j < n * n; j++) { Dprev[j] = isnan(D[j]) ? NAN : D[j] + 0.05f * frand(); Pl[j] = 0.0f; }
    for (int v = 0; v < n; v++) { wexp[v] = 25.0f; wseen[v] = v < liars ? 0.0f : 24.0f; wvar[v] = 2.0f; }
    return liars;
}

/* ------------------------------------------------------------------ main */
static float Cm[GT_MAXN * GT_MAXK], tg[GT_MAXK];
static uint8_t S[GT_MAXN];

int main(void)
{
    gt_results.magic = 0x47545031;                           /* "GTP1" */
    gt_results.clock_hz = clock_init();
    gt_results.flash_ws = FLASH_ACR & 7u;
    timer_init();
    gt_results.timer = use_dwt ? 1 : 2;
    PRINT("GEOTRUST-HW clock_hz=%lu flash_ws=%lu timer=%s\n", (unsigned long)gt_results.clock_hz,
          (unsigned long)gt_results.flash_ws, use_dwt ? "DWT" : "SysTick");
    uint32_t c[REPS];

    /* timer overhead */
    for (int r = 0; r < REPS; r++) { uint32_t a = now(); c[r] = since(a); }
    record("timer_overhead", c, 0);

    /* AES-CMAC-8 over a 229-byte ranging unit (20 neighbours); RFC 4493 example 2 as check */
    uint8_t key[16], msg[229], tag[8];
    for (int i = 0; i < 16; i++) key[i] = (uint8_t)i;
    for (int i = 0; i < 229; i++) msg[i] = (uint8_t)(i * 7);
    for (int r = 0; r < REPS; r++) { uint32_t a = now(); gt_cmac8(key, msg, 229, tag); c[r] = since(a); }
    static const uint8_t k2[16] = {0x2b,0x7e,0x15,0x16,0x28,0xae,0xd2,0xa6,0xab,0xf7,0x15,0x88,0x09,0xcf,0x4f,0x3c};
    static const uint8_t m2[16] = {0x6b,0xc1,0xbe,0xe2,0x2e,0x40,0x9f,0x96,0xe9,0x3d,0x7e,0x11,0x73,0x93,0x17,0x2a};
    uint8_t t2[8];
    gt_cmac8(k2, m2, 16, t2);
    int rfc_ok = t2[0] == 0x07 && t2[1] == 0x0a && t2[2] == 0x16 && t2[3] == 0xb4;
    record("cmac8_229B", c, rfc_ok);

    /* reciprocity graph + vertex cover, kinematic, clamp, watchdog for 12/20/30 neighbours */
    static const int ns[3] = {12, 20, 30};
    char nm[24];
    for (int s = 0; s < 3; s++) {
        int n = ns[s], liars = n / 10;
        int caught = 0;
        for (int r = 0; r < REPS; r++) {
            make_neighbourhood(n, liars);
            uint32_t a = now();
            gt_reciprocity(n, D, Dvi, own, 0.0428f, RC, bl);
            c[r] = since(a);
            if (r == 0) for (int v = 0; v < liars; v++) caught += bl[v];
        }
        snprintf(nm, sizeof nm, "reciprocity_n%d", n); record(nm, c, caught);
        for (int r = 0; r < REPS; r++) {
            make_neighbourhood(n, liars);
            uint32_t a = now();
            gt_kinematic(n, n, Dprev, D, 0.3428f, fk);
            memcpy(Pw, Pl, sizeof(float) * n * n);
            gt_clamp(n * n, D, Pw, -1.61f, 9.97f, 1.01f, 0.0093f);
            gt_watchdog(n, wexp, wseen, wvar, 3.5f, fw);
            c[r] = since(a);
        }
        snprintf(nm, sizeof nm, "kin_clamp_wd_n%d", n); record(nm, c, 0);
        /* whole interval: n CMAC verifications + all checks */
        for (int r = 0; r < 5; r++) {
            make_neighbourhood(n, liars);
            uint32_t a = now();
            for (int v = 0; v < n; v++) gt_cmac8(key, msg, 229, tag);
            gt_reciprocity(n, D, Dvi, own, 0.0428f, RC, bl);
            gt_kinematic(n, n, Dprev, D, 0.3428f, fk);
            gt_clamp(n * n, D, Pl, -1.61f, 9.97f, 1.01f, 0.0093f);
            gt_watchdog(n, wexp, wseen, wvar, 3.5f, fw);
            c[r] = since(a);
        }
        for (int r = 5; r < REPS; r++) c[r] = c[r % 5];
        snprintf(nm, sizeof nm, "round_total_n%d", n); record(nm, c, 0);
    }

    /* reference: RFC 3626-size relay computation used in earlier drafts (R-GRR f=0, n=20, k=25) */
    for (int r = 0; r < REPS; r++) {
        for (int v = 0; v < 20; v++) for (int u = 0; u < 25; u++) Cm[v * 25 + u] = frand() < 0.35f ? 2.0f + 5.0f * frand() : 0.0f;
        for (int u = 0; u < 25; u++) { float mx = 0; for (int v = 0; v < 20; v++) mx = fmaxf(mx, Cm[v * 25 + u]); if (mx == 0) { Cm[u] = 3; mx = 3; } tg[u] = mx - logf(1.2f); }
        uint32_t a = now(); gt_robust_greedy(20, 25, Cm, tg, 0, S); c[r] = since(a);
    }
    record("rgrr_n20_k25_f0", c, 0);

    gt_results.done = 0x444F4E45;                            /* "DONE" */
    PRINT("DONE\n");
#ifdef QEMU_TEST
    { register int r0 __asm__("r0") = 0x18; register int r1 __asm__("r1") = 0x20026;
      __asm__ volatile("bkpt 0xAB" : "+r"(r0) : "r"(r1) : "memory"); }
#endif
    for (;;) { __asm__ volatile("wfi"); }
}

/* ------------------------------------------------------------------ minimal startup */
extern uint32_t _estack, _sidata, _sdata, _edata, _sbss, _ebss;
void Reset_C(void);
void __attribute__((naked)) Reset_Handler(void)
{
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
