# Cycle-accurate timing on hardware

Goal: replace the QEMU instruction-count estimates in the paper (Sec. VII-F, Table VIII) with cycle counts measured
on a real STM32F405/F407 at 168 MHz with 5 flash wait states and the ART accelerator enabled.

## 1. Board

There is **no NUCLEO board with an STM32F405** (the earlier plan named a "NUCLEO-F405RG", which does not exist). Any
of the following works with the same binary (`bench_hw.elf`), which clocks itself from the internal HSI and needs no
board-specific crystal:

| Board | MCU | Debugger | Notes |
|---|---|---|---|
| **STM32F4DISCOVERY** (STM32F407G-DISC1) | STM32F407VG | on-board ST-LINK/V2-A | Recommended: cheap (~25 USD), plug-and-play. The F407 is the same die as the F405 (core, flash, ART) plus peripherals. |
| **Crazyflie 2.1** | STM32F405RG | external ST-LINK + Crazyflie debug adapter | The platform of the UWB swarm-ranging testbeds; the strongest statement for the paper. Erases the Crazyflie firmware (re-flash with `cfloader` afterwards). |
| Olimex STM32-H405, pyboard v1.1, WeAct STM32F405 | STM32F405RG | external ST-LINK V2/V3 (SWD: SWDIO, SWCLK, GND, 3V3) | Also fine. |
| NUCLEO-F446RE | STM32F446RE | on-board ST-LINK/V2-1 | Surrogate only: same Cortex-M4F at 168 MHz, but a different chip. Report it as such. |

## 2. Tools

**Linux / WSL:** `sudo apt install openocd gcc-arm-none-eabi`. Under WSL, attach the ST-LINK with
[usbipd-win](https://github.com/dorssel/usbipd-win): `usbipd list`, `usbipd bind --busid <id>` (admin),
`usbipd attach --wsl --busid <id>`.

**Windows only:** install the xPack OpenOCD build and run the `openocd` line of `run_hw.sh` in a terminal.

## 3. Run

```bash
cd embedded
bash build_hw.sh                                   # builds bench_hw.elf (and prints footprints)
bash run_hw.sh "STM32F4DISCOVERY (F407VG)"         # flash, run, capture, parse -> results_hw.json
```

`run_hw.sh` flashes through OpenOCD, enables semihosting and waits for `DONE` (about 2 s). `parse_hw.py` then:
- writes `results_hw.json`;
- checks the header: `clock_hz=168000000 flash_ws=5 timer=DWT` means cycle-accurate;
- checks correctness: the RFC 4493 CMAC vector, and that all liars in the synthetic neighbourhoods are convicted;
- prints LaTeX rows for the overhead table.

**If semihosting output does not appear:** the results are also in RAM. With OpenOCD running (`openocd -f
interface/stlink.cfg -f target/stm32f4x.cfg`):

```bash
arm-none-eabi-gdb bench_hw.elf -batch -ex "target extended-remote :3333" -ex "monitor arm semihosting enable" \
  -ex "monitor reset run" -ex "shell sleep 3" -ex "monitor halt" -ex "print gt_results"
```

Note that the binary uses semihosting and must run under a debugger; without one, the first `bkpt` halts the core.

## 4. What is measured

| Kernel | Content |
|---|---|
| `cmac8_229B` | 8-byte AES-CMAC over a 229-byte ranging unit (tiny-AES, table-less) |
| `reciprocity_n{12,20,30}` | disagreement graph, own-link blame, degree ≥ 2 greedy vertex cover, one-sided claims (bit-exact with the Python verifier: `crosscheck.c`, 400 cases, 0 mismatches) |
| `kin_clamp_wd_n*` | kinematic test + loss clamp over all reported links + watchdog z-tests |
| `round_total_n*` | one full interval: n CMAC verifications + all checks |
| `rgrr_n20_k25_f0` | relay computation of the earlier draft (reference) |

Each kernel is run 31 times on fresh synthetic neighbourhoods (10 % adaptive liars); the median, minimum and maximum
are reported.

## 5. QEMU reference (instruction counts, not cycles)

`bash build_hw.sh --qemu` runs the same code on QEMU (`qemu_hw_test.txt`). At 168 MHz, with 20 neighbours, it takes:
- CMAC over one 229-byte RU: 76.3 k instructions;
- reciprocity graph: 6.3 k instructions (about 38 µs at CPI 1);
- kinematic test, clamp and watchdog: 34 k instructions;
- whole interval: 1.57 M instructions, i.e. 9.3–14 ms per 100 ms at a CPI of 1.0–1.5.

The hardware run replaces the assumed CPI with measured cycles.
