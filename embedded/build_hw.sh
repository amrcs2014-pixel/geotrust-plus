#!/bin/bash
# Build the hardware benchmark (bench_hw.elf/.bin) and a QEMU functional test of the same code.
# Run inside WSL/Linux:  bash build_hw.sh [--qemu]
set -e
cd "$(dirname "$0")"
CF="-mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16 -O2 -ffunction-sections -fdata-sections -DAES128=1 -DECB=1 -DCBC=0 -DCTR=0 -Wall"
LF="--specs=nano.specs --specs=nosys.specs -u _printf_float -T stm32f405.ld -nostartfiles -Wl,--gc-sections"
SRC="geotrust_mcu.c geotrust_verify.c tiny-AES-c/aes.c"
echo "== footprint of the verification kernels (GeoTrust+ final design) =="
arm-none-eabi-gcc $CF -c geotrust_verify.c -o /tmp/geotrust_verify.o && arm-none-eabi-size /tmp/geotrust_verify.o
arm-none-eabi-gcc $CF $LF bench_hw.c $SRC -lm -o bench_hw.elf
arm-none-eabi-objcopy -O binary bench_hw.elf bench_hw.bin
arm-none-eabi-size bench_hw.elf
if [ "$1" = "--qemu" ]; then
  arm-none-eabi-gcc $CF -DQEMU_TEST $LF bench_hw.c $SRC -lm -o /tmp/bench_hw_qemu.elf
  timeout 300 qemu-system-arm -M netduinoplus2 -nographic -semihosting -icount shift=0 -kernel /tmp/bench_hw_qemu.elf
fi
