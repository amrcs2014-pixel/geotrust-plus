#!/bin/bash
# Build for STM32F405 (Cortex-M4F, -O2, hard float) and run on QEMU netduinoplus2.
# Run inside WSL:  bash build_and_run.sh
set -e
cd "$(dirname "$0")"
CF="-mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16 -O2 -ffunction-sections -fdata-sections -DAES128=1 -DECB=1 -DCBC=0 -DCTR=0"
arm-none-eabi-gcc $CF -c geotrust_mcu.c -o geotrust_mcu.o
arm-none-eabi-gcc $CF -c tiny-AES-c/aes.c -o aes.o
echo "== library footprint (GeoTrust kernels + AES) =="
arm-none-eabi-size -t geotrust_mcu.o aes.o
arm-none-eabi-gcc $CF --specs=nano.specs --specs=nosys.specs -u _printf_float -T stm32f405.ld -nostartfiles \
  -Wl,--gc-sections bench.c geotrust_mcu.o aes.o -lm -o bench.elf
arm-none-eabi-size bench.elf
timeout 300 qemu-system-arm -M netduinoplus2 -nographic -semihosting -icount shift=0 -kernel bench.elf
