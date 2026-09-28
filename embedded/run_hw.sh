#!/bin/bash
# Flash bench_hw.elf through an ST-LINK, capture the semihosting output and parse it.
# Usage:  bash run_hw.sh "STM32F4DISCOVERY (F407VG)" [interface/stlink.cfg]
set -e
cd "$(dirname "$0")"
BOARD=${1:-unknown}
IFACE=${2:-interface/stlink.cfg}
[ -f bench_hw.elf ] || bash build_hw.sh
LOG=hw_log_$(date +%Y%m%d_%H%M%S).txt
openocd -f "$IFACE" -f target/stm32f4x.cfg \
  -c "init" -c "reset halt" -c "flash write_image erase bench_hw.elf" -c "verify_image bench_hw.elf" \
  -c "arm semihosting enable" -c "reset run" > "$LOG" 2>&1 &
PID=$!
for i in $(seq 1 60); do
  sleep 1
  grep -q "^DONE" "$LOG" && break
  kill -0 $PID 2>/dev/null || break
done
kill $PID 2>/dev/null || true
wait $PID 2>/dev/null || true
grep -E "GEOTRUST-HW|KERNEL|DONE|Error" "$LOG" || { echo "no benchmark output - see $LOG"; exit 1; }
python3 parse_hw.py "$LOG" --board "$BOARD"
