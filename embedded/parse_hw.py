"""Parse the semihosting log of bench_hw.elf into results_hw.json and print LaTeX rows for the
overhead table.  Usage:  python parse_hw.py hw_log.txt [--board "STM32F4DISCOVERY (F407VG)"]"""
import argparse, json, re, statistics

ap = argparse.ArgumentParser()
ap.add_argument("log")
ap.add_argument("--board", default="unknown")
ap.add_argument("--out", default="results_hw.json")
a = ap.parse_args()

txt = open(a.log, encoding="utf8", errors="replace").read()
h = re.search(r"GEOTRUST-HW clock_hz=(\d+) flash_ws=(\d+) timer=(\w+)", txt)
if not h:
    raise SystemExit("no GEOTRUST-HW header in log")
clock, ws, timer = int(h[1]), int(h[2]), h[3]
k = {m[1]: dict(cycles_med=int(m[2]), cycles_min=int(m[3]), cycles_max=int(m[4]), check=int(m[5]))
     for m in re.finditer(r"KERNEL (\S+)\s+cycles_med=(\d+) min=(\d+) max=(\d+) us=[\d.]+ check=(-?\d+)", txt)}
if "DONE" not in txt:
    print("warning: log has no DONE line (benchmark incomplete)")
for v in k.values():
    v["us"] = round(v["cycles_med"] * 1e6 / clock, 2)
real = timer == "DWT"
res = {"board": a.board, "clock_hz": clock, "flash_wait_states": ws, "timer": timer,
       "cycle_accurate": real and clock == 168000000 and ws == 5,
       "note": ("DWT CYCCNT on hardware" if real else "QEMU functional test: counts are instructions, not cycles"),
       "kernels": k}
json.dump(res, open(a.out, "w"), indent=1)
print(f"{a.board}: {clock / 1e6:.0f} MHz, {ws} WS, timer {timer} -> {a.out}")
if not res["cycle_accurate"]:
    print("WARNING: not a cycle-accurate 168 MHz / 5 WS measurement")
checks = {"RFC 4493 CMAC": k.get("cmac8_229B", {}).get("check") == 1,
          "liars convicted": all(k.get(f"reciprocity_n{n}", {}).get("check") == n // 10 for n in (12, 20, 30))}
print("checks:", checks)
u = lambda n: k[n]["us"]
print("\n% LaTeX rows (overhead table)")
print(f"CMAC verify, 229\\,B RU & {u('cmac8_229B') / 1000:.2f}\\,ms ({k['cmac8_229B']['cycles_med']} cycles) \\\\")
print(f"Reciprocity graph, 20 neighbours & {u('reciprocity_n20'):.0f}\\,$\\mu$s \\\\")
print(f"Kinematic + clamp + watchdog, 20 nb. & {u('kin_clamp_wd_n20'):.0f}\\,$\\mu$s \\\\")
print(f"Whole interval, 12/20/30 nb. & {u('round_total_n12') / 1000:.1f} / {u('round_total_n20') / 1000:.1f} / "
      f"{u('round_total_n30') / 1000:.1f}\\,ms \\\\")
print(f"CPU share at a 100\\,ms interval (20 nb.) & {u('round_total_n20') / 1000:.1f}\\,\\% \\\\")
