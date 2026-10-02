"""Re-render the paper figures for PLOS ONE: Arial text at 8-12 pt, RGB TIFF with LZW compression at 300 dpi.

PLOS figure rules (journals.plos.org/plosone/s/figures): TIFF or EPS, 300-600 dpi, width 789-2250 px (2.63-7.5 in),
height <= 2625 px, RGB, no alpha channel, LZW, <= 10 MB, text in Arial/Times/Symbol at 8-12 pt, no figure number or
caption inside the image.  The original figure scripts are run unchanged; this wrapper sets the font, clamps every
font size to [8, 12] pt and redirects their output to eval/figures/plos/src (the paper figures in eval/figures are kept);
the TIFF files go to eval/figures/plos.
Run from the project root:  python scripts/plos_figs.py
"""
import os, runpy, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from PIL import Image

OUT = "eval/figures/plos/src"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "mathtext.fontset": "custom",
                     "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
                     "pdf.fonttype": 42})

_set_size = FontProperties.set_size


def set_size(self, size):
    _set_size(self, size)
    self._size = min(max(self._size, 8.0), 12.0)


FontProperties.set_size = set_size
_savefig = Figure.savefig


def savefig(self, fname, *a, **k):
    base = os.path.splitext(os.path.basename(str(fname)))[0]
    _savefig(self, os.path.join(OUT, base + ".pdf"))
    _savefig(self, os.path.join(OUT, base + ".png"), dpi=300)


Figure.savefig = savefig

# figure in the paper -> PLOS file name (order of first citation in the PLOS manuscript)
PLOS = {"fig_zhou_diag": "Fig1", "fig_rfc_attacks_dense": "Fig2", "fig_rfc_fraction": "Fig3",
        "fig_rfc_attacks_sparse": "Fig4", "fig_sensitivity": "Fig5", "fig_netsci": "Fig6"}

if __name__ == "__main__":
    for script in ["scripts/figures_final.py", "scripts/fig_sensitivity.py", "scripts/fig_netsci.py"]:
        sys.argv = [script]
        runpy.run_path(script, run_name="__main__")
        plt.close("all")
    os.makedirs("eval/figures/plos", exist_ok=True)
    for src, dst in PLOS.items():
        im = Image.open(os.path.join(OUT, src + ".png"))
        rgb = Image.new("RGB", im.size, "white")
        rgb.paste(im, mask=im.split()[3] if im.mode == "RGBA" else None)
        out = f"eval/figures/plos/{dst}.tif"
        rgb.save(out, compression="tiff_lzw", dpi=(300, 300))
        w, h = rgb.size
        print(f"{dst}.tif <- {src}: {w}x{h} px = {w / 300:.2f}x{h / 300:.2f} in, "
              f"{os.path.getsize(out) / 1e6:.2f} MB, mode {rgb.mode}")
