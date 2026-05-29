"""
================================================================================
  ROSENTHAL WELDING HEAT MODEL — Example Run
  Run this script to generate all output figures.
================================================================================
Usage:
    python example/run_example.py

Output figures are saved to the outputs/ folder.
================================================================================
"""

import os
import sys

# Add parent directory to path so the package can be found
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rosenthal.parameters import WeldingConditions
from rosenthal.model import haz_width_estimate, cooling_rate_centreline
from rosenthal.plots import plot_main, plot_haz, plot_tensile, plot_hardness

# ── output folder ──
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUT, exist_ok=True)

# ── build conditions from default preset ──
# To run with different parameters, change values here:
#   cond = WeldingConditions(V=25, I=200, v_mm_min=150)
cond = WeldingConditions()

# ── print summary ──
print("=" * 60)
print("  ROSENTHAL WELDING HEAT MODEL  —  Parameter Summary")
print("=" * 60)
print(f"  Voltage               : {cond.V} V")
print(f"  Current               : {cond.I} A")
print(f"  Thermal efficiency    : {cond.te}")
print(f"  Net heat input (Q)    : {cond.Q_net:.1f} W")
print(f"  Heat input per length : {cond.HI_kJ_mm:.2f} kJ/mm")
print(f"  Welding speed         : {cond.v_mm_min} mm/min  ({cond.v * 1000:.2f} mm/s)")
print(f"  Thermal diffusivity α : {cond.alpha:.3e} m²/s")
print(f"  Cooling type          : {cond.cooling_type}")
print(f"  Plate thickness       : {cond.thickness} mm")
print(f"  Max electrode temp    : {cond.max_electrode_temp:.0f} °C")
print("-" * 60)

cr_800 = cooling_rate_centreline(800, cond)
print(f"  Cooling rate at 800°C : {abs(cr_800):.1f} °C/s")

# ── HAZ widths ──
haz_w = haz_width_estimate(cond.melting_temp, cond)
haz_w_ac1 = haz_width_estimate(cond.AC1_temp, cond)

if haz_w:
    print(f"  Fusion zone half-width (>{cond.melting_temp}°C) : {haz_w:.2f} mm")
else:
    print("  Fusion zone not detected")

if haz_w_ac1:
    print(f"  HAZ half-width (>{cond.AC1_temp}°C AC1)         : {haz_w_ac1:.2f} mm")
else:
    print("  HAZ not detected")

print("=" * 60)

# ── generate all plots ──
print("\n  Generating figures...")

plot_main(os.path.join(OUT, "rosenthal_welding_results.png"), cond)
plot_haz(haz_w, haz_w_ac1, os.path.join(OUT, "rosenthal_HAZ_width.png"), cond)
plot_tensile(haz_w, haz_w_ac1, os.path.join(OUT, "rosenthal_tensile_strength.png"), cond)
hv_base, hv_weld = plot_hardness(
    haz_w, haz_w_ac1, os.path.join(OUT, "rosenthal_hardness.png"), cond
)

print(f"\n  Base material hardness : {hv_base:.0f} HV")
print(f"  Weld metal hardness    : {hv_weld:.0f} HV")
print("\n  All figures saved to outputs/")
