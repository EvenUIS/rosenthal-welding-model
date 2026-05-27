"""
================================================================================
  ROSENTHAL WELDING HEAT MODEL — Example Run
  Run this script to generate all output figures.
================================================================================

Usage:
    cd example
    python run_example.py

Output figures are saved to the outputs/ folder.
================================================================================
"""

import os
import sys

# add parent directory to path so the package can be found
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from rosenthal.model import (
    haz_width_estimate, cooling_rate_centreline,
    max_electrode_temp, HI_kJ_mm, alpha,
    v, Q_net
)
from rosenthal.plots import plot_main, plot_haz, plot_tensile, plot_hardness
from rosenthal.parameters import (
    V, I, te, v_mm_min, k, T0, thickness,
    cooling_type, melting_temp, AC1_temp
)

# ── output folder ──
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUT, exist_ok=True)

# ── print summary ──
print("=" * 60)
print("  ROSENTHAL WELDING HEAT MODEL  —  Parameter Summary")
print("=" * 60)
print(f"  Voltage               : {V} V")
print(f"  Current               : {I} A")
print(f"  Thermal efficiency    : {te}")
print(f"  Net heat input (Q)    : {Q_net:.1f} W")
print(f"  Heat input per length : {HI_kJ_mm:.2f} kJ/mm")
print(f"  Welding speed         : {v_mm_min} mm/min  ({v*1000:.2f} mm/s)")
print(f"  Thermal diffusivity α : {alpha:.3e} m²/s")
print(f"  Cooling type          : {cooling_type}")
print(f"  Plate thickness       : {thickness} mm")
print(f"  Max electrode temp    : {max_electrode_temp:.0f} °C")
print("-" * 60)
CR_800 = cooling_rate_centreline(800)
print(f"  Cooling rate at 800°C : {abs(CR_800):.1f} °C/s")

# ── HAZ widths ──
haz_w     = haz_width_estimate(melting_temp)
haz_w_ac1 = haz_width_estimate(AC1_temp)
print(f"  Fusion zone half-width (>{melting_temp}°C) : {haz_w:.2f} mm" if haz_w
      else "  Fusion zone not detected")
print(f"  HAZ half-width (>{AC1_temp}°C AC1)         : {haz_w_ac1:.2f} mm" if haz_w_ac1
      else "  HAZ not detected")
print("=" * 60)

# ── generate all plots ──
print("\n  Generating figures...")
plot_main    (os.path.join(OUT, "rosenthal_welding_results.png"))
plot_haz     (haz_w, haz_w_ac1, os.path.join(OUT, "rosenthal_HAZ_width.png"))
plot_tensile (haz_w, haz_w_ac1, os.path.join(OUT, "rosenthal_tensile_strength.png"))
HV_base, HV_weld = plot_hardness(haz_w, haz_w_ac1,
                                  os.path.join(OUT, "rosenthal_hardness.png"))

print(f"\n  Base material hardness : {HV_base:.0f} HV")
print(f"  Weld metal hardness    : {HV_weld:.0f} HV")
print("\n  All figures saved to outputs/")
