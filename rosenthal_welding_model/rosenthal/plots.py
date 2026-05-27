"""
================================================================================
  ROSENTHAL WELDING HEAT MODEL — Plotting Functions
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec

from rosenthal.model import (
    rosenthal_temperature, thermal_cycle, thermal_cycle_xslice,
    tensile_strength_profile, hardness_profile,
    haz_width_estimate, max_electrode_temp,
    v, Q_net, alpha
)
from rosenthal.parameters import (
    V, I, te, v_mm_min, k, T0,
    AC1_temp, AC3_temp, max_elastic_temp, melting_temp,
    cooling_type, yt
)

# ── shared colour scheme ──
DARK_BG   = "#0d1117"
TEXT_COL  = "#c9d1d9"
TITLE_COL = "#f0f6fc"
SPINE_COL = "#30363d"
LEGEND_BG = "#161b22"

ISO_COLORS = ["#ff4040", "#ff9500", "#ffe066", "#7bc8f6"]

REF_LINES = [
    (melting_temp,     "#ff4040", "Melting"),
    (AC3_temp,         "#ff9500", "AC3"),
    (AC1_temp,         "#ffe066", "AC1"),
    (max_elastic_temp, "#7bc8f6", "Max elastic"),
]


def _style_ax(ax):
    """Apply common dark-theme styling to an axes."""
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors=TEXT_COL)
    for spine in ax.spines.values():
        spine.set_edgecolor(SPINE_COL)


def _legend(ax, **kwargs):
    ax.legend(facecolor=LEGEND_BG, edgecolor=SPINE_COL,
              labelcolor=TEXT_COL, **kwargs)


# ==============================
# PLOT 1 — MAIN 4-PANEL FIGURE
# ==============================
def plot_main(output_path):
    """
    Generate the main 4-panel figure:
      - 2D temperature field
      - Centreline temperature profile
      - Thermal cycles (Y direction)
      - Transverse temperature profiles (X slices)
    """
    iso_temps = {
        "Melting front":              melting_temp,
        "Austenite / AC3 (~900 °C)": AC3_temp,
        "AC1 / Lower transform":      AC1_temp,
        "Max elastic limit (600 °C)": max_elastic_temp,
    }

    x_mm = np.linspace(-80, 20, 500)
    y_mm = np.linspace(-60, 60, 500)
    X_mm, Y_mm = np.meshgrid(x_mm, y_mm)
    X_m, Y_m   = X_mm * 1e-3, Y_mm * 1e-3

    T_field = rosenthal_temperature(X_m, Y_m)
    T_field = np.clip(T_field, T0, max_electrode_temp)

    fig = plt.figure(figsize=(18, 13), facecolor=DARK_BG)
    gs  = GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.38,
                   left=0.07, right=0.97, top=0.91, bottom=0.08)

    ax1 = fig.add_subplot(gs[0, :2])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, :2])
    ax4 = fig.add_subplot(gs[1, 2])

    # ── ax1: temperature field ──
    cmap_weld = mcolors.LinearSegmentedColormap.from_list(
        "weld",
        [DARK_BG, "#1a2940", "#1e4d6b", "#e65c00", "#f9d423", "#ffffff"],
        N=512
    )
    T_plot = np.clip(T_field, T0, 2500)
    im = ax1.contourf(X_mm, Y_mm, T_plot, levels=256, cmap=cmap_weld)
    cb = fig.colorbar(im, ax=ax1, pad=0.02)
    cb.set_label("Temperature [°C]", color=TEXT_COL, fontsize=10)
    cb.ax.yaxis.set_tick_params(color=TEXT_COL)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT_COL)

    iso_labels = []
    for (label, Tiso), col in zip(iso_temps.items(), ISO_COLORS):
        try:
            ax1.contour(X_mm, Y_mm, T_field, levels=[Tiso],
                        colors=[col], linewidths=1.4, linestyles="--")
            iso_labels.append(Patch(facecolor=col, label=f"{Tiso}°C  {label}"))
        except Exception:
            pass

    ax1.plot(0, 0, "o", color="white", ms=7, zorder=10)
    ax1.annotate("Heat\nsource", (0, 0), (5, 8), color="white", fontsize=7.5,
                 arrowprops=dict(arrowstyle="->", color="white", lw=0.8))
    _style_ax(ax1)
    ax1.set_xlabel("X — Along welding direction [mm]", color=TEXT_COL)
    ax1.set_ylabel("Y — Transverse [mm]", color=TEXT_COL)
    ax1.set_title("Rosenthal Temperature Field (2D, moving frame)",
                  color=TITLE_COL, fontweight="bold", fontsize=12)
    ax1.legend(handles=iso_labels, loc="lower left", facecolor=LEGEND_BG,
               edgecolor=SPINE_COL, labelcolor=TEXT_COL, fontsize=7.5, framealpha=0.85)

    # ── ax2: centreline profile ──
    x_cl    = np.linspace(-80, 10, 600) * 1e-3
    T_cl    = rosenthal_temperature(x_cl, np.zeros_like(x_cl))
    x_cl_mm = x_cl * 1000

    ax2.plot(x_cl_mm, T_cl, color="#f9d423", lw=1.8)
    for (label, Tiso), col in zip(iso_temps.items(), ISO_COLORS):
        ax2.axhline(Tiso, color=col, lw=0.9, ls="--", alpha=0.75)
        ax2.text(-78, Tiso + 30, f"{Tiso}°C", color=col, fontsize=6.5)
    ax2.axvline(0, color="white", lw=0.8, ls=":")
    _style_ax(ax2)
    ax2.set_xlabel("X [mm]", color=TEXT_COL)
    ax2.set_ylabel("Temperature [°C]", color=TEXT_COL)
    ax2.set_title("Centreline Profile", color=TITLE_COL, fontweight="bold", fontsize=10)
    ax2.set_ylim(0, 2600)

    # ── ax3: thermal cycles Y direction ──
    y_positions  = [0, 2, 4, 8, 12, 20]
    cycle_colors = plt.cm.plasma(np.linspace(0.15, 0.95, len(y_positions)))
    for yp, col in zip(y_positions, cycle_colors):
        t_arr, T_arr = thermal_cycle(yp)
        ax3.plot(t_arr, T_arr, color=col, lw=1.5, label=f"y = {yp} mm")
    for Tref, col, lbl in REF_LINES:
        ax3.axhline(Tref, color=col, lw=0.8, ls="--", alpha=0.7)
        ax3.text(-29, Tref + 20, lbl, color=col, fontsize=7)
    _style_ax(ax3)
    ax3.set_xlabel("Time [s]  (negative = source approaching, positive = source passed)",
                   color=TEXT_COL)
    ax3.set_ylabel("Temperature [°C]", color=TEXT_COL)
    ax3.set_title("Thermal Cycles at Various Transverse Distances from Weld Centreline",
                  color=TITLE_COL, fontweight="bold", fontsize=11)
    ax3.set_xlim(-30, 120)
    ax3.set_ylim(0, 2200)
    _legend(ax3, fontsize=8, ncol=3, loc="upper right")

    # ── ax4: transverse profiles at fixed X ──
    x_positions   = [-40, -20, -10, -5, 0, 10]
    xcycle_colors = plt.cm.cool(np.linspace(0.15, 0.95, len(x_positions)))
    for xp, col in zip(x_positions, xcycle_colors):
        y_arr_mm, T_arr = thermal_cycle_xslice(xp)
        ax4.plot(y_arr_mm, T_arr, color=col, lw=1.5, label=f"x = {xp} mm")
    for Tref, col, lbl in REF_LINES:
        ax4.axhline(Tref, color=col, lw=0.8, ls="--", alpha=0.7)
    _style_ax(ax4)
    ax4.set_xlabel("Y — Transverse distance [mm]", color=TEXT_COL)
    ax4.set_ylabel("Temperature [°C]", color=TEXT_COL)
    ax4.set_title("Transverse Temperature Profile\nat Fixed X Positions",
                  color=TITLE_COL, fontweight="bold", fontsize=9)
    ax4.set_xlim(-60, 60)
    ax4.set_ylim(0, 2200)
    _legend(ax4, fontsize=7.5, ncol=2, loc="upper right")

    fig.suptitle(
        f"Rosenthal Welding Heat Model  |  V={V}V  I={I}A  η={te}  "
        f"v={v_mm_min}mm/min  k={k}W/m·K  Cooling: {cooling_type}",
        color=TITLE_COL, fontsize=11, fontweight="bold", y=0.97
    )
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  Figure saved → {output_path}")


# ==============================
# PLOT 2 — HAZ WIDTH
# ==============================
def plot_haz(haz_w, haz_w_ac1, output_path):
    """Generate the HAZ width estimate plot."""
    y_scan_mm = np.linspace(-60, 60, 1000)
    y_scan_m  = y_scan_mm * 1e-3
    T_scan    = rosenthal_temperature(np.zeros_like(y_scan_m), y_scan_m)

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=DARK_BG)
    _style_ax(ax)

    ax.plot(y_scan_mm, T_scan, color="#f9d423", lw=2, label="Temperature profile at X=0")
    ax.axvline(0, color="white", lw=0.8, ls=":", alpha=0.5)

    ax.axhspan(AC1_temp,    AC3_temp,          alpha=0.15, color="#ff9500",
               label=f"Intercritical HAZ  ({AC1_temp}–{AC3_temp}°C)")
    ax.axhspan(AC3_temp,    melting_temp,       alpha=0.15, color="#ff4040",
               label=f"Full austenite HAZ  ({AC3_temp}–{melting_temp}°C)")
    ax.axhspan(melting_temp, max_electrode_temp, alpha=0.25, color="#ffffff",
               label=f"Fusion zone  (>{melting_temp}°C)")

    for Tref, col, lbl in REF_LINES:
        ax.axhline(Tref, color=col, lw=1.0, ls="--", alpha=0.8)
        ax.text(-39, Tref + 20, f"{Tref}°C  {lbl}", color=col, fontsize=8)

    if haz_w:
        ax.axvline( haz_w, color="#ff4040", lw=1.2, ls=":")
        ax.axvline(-haz_w, color="#ff4040", lw=1.2, ls=":")
        ax.text( haz_w + 0.3, 100, f"Fusion\n+{haz_w:.2f} mm", color="#ff4040", fontsize=8)
        ax.text(-haz_w + 0.3, 100, f"Fusion\n-{haz_w:.2f} mm", color="#ff4040", fontsize=8)
    if haz_w_ac1:
        ax.axvline( haz_w_ac1, color="#ffe066", lw=1.2, ls=":")
        ax.axvline(-haz_w_ac1, color="#ffe066", lw=1.2, ls=":")
        ax.text( haz_w_ac1 + 0.3, 250, f"HAZ\n+{haz_w_ac1:.2f} mm", color="#ffe066", fontsize=8)
        ax.text(-haz_w_ac1 - 5,   250, f"HAZ\n-{haz_w_ac1:.2f} mm", color="#ffe066", fontsize=8)

    ax.set_xlabel("Y — Transverse distance from centreline [mm]", color=TEXT_COL)
    ax.set_ylabel("Temperature [°C]", color=TEXT_COL)
    ax.set_title(f"HAZ Width Estimate  |  V={V}V  I={I}A  η={te}  v={v_mm_min}mm/min",
                 color=TITLE_COL, fontweight="bold", fontsize=11)
    ax.set_xlim(-40, 40)
    ax.set_ylim(0, 4000)
    _legend(ax, fontsize=8)

    fig.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  HAZ figure saved → {output_path}")


# ==============================
# PLOT 3 — TENSILE STRENGTH
# ==============================
def plot_tensile(haz_w, haz_w_ac1, output_path):
    """Generate the tensile strength profile plot."""
    y_ts_mm       = np.linspace(-60, 60, 1000)
    yt_w, yt_weld = tensile_strength_profile(y_ts_mm)

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=DARK_BG)
    _style_ax(ax)

    ax.plot(y_ts_mm, yt_w, color="#f9d423", lw=2, label="Tensile strength estimate yt_w")
    ax.axhline(yt / 1e6, color="#ff4040", lw=1.2, ls="--",
               label=f"Base material yt = {yt/1e6:.0f} MPa")
    ax.axvline(0, color="white", lw=0.8, ls=":", alpha=0.5)

    ax.axvspan(-haz_w,     -haz_w_ac1, alpha=0.10, color="#ff4040", label="Full austenite HAZ")
    ax.axvspan( haz_w_ac1,  haz_w,     alpha=0.10, color="#ff4040")
    ax.axvspan(-haz_w_ac1,  0,         alpha=0.08, color="#ff9500", label="Intercritical HAZ")
    ax.axvspan( 0,           haz_w_ac1, alpha=0.08, color="#ff9500")

    if haz_w:
        ax.axvline( haz_w, color="#ff4040", lw=1.0, ls=":")
        ax.axvline(-haz_w, color="#ff4040", lw=1.0, ls=":")
        ax.text( haz_w + 0.3, yt/1e6 * 0.72, f"Fusion\n+{haz_w:.2f}mm", color="#ff4040", fontsize=7.5)
        ax.text(-haz_w - 5,   yt/1e6 * 0.72, f"Fusion\n-{haz_w:.2f}mm", color="#ff4040", fontsize=7.5)
    if haz_w_ac1:
        ax.axvline( haz_w_ac1, color="#ffe066", lw=1.0, ls=":")
        ax.axvline(-haz_w_ac1, color="#ffe066", lw=1.0, ls=":")
        ax.text( haz_w_ac1 + 0.3, yt/1e6 * 0.60, f"HAZ\n+{haz_w_ac1:.2f}mm", color="#ffe066", fontsize=7.5)
        ax.text(-haz_w_ac1 - 5,   yt/1e6 * 0.60, f"HAZ\n-{haz_w_ac1:.2f}mm", color="#ffe066", fontsize=7.5)

    ax.set_xlabel("Y — Transverse distance from centreline [mm]", color=TEXT_COL)
    ax.set_ylabel("Tensile Strength [MPa]", color=TEXT_COL)
    ax.set_title(f"Tensile Strength Estimate  |  V={V}V  I={I}A  η={te}  v={v_mm_min}mm/min",
                 color=TITLE_COL, fontweight="bold", fontsize=11)
    ax.set_xlim(-40, 40)
    ax.set_ylim((yt/1e6) - 200, (yt/1e6) + 400)
    _legend(ax, fontsize=8)

    fig.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  Tensile strength figure saved → {output_path}")


# ==============================
# PLOT 4 — HARDNESS
# ==============================
def plot_hardness(haz_w, haz_w_ac1, output_path):
    """Generate the Vickers hardness profile plot."""
    y_ts_mm       = np.linspace(-60, 60, 1000)
    yt_w, yt_weld = tensile_strength_profile(y_ts_mm)
    HV_profile    = hardness_profile(yt_w)

    HV_base = 0.3328 * (yt / 1e6) - 14.144
    HV_weld = 0.3328 * yt_weld    - 14.144

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=DARK_BG)
    _style_ax(ax)

    ax.plot(y_ts_mm, HV_profile, color="#7bc8f6", lw=2, label="Hardness profile HV")
    ax.axhline(HV_base, color="#ff4040", lw=1.2, ls="--",
               label=f"Base material HV = {HV_base:.0f}")
    ax.axhline(HV_weld, color="#f9d423", lw=1.2, ls="--",
               label=f"Weld metal HV = {HV_weld:.0f}")
    ax.axvline(0, color="white", lw=0.8, ls=":", alpha=0.5)

    ax.axvspan(-haz_w,     -haz_w_ac1, alpha=0.10, color="#ff4040", label="Full austenite HAZ")
    ax.axvspan( haz_w_ac1,  haz_w,     alpha=0.10, color="#ff4040")
    ax.axvspan(-haz_w_ac1,  0,         alpha=0.08, color="#ff9500", label="Intercritical HAZ")
    ax.axvspan( 0,           haz_w_ac1, alpha=0.08, color="#ff9500")

    if haz_w:
        ax.axvline( haz_w, color="#ff4040", lw=1.0, ls=":")
        ax.axvline(-haz_w, color="#ff4040", lw=1.0, ls=":")
        ax.text( haz_w + 0.3, HV_base * 0.72, f"Fusion\n+{haz_w:.2f}mm", color="#ff4040", fontsize=7.5)
        ax.text(-haz_w - 5,   HV_base * 0.72, f"Fusion\n-{haz_w:.2f}mm", color="#ff4040", fontsize=7.5)
    if haz_w_ac1:
        ax.axvline( haz_w_ac1, color="#ffe066", lw=1.0, ls=":")
        ax.axvline(-haz_w_ac1, color="#ffe066", lw=1.0, ls=":")
        ax.text( haz_w_ac1 + 0.3, HV_base * 0.60, f"HAZ\n+{haz_w_ac1:.2f}mm", color="#ffe066", fontsize=7.5)
        ax.text(-haz_w_ac1 - 5,   HV_base * 0.60, f"HAZ\n-{haz_w_ac1:.2f}mm", color="#ffe066", fontsize=7.5)

    ax.set_xlabel("Y — Transverse distance from centreline [mm]", color=TEXT_COL)
    ax.set_ylabel("Hardness [HV]", color=TEXT_COL)
    ax.set_title(f"Hardness Profile  |  V={V}V  I={I}A  η={te}  v={v_mm_min}mm/min",
                 color=TITLE_COL, fontweight="bold", fontsize=11)
    ax.set_xlim(-40, 40)
    ax.set_ylim(HV_base - 50, HV_weld + 50)
    _legend(ax, fontsize=8)

    fig.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  Hardness figure saved → {output_path}")

    return HV_base, HV_weld
