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

from rosenthal.parameters import WeldingConditions, DEFAULT_CONDITIONS
from rosenthal.model import (
    rosenthal_temperature,
    thermal_cycle,
    thermal_cycle_xslice,
    tensile_strength_profile,
    hardness_profile,
    haz_width_estimate,
)

# ── shared colour scheme ──
DARK_BG = "#0d1117"
TEXT_COL = "#c9d1d9"
TITLE_COL = "#f0f6fc"
SPINE_COL = "#30363d"
LEGEND_BG = "#161b22"

ISO_COLORS = ["#ff4040", "#ff9500", "#ffe066", "#7bc8f6"]


def _ref_lines(cond):
    """Return reference isotherm lines for the given conditions."""
    return [
        (cond.melting_temp,     "#ff4040", "Melting"),
        (cond.AC3_temp,         "#ff9500", "AC3"),
        (cond.AC1_temp,         "#ffe066", "AC1"),
        (cond.max_elastic_temp, "#7bc8f6", "Max elastic"),
    ]


def _style_ax(ax):
    """Apply common dark-theme styling to an axes."""
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors=TEXT_COL)
    for spine in ax.spines.values():
        spine.set_edgecolor(SPINE_COL)


def _legend(ax, **kwargs):
    """Add a dark-themed legend to an axes."""
    ax.legend(
        facecolor=LEGEND_BG,
        edgecolor=SPINE_COL,
        labelcolor=TEXT_COL,
        **kwargs,
    )


def _param_label(cond):
    """Return a short parameter summary string for plot titles."""
    return (
        f"V={cond.V}V  I={cond.I}A  η={cond.te}  "
        f"v={cond.v_mm_min}mm/min  k={cond.k}W/m·K  "
        f"Cooling: {cond.cooling_type}"
    )


# ==============================
# PLOT 1 — MAIN 4-PANEL FIGURE
# ==============================
def plot_main(output_path, cond=None):
    """
    Generate the main 4-panel figure.

    Panels
    ------
    - 2D temperature field with isotherms
    - Centreline temperature profile
    - Thermal cycles (Y direction)
    - Transverse temperature profiles (X slices)

    Parameters
    ----------
    output_path : str
        File path for the saved figure.
    cond : WeldingConditions, optional
        Welding parameters. Uses DEFAULT_CONDITIONS if not provided.
    """
    if cond is None:
        cond = DEFAULT_CONDITIONS

    ref_lines = _ref_lines(cond)

    iso_temps = {
        "Melting front":              cond.melting_temp,
        "Austenite / AC3 (~900 °C)": cond.AC3_temp,
        "AC1 / Lower transform":      cond.AC1_temp,
        "Max elastic limit (600 °C)": cond.max_elastic_temp,
    }

    x_mm = np.linspace(-80, 20, 500)
    y_mm = np.linspace(-60, 60, 500)
    x_grid, y_grid = np.meshgrid(x_mm, y_mm)
    t_field = rosenthal_temperature(x_grid * 1e-3, y_grid * 1e-3, cond)
    t_field = np.clip(t_field, cond.T0, cond.max_electrode_temp)

    fig = plt.figure(figsize=(18, 13), facecolor=DARK_BG)
    gs = GridSpec(
        2, 3, figure=fig, hspace=0.48, wspace=0.38,
        left=0.07, right=0.97, top=0.91, bottom=0.08,
    )
    ax1 = fig.add_subplot(gs[0, :2])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, :2])
    ax4 = fig.add_subplot(gs[1, 2])

    # ── ax1: temperature field ──
    cmap_weld = mcolors.LinearSegmentedColormap.from_list(
        "weld",
        [DARK_BG, "#1a2940", "#1e4d6b", "#e65c00", "#f9d423", "#ffffff"],
        N=512,
    )
    img = ax1.contourf(
        x_mm, y_mm, np.clip(t_field, cond.T0, 2500), levels=256, cmap=cmap_weld
    )
    cb = fig.colorbar(img, ax=ax1, pad=0.02)
    cb.set_label("Temperature [°C]", color=TEXT_COL, fontsize=10)
    cb.ax.yaxis.set_tick_params(color=TEXT_COL)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT_COL)

    iso_labels = []
    for (label, t_iso), col in zip(iso_temps.items(), ISO_COLORS):
        try:
            ax1.contour(
                x_mm, y_mm, t_field, levels=[t_iso],
                colors=[col], linewidths=1.4, linestyles="--",
            )
            iso_labels.append(Patch(facecolor=col, label=f"{t_iso}°C  {label}"))
        except Exception:  # noqa: BLE001
            pass

    ax1.plot(0, 0, "o", color="white", ms=7, zorder=10)
    ax1.annotate(
        "Heat\nsource", (0, 0), (5, 8), color="white", fontsize=7.5,
        arrowprops=dict(arrowstyle="->", color="white", lw=0.8),
    )
    _style_ax(ax1)
    ax1.set_xlabel("X — Along welding direction [mm]", color=TEXT_COL)
    ax1.set_ylabel("Y — Transverse [mm]", color=TEXT_COL)
    ax1.set_title(
        "Rosenthal Temperature Field (2D, moving frame)",
        color=TITLE_COL, fontweight="bold", fontsize=12,
    )
    ax1.legend(
        handles=iso_labels, loc="lower left", facecolor=LEGEND_BG,
        edgecolor=SPINE_COL, labelcolor=TEXT_COL, fontsize=7.5, framealpha=0.85,
    )

    # ── ax2: centreline profile ──
    x_cl = np.linspace(-80, 10, 600) * 1e-3
    t_cl = rosenthal_temperature(x_cl, np.zeros_like(x_cl), cond)

    ax2.plot(x_cl * 1000, t_cl, color="#f9d423", lw=1.8)
    for (label, t_iso), col in zip(iso_temps.items(), ISO_COLORS):
        ax2.axhline(t_iso, color=col, lw=0.9, ls="--", alpha=0.75)
        ax2.text(-78, t_iso + 30, f"{t_iso}°C", color=col, fontsize=6.5)
    ax2.axvline(0, color="white", lw=0.8, ls=":")
    _style_ax(ax2)
    ax2.set_xlabel("X [mm]", color=TEXT_COL)
    ax2.set_ylabel("Temperature [°C]", color=TEXT_COL)
    ax2.set_title(
        "Centreline Profile", color=TITLE_COL, fontweight="bold", fontsize=10
    )
    ax2.set_ylim(0, 2600)

    # ── ax3: thermal cycles Y direction ──
    y_positions = [0, 2, 4, 8, 12, 20]
    cycle_colors = plt.cm.plasma(np.linspace(0.15, 0.95, len(y_positions)))
    for yp, col in zip(y_positions, cycle_colors):
        t_arr, t_cycle = thermal_cycle(yp, cond)
        ax3.plot(t_arr, t_cycle, color=col, lw=1.5, label=f"y = {yp} mm")
    for t_ref, col, lbl in ref_lines:
        ax3.axhline(t_ref, color=col, lw=0.8, ls="--", alpha=0.7)
        ax3.text(-29, t_ref + 20, lbl, color=col, fontsize=7)
    _style_ax(ax3)
    ax3.set_xlabel(
        "Time [s]  (negative = source approaching, positive = source passed)",
        color=TEXT_COL,
    )
    ax3.set_ylabel("Temperature [°C]", color=TEXT_COL)
    ax3.set_title(
        "Thermal Cycles at Various Transverse Distances from Weld Centreline",
        color=TITLE_COL, fontweight="bold", fontsize=11,
    )
    ax3.set_xlim(-30, 120)
    ax3.set_ylim(0, 2200)
    _legend(ax3, fontsize=8, ncol=3, loc="upper right")

    # ── ax4: transverse profiles at fixed X ──
    x_positions = [-40, -20, -10, -5, 0, 10]
    xcycle_colors = plt.cm.cool(np.linspace(0.15, 0.95, len(x_positions)))
    for xp, col in zip(x_positions, xcycle_colors):
        y_arr_mm, t_arr = thermal_cycle_xslice(xp, cond)
        ax4.plot(y_arr_mm, t_arr, color=col, lw=1.5, label=f"x = {xp} mm")
    for t_ref, col, _ in ref_lines:
        ax4.axhline(t_ref, color=col, lw=0.8, ls="--", alpha=0.7)
    _style_ax(ax4)
    ax4.set_xlabel("Y — Transverse distance [mm]", color=TEXT_COL)
    ax4.set_ylabel("Temperature [°C]", color=TEXT_COL)
    ax4.set_title(
        "Transverse Temperature Profile\nat Fixed X Positions",
        color=TITLE_COL, fontweight="bold", fontsize=9,
    )
    ax4.set_xlim(-60, 60)
    ax4.set_ylim(0, 2200)
    _legend(ax4, fontsize=7.5, ncol=2, loc="upper right")

    fig.suptitle(
        f"Rosenthal Welding Heat Model  |  {_param_label(cond)}",
        color=TITLE_COL, fontsize=11, fontweight="bold", y=0.97,
    )
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  Figure saved → {output_path}")


# ==============================
# PLOT 2 — HAZ WIDTH
# ==============================
def plot_haz(haz_w, haz_w_ac1, output_path, cond=None):
    """
    Generate the HAZ width estimate plot.

    Parameters
    ----------
    haz_w : float or None
        Fusion zone half-width [mm].
    haz_w_ac1 : float or None
        HAZ half-width at AC1 temperature [mm].
    output_path : str
        File path for the saved figure.
    cond : WeldingConditions, optional
        Welding parameters. Uses DEFAULT_CONDITIONS if not provided.
    """
    if cond is None:
        cond = DEFAULT_CONDITIONS

    ref_lines = _ref_lines(cond)
    y_scan_mm = np.linspace(-60, 60, 1000)
    t_scan = rosenthal_temperature(np.zeros_like(y_scan_mm * 1e-3), y_scan_mm * 1e-3, cond)

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=DARK_BG)
    _style_ax(ax)

    ax.plot(y_scan_mm, t_scan, color="#f9d423", lw=2, label="Temperature profile at X=0")
    ax.axvline(0, color="white", lw=0.8, ls=":", alpha=0.5)
    ax.axhspan(
        cond.AC1_temp, cond.AC3_temp, alpha=0.15, color="#ff9500",
        label=f"Intercritical HAZ  ({cond.AC1_temp}–{cond.AC3_temp}°C)",
    )
    ax.axhspan(
        cond.AC3_temp, cond.melting_temp, alpha=0.15, color="#ff4040",
        label=f"Full austenite HAZ  ({cond.AC3_temp}–{cond.melting_temp}°C)",
    )
    ax.axhspan(
        cond.melting_temp, cond.max_electrode_temp, alpha=0.25, color="#ffffff",
        label=f"Fusion zone  (>{cond.melting_temp}°C)",
    )

    for t_ref, col, lbl in ref_lines:
        ax.axhline(t_ref, color=col, lw=1.0, ls="--", alpha=0.8)
        ax.text(-39, t_ref + 20, f"{t_ref}°C  {lbl}", color=col, fontsize=8)

    if haz_w:
        ax.axvline(haz_w, color="#ff4040", lw=1.2, ls=":")
        ax.axvline(-haz_w, color="#ff4040", lw=1.2, ls=":")
        ax.text(haz_w + 0.3, 100, f"Fusion\n+{haz_w:.2f} mm", color="#ff4040", fontsize=8)
        ax.text(-haz_w + 0.3, 100, f"Fusion\n-{haz_w:.2f} mm", color="#ff4040", fontsize=8)
    if haz_w_ac1:
        ax.axvline(haz_w_ac1, color="#ffe066", lw=1.2, ls=":")
        ax.axvline(-haz_w_ac1, color="#ffe066", lw=1.2, ls=":")
        ax.text(haz_w_ac1 + 0.3, 250, f"HAZ\n+{haz_w_ac1:.2f} mm",
                color="#ffe066", fontsize=8)
        ax.text(-haz_w_ac1 - 5, 250, f"HAZ\n-{haz_w_ac1:.2f} mm",
                color="#ffe066", fontsize=8)

    ax.set_xlabel("Y — Transverse distance from centreline [mm]", color=TEXT_COL)
    ax.set_ylabel("Temperature [°C]", color=TEXT_COL)
    ax.set_title(
        f"HAZ Width Estimate  |  {_param_label(cond)}",
        color=TITLE_COL, fontweight="bold", fontsize=11,
    )
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
def plot_tensile(haz_w, haz_w_ac1, output_path, cond=None):
    """
    Generate the tensile strength profile plot.

    Parameters
    ----------
    haz_w : float or None
        Fusion zone half-width [mm].
    haz_w_ac1 : float or None
        HAZ half-width at AC1 temperature [mm].
    output_path : str
        File path for the saved figure.
    cond : WeldingConditions, optional
        Welding parameters. Uses DEFAULT_CONDITIONS if not provided.
    """
    if cond is None:
        cond = DEFAULT_CONDITIONS

    yt_base_mpa = cond.yt / 1e6
    y_ts_mm = np.linspace(-60, 60, 1000)
    yt_w, _ = tensile_strength_profile(y_ts_mm, cond)

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=DARK_BG)
    _style_ax(ax)

    ax.plot(y_ts_mm, yt_w, color="#f9d423", lw=2, label="Tensile strength estimate yt_w")
    ax.axhline(
        yt_base_mpa, color="#ff4040", lw=1.2, ls="--",
        label=f"Base material yt = {yt_base_mpa:.0f} MPa",
    )
    ax.axvline(0, color="white", lw=0.8, ls=":", alpha=0.5)

    if haz_w and haz_w_ac1:
        ax.axvspan(-haz_w, -haz_w_ac1, alpha=0.10, color="#ff4040",
                   label="Full austenite HAZ")
        ax.axvspan(haz_w_ac1, haz_w, alpha=0.10, color="#ff4040")
        ax.axvspan(-haz_w_ac1, 0, alpha=0.08, color="#ff9500",
                   label="Intercritical HAZ")
        ax.axvspan(0, haz_w_ac1, alpha=0.08, color="#ff9500")

    if haz_w:
        ax.axvline(haz_w, color="#ff4040", lw=1.0, ls=":")
        ax.axvline(-haz_w, color="#ff4040", lw=1.0, ls=":")
        ax.text(haz_w + 0.3, yt_base_mpa * 0.72,
                f"Fusion\n+{haz_w:.2f}mm", color="#ff4040", fontsize=7.5)
        ax.text(-haz_w - 5, yt_base_mpa * 0.72,
                f"Fusion\n-{haz_w:.2f}mm", color="#ff4040", fontsize=7.5)
    if haz_w_ac1:
        ax.axvline(haz_w_ac1, color="#ffe066", lw=1.0, ls=":")
        ax.axvline(-haz_w_ac1, color="#ffe066", lw=1.0, ls=":")
        ax.text(haz_w_ac1 + 0.3, yt_base_mpa * 0.60,
                f"HAZ\n+{haz_w_ac1:.2f}mm", color="#ffe066", fontsize=7.5)
        ax.text(-haz_w_ac1 - 5, yt_base_mpa * 0.60,
                f"HAZ\n-{haz_w_ac1:.2f}mm", color="#ffe066", fontsize=7.5)

    ax.set_xlabel("Y — Transverse distance from centreline [mm]", color=TEXT_COL)
    ax.set_ylabel("Tensile Strength [MPa]", color=TEXT_COL)
    ax.set_title(
        f"Tensile Strength Estimate  |  {_param_label(cond)}",
        color=TITLE_COL, fontweight="bold", fontsize=11,
    )
    ax.set_xlim(-40, 40)
    ax.set_ylim(yt_base_mpa - 200, yt_base_mpa + 400)
    _legend(ax, fontsize=8)

    fig.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  Tensile strength figure saved → {output_path}")


# ==============================
# PLOT 4 — HARDNESS
# ==============================
def plot_hardness(haz_w, haz_w_ac1, output_path, cond=None):
    """
    Generate the Vickers hardness profile plot.

    Parameters
    ----------
    haz_w : float or None
        Fusion zone half-width [mm].
    haz_w_ac1 : float or None
        HAZ half-width at AC1 temperature [mm].
    output_path : str
        File path for the saved figure.
    cond : WeldingConditions, optional
        Welding parameters. Uses DEFAULT_CONDITIONS if not provided.

    Returns
    -------
    hv_base : float
        Base material hardness [HV].
    hv_weld : float
        Weld metal hardness [HV].
    """
    if cond is None:
        cond = DEFAULT_CONDITIONS

    yt_base_mpa = cond.yt / 1e6
    y_ts_mm = np.linspace(-60, 60, 1000)
    yt_w, yt_weld = tensile_strength_profile(y_ts_mm, cond)
    hv_profile = hardness_profile(yt_w)
    hv_base = 0.3328 * yt_base_mpa - 14.144
    hv_weld = 0.3328 * yt_weld - 14.144

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=DARK_BG)
    _style_ax(ax)

    ax.plot(y_ts_mm, hv_profile, color="#7bc8f6", lw=2, label="Hardness profile HV")
    ax.axhline(
        hv_base, color="#ff4040", lw=1.2, ls="--",
        label=f"Base material HV = {hv_base:.0f}",
    )
    ax.axhline(
        hv_weld, color="#f9d423", lw=1.2, ls="--",
        label=f"Weld metal HV = {hv_weld:.0f}",
    )
    ax.axvline(0, color="white", lw=0.8, ls=":", alpha=0.5)

    if haz_w and haz_w_ac1:
        ax.axvspan(-haz_w, -haz_w_ac1, alpha=0.10, color="#ff4040",
                   label="Full austenite HAZ")
        ax.axvspan(haz_w_ac1, haz_w, alpha=0.10, color="#ff4040")
        ax.axvspan(-haz_w_ac1, 0, alpha=0.08, color="#ff9500",
                   label="Intercritical HAZ")
        ax.axvspan(0, haz_w_ac1, alpha=0.08, color="#ff9500")

    if haz_w:
        ax.axvline(haz_w, color="#ff4040", lw=1.0, ls=":")
        ax.axvline(-haz_w, color="#ff4040", lw=1.0, ls=":")
        ax.text(haz_w + 0.3, hv_base * 0.72,
                f"Fusion\n+{haz_w:.2f}mm", color="#ff4040", fontsize=7.5)
        ax.text(-haz_w - 5, hv_base * 0.72,
                f"Fusion\n-{haz_w:.2f}mm", color="#ff4040", fontsize=7.5)
    if haz_w_ac1:
        ax.axvline(haz_w_ac1, color="#ffe066", lw=1.0, ls=":")
        ax.axvline(-haz_w_ac1, color="#ffe066", lw=1.0, ls=":")
        ax.text(haz_w_ac1 + 0.3, hv_base * 0.60,
                f"HAZ\n+{haz_w_ac1:.2f}mm", color="#ffe066", fontsize=7.5)
        ax.text(-haz_w_ac1 - 5, hv_base * 0.60,
                f"HAZ\n-{haz_w_ac1:.2f}mm", color="#ffe066", fontsize=7.5)

    ax.set_xlabel("Y — Transverse distance from centreline [mm]", color=TEXT_COL)
    ax.set_ylabel("Hardness [HV]", color=TEXT_COL)
    ax.set_title(
        f"Hardness Profile  |  {_param_label(cond)}",
        color=TITLE_COL, fontweight="bold", fontsize=11,
    )
    ax.set_xlim(-40, 40)
    ax.set_ylim(hv_base - 50, hv_weld + 50)
    _legend(ax, fontsize=8)

    fig.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  Hardness figure saved → {output_path}")

    return hv_base, hv_weld
