"""
================================================================================
  ROSENTHAL WELDING HEAT MODEL — Streamlit Application
  Run with:  streamlit run app.py
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import streamlit as st

from rosenthal.parameters import WeldingConditions
from rosenthal.model import (
    rosenthal_temperature,
    thermal_cycle,
    haz_width_estimate,
    cooling_rate_centreline,
    tensile_strength_profile,
    hardness_profile,
)

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Rosenthal Welding Heat Model",
    page_icon="🔥",
    layout="wide"
)

# ==============================
# DARK PLOT STYLE
# ==============================
DARK_BG = "#0d1117"
TEXT_COL = "#c9d1d9"
TITLE_COL = "#f0f6fc"
SPINE_COL = "#30363d"


def style_ax(ax):
    """Apply dark theme styling to a matplotlib Axes object."""
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors=TEXT_COL)
    for spine in ax.spines.values():
        spine.set_edgecolor(SPINE_COL)
    ax.xaxis.label.set_color(TEXT_COL)
    ax.yaxis.label.set_color(TEXT_COL)
    ax.title.set_color(TITLE_COL)


# ==============================
# SIDEBAR — INPUT PARAMETERS
# ==============================
st.sidebar.title("⚡ Input Parameters")

st.sidebar.subheader("Electrical")
v_input = st.sidebar.number_input("Voltage V (V)", value=20.0, step=1.0)
i_input = st.sidebar.number_input("Current I (A)", value=180.0, step=5.0)
te_input = st.sidebar.slider(
    "Thermal efficiency η", min_value=0.5, max_value=1.0, value=0.8, step=0.05
)

st.sidebar.subheader("Welding")
v_mm_min_input = st.sidebar.number_input(
    "Welding speed (mm/min)", value=180.0, step=10.0
)
cooling_type_input = st.sidebar.selectbox(
    "Cooling type", ["Natural", "Quenched"]
)

st.sidebar.subheader("Thermal Properties")
t0_input = st.sidebar.number_input(
    "Ambient temperature T₀ (°C)", value=25.0, step=5.0
)
k_input = st.sidebar.number_input(
    "Thermal conductivity k (W/m·K)", value=45.0, step=1.0
)
rho_input = st.sidebar.number_input(
    "Density ρ (kg/m³)", value=7850.0, step=50.0
)
c_input = st.sidebar.number_input(
    "Specific heat c (J/kg·K)", value=470.0, step=10.0
)

st.sidebar.subheader("Mechanical Properties")
yt_mpa_input = st.sidebar.number_input(
    "Tensile strength yt (MPa)", value=535.0, step=5.0
)
ys_mpa_input = st.sidebar.number_input(
    "Yield strength ys (MPa)", value=385.0, step=5.0
)

st.sidebar.subheader("Temperature Limits")
ac1_input = st.sidebar.number_input(
    "AC1 — Lower transform (°C)", value=720.0, step=10.0
)
ac3_input = st.sidebar.number_input(
    "AC3 — Upper transform (°C)", value=900.0, step=10.0
)
melting_input = st.sidebar.number_input(
    "Melting temperature (°C)", value=1450.0, step=10.0
)
max_elastic_input = st.sidebar.number_input(
    "Max elastic temp (°C)", value=600.0, step=10.0
)

st.sidebar.subheader("Geometry")
thickness_input = st.sidebar.number_input(
    "Plate thickness (mm)", value=8.1, step=0.1
)
ed_input = st.sidebar.number_input(
    "Electrode diameter (mm)", value=1.2, step=0.1
)

# ==============================
# BUILD CONDITIONS FROM SIDEBAR
# ==============================
cond = WeldingConditions(
    V=v_input,
    I=i_input,
    te=te_input,
    v_mm_min=v_mm_min_input,
    cooling_type=cooling_type_input,
    T0=t0_input,
    k=k_input,
    rho=rho_input,
    c=c_input,
    yt=yt_mpa_input * 1e6,
    ys=ys_mpa_input * 1e6,
    AC1_temp=ac1_input,
    AC3_temp=ac3_input,
    melting_temp=melting_input,
    max_elastic_temp=max_elastic_input,
    thickness=thickness_input,
    ed=ed_input,
)

# ==============================
# COMPUTED OUTPUTS (via library)
# ==============================
cr_800 = cooling_rate_centreline(800, cond)
haz_w = haz_width_estimate(cond.melting_temp, cond)
haz_w_ac1 = haz_width_estimate(cond.AC1_temp, cond)

y_profile = np.linspace(-60, 60, 1000)
yt_w, yt_weld = tensile_strength_profile(y_profile, cond)
hv_profile = hardness_profile(yt_w)

t_profile = rosenthal_temperature(
    np.zeros_like(y_profile * 1e-3), y_profile * 1e-3, cond
)
haz_mask = (t_profile >= cond.AC1_temp) & (t_profile < cond.melting_temp)
hv_haz_max = hv_profile[haz_mask].max() if haz_mask.any() else 0.0
hv_weld = hardness_profile(np.array([yt_weld]))[0]

ref_lines = [
    (cond.melting_temp,    "#ff4040", "Melting"),
    (cond.AC3_temp,        "#ff9500", "AC3"),
    (cond.AC1_temp,        "#ffe066", "AC1"),
    (cond.max_elastic_temp,"#7bc8f6", "Max elastic"),
]

# ==============================
# HEADER
# ==============================
st.title("🔥 Rosenthal Welding Heat Model")
st.caption(
    f"V={v_input}V  |  I={i_input}A  |  η={te_input}  |  "
    f"v={v_mm_min_input}mm/min  |  k={k_input}W/m·K  |  "
    f"Cooling: {cooling_type_input}"
)

# ==============================
# METRIC CARDS
# ==============================
cols = st.columns(8)
cols[0].metric("Net heat input",      f"{cond.Q_net:.0f} W")
cols[1].metric("Heat input",          f"{cond.HI_kJ_mm:.2f} kJ/mm")
cols[2].metric("Fusion zone ½-width", f"{haz_w:.2f} mm"    if haz_w     else "N/A")
cols[3].metric("HAZ ½-width (AC1)",   f"{haz_w_ac1:.2f} mm" if haz_w_ac1 else "N/A")
cols[4].metric("Cooling rate @800°C", f"{abs(cr_800):.0f} °C/s")
cols[5].metric("Max electrode T",     f"{cond.max_electrode_temp:.0f} °C")
cols[6].metric("HAZ peak hardness",   f"{hv_haz_max:.0f} HV")
cols[7].metric("Weld hardness",       f"{hv_weld:.0f} HV")

st.divider()

# ==============================
# PLOT — 2D TEMPERATURE FIELD
# ==============================
st.subheader("2D Temperature Field")

x_mm = np.linspace(-80, 20, 400)
y_mm = np.linspace(-60, 60, 400)
x_grid, y_grid = np.meshgrid(x_mm, y_mm)
t_field = rosenthal_temperature(x_grid * 1e-3, y_grid * 1e-3, cond)
t_field = np.clip(t_field, cond.T0, cond.max_electrode_temp)

iso_temps = {
    "Melting front": cond.melting_temp,
    "AC3":           cond.AC3_temp,
    "AC1":           cond.AC1_temp,
    "Max elastic":   cond.max_elastic_temp,
}
iso_colors = ["#ff4040", "#ff9500", "#ffe066", "#7bc8f6"]

cmap_weld = mcolors.LinearSegmentedColormap.from_list(
    "weld",
    [DARK_BG, "#1a2940", "#1e4d6b", "#e65c00", "#f9d423", "#ffffff"],
    N=512,
)

fig1, ax = plt.subplots(figsize=(14, 5), facecolor=DARK_BG)
img = ax.contourf(
    x_mm, y_mm, np.clip(t_field, cond.T0, 2500), levels=256, cmap=cmap_weld
)
cb = fig1.colorbar(img, ax=ax, pad=0.02)
cb.set_label("Temperature [°C]", color=TEXT_COL)
cb.ax.yaxis.set_tick_params(color=TEXT_COL)
plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT_COL)

iso_labels = []
for (label, t_iso), col in zip(iso_temps.items(), iso_colors):
    try:
        ax.contour(x_mm, y_mm, t_field, levels=[t_iso],
                   colors=[col], linewidths=1.4, linestyles="--")
        iso_labels.append(Patch(facecolor=col, label=f"{t_iso:.0f}°C  {label}"))
    except Exception:  # noqa: BLE001
        pass

ax.plot(0, 0, "o", color="white", ms=7, zorder=10)
ax.annotate(
    "Heat\nsource", (0, 0), (5, 8), color="white", fontsize=8,
    arrowprops=dict(arrowstyle="->", color="white", lw=0.8),
)
ax.legend(
    handles=iso_labels, loc="lower left", facecolor="#161b22",
    edgecolor=SPINE_COL, labelcolor=TEXT_COL, fontsize=8, framealpha=0.85,
)
style_ax(ax)
ax.set_xlabel("X — Along welding direction [mm]")
ax.set_ylabel("Y — Transverse [mm]")
ax.set_title("Rosenthal Temperature Field (moving frame)", fontweight="bold")
st.pyplot(fig1, use_container_width=True)
plt.close()

st.divider()

# ==============================
# TABS — PROFILE PLOTS
# ==============================
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 HAZ Width", "💪 Tensile Strength", "🔨 Hardness Profile", "⏱ Thermal Cycles"
])

y_scan_mm = np.linspace(-60, 60, 1000)
t_scan = rosenthal_temperature(
    np.zeros_like(y_scan_mm * 1e-3), y_scan_mm * 1e-3, cond
)

# ── TAB 1: HAZ WIDTH ──
with tab1:
    fig2, ax = plt.subplots(figsize=(10, 5), facecolor=DARK_BG)
    ax.plot(y_scan_mm, t_scan, color="#f9d423", lw=2, label="Temperature at X=0")
    ax.axvline(0, color="white", lw=0.8, ls=":", alpha=0.5)
    ax.axhspan(
        cond.AC1_temp, cond.AC3_temp, alpha=0.15, color="#ff9500",
        label=f"Intercritical HAZ ({cond.AC1_temp:.0f}–{cond.AC3_temp:.0f}°C)",
    )
    ax.axhspan(
        cond.AC3_temp, cond.melting_temp, alpha=0.15, color="#ff4040",
        label=f"Full austenite HAZ ({cond.AC3_temp:.0f}–{cond.melting_temp:.0f}°C)",
    )
    ax.axhspan(
        cond.melting_temp, cond.max_electrode_temp, alpha=0.20, color="#ffffff",
        label=f"Fusion zone (>{cond.melting_temp:.0f}°C)",
    )
    for t_ref, col, lbl in ref_lines:
        ax.axhline(t_ref, color=col, lw=1.0, ls="--", alpha=0.8)
        ax.text(-39, t_ref + 20, f"{t_ref:.0f}°C  {lbl}", color=col, fontsize=8)
    if haz_w:
        ax.axvline(haz_w, color="#ff4040", lw=1.2, ls=":")
        ax.axvline(-haz_w, color="#ff4040", lw=1.2, ls=":")
        ax.text(haz_w + 0.3, 100, f"Fusion\n+{haz_w:.2f}mm", color="#ff4040", fontsize=8)
        ax.text(-haz_w + 0.3, 100, f"Fusion\n-{haz_w:.2f}mm", color="#ff4040", fontsize=8)
    if haz_w_ac1:
        ax.axvline(haz_w_ac1, color="#ffe066", lw=1.2, ls=":")
        ax.axvline(-haz_w_ac1, color="#ffe066", lw=1.2, ls=":")
        ax.text(haz_w_ac1 + 0.3, 250, f"HAZ\n+{haz_w_ac1:.2f}mm",
                color="#ffe066", fontsize=8)
        ax.text(-haz_w_ac1 - 5, 250, f"HAZ\n-{haz_w_ac1:.2f}mm",
                color="#ffe066", fontsize=8)
    style_ax(ax)
    ax.set_xlabel("Y — Transverse distance from centreline [mm]")
    ax.set_ylabel("Temperature [°C]")
    ax.set_title("HAZ Width Estimate", fontweight="bold")
    ax.set_xlim(-40, 40)
    ax.set_ylim(0, 4000)
    ax.legend(facecolor="#161b22", edgecolor=SPINE_COL,
              labelcolor=TEXT_COL, fontsize=8)
    st.pyplot(fig2, use_container_width=True)
    plt.close()

# ── TAB 2: TENSILE STRENGTH ──
with tab2:
    fig3, ax = plt.subplots(figsize=(10, 5), facecolor=DARK_BG)
    ax.plot(y_profile, yt_w, color="#f9d423", lw=2, label="Tensile strength yt_w")
    ax.axhline(
        yt_mpa_input, color="#ff4040", lw=1.2, ls="--",
        label=f"Base material yt = {yt_mpa_input:.0f} MPa",
    )
    ax.axvline(0, color="white", lw=0.8, ls=":", alpha=0.5)
    if haz_w and haz_w_ac1:
        ax.axvspan(-haz_w, -haz_w_ac1, alpha=0.10, color="#ff4040",
                   label="Full austenite HAZ")
        ax.axvspan(haz_w_ac1, haz_w, alpha=0.10, color="#ff4040")
        ax.axvspan(-haz_w_ac1, 0, alpha=0.08, color="#ff9500",
                   label="Intercritical HAZ")
        ax.axvspan(0, haz_w_ac1, alpha=0.08, color="#ff9500")
    style_ax(ax)
    ax.set_xlabel("Y — Transverse distance from centreline [mm]")
    ax.set_ylabel("Tensile Strength [MPa]")
    ax.set_title("Tensile Strength Profile across Weld", fontweight="bold")
    ax.set_xlim(-40, 40)
    ax.set_ylim(yt_mpa_input - 200, yt_mpa_input + 400)
    ax.legend(facecolor="#161b22", edgecolor=SPINE_COL,
              labelcolor=TEXT_COL, fontsize=8)
    st.pyplot(fig3, use_container_width=True)
    plt.close()

# ── TAB 3: HARDNESS ──
with tab3:
    fig4, ax = plt.subplots(figsize=(10, 5), facecolor=DARK_BG)
    ax.plot(y_profile, hv_profile, color="#7bc8f6", lw=2, label="Hardness profile HV")
    ax.axhline(
        hv_haz_max, color="#ffe066", lw=1.2, ls="--",
        label=f"HAZ peak HV = {hv_haz_max:.0f}",
    )
    ax.axhline(
        hv_weld, color="#f9d423", lw=1.2, ls="--",
        label=f"Weld metal HV = {hv_weld:.0f}",
    )
    ax.axvline(0, color="white", lw=0.8, ls=":", alpha=0.5)
    style_ax(ax)
    ax.set_xlabel("Y — Transverse distance from centreline [mm]")
    ax.set_ylabel("Hardness [HV]")
    ax.set_title("Vickers Hardness Profile across Weld", fontweight="bold")
    ax.set_xlim(-40, 40)
    ax.set_ylim(hv_haz_max - 50, hv_weld + 80)
    ax.legend(facecolor="#161b22", edgecolor=SPINE_COL,
              labelcolor=TEXT_COL, fontsize=8)
    st.pyplot(fig4, use_container_width=True)
    plt.close()

# ── TAB 4: THERMAL CYCLES ──
with tab4:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Thermal cycles — Y direction**")
        fig5, ax = plt.subplots(figsize=(6, 4), facecolor=DARK_BG)
        y_positions = [0, 2, 4, 8, 12, 20]
        cycle_colors = plt.cm.plasma(np.linspace(0.15, 0.95, len(y_positions)))
        for yp, col in zip(y_positions, cycle_colors):
            t_arr, t_cycle = thermal_cycle(yp, cond)
            ax.plot(t_arr, t_cycle, color=col, lw=1.5, label=f"y={yp}mm")
        for t_ref, col, _ in ref_lines:
            ax.axhline(t_ref, color=col, lw=0.8, ls="--", alpha=0.7)
        style_ax(ax)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Temperature [°C]")
        ax.set_xlim(-30, 120)
        ax.set_ylim(0, 2200)
        ax.legend(facecolor="#161b22", edgecolor=SPINE_COL,
                  labelcolor=TEXT_COL, fontsize=7, ncol=2)
        st.pyplot(fig5, use_container_width=True)
        plt.close()

    with col_b:
        st.markdown("**Centreline temperature profile**")
        fig6, ax = plt.subplots(figsize=(6, 4), facecolor=DARK_BG)
        x_cl = np.linspace(-80, 10, 500) * 1e-3
        t_cl = rosenthal_temperature(x_cl, np.zeros_like(x_cl), cond)
        ax.plot(x_cl * 1000, t_cl, color="#f9d423", lw=1.8)
        for t_ref, col, lbl in ref_lines:
            ax.axhline(t_ref, color=col, lw=0.9, ls="--", alpha=0.75)
            ax.text(-78, t_ref + 30, f"{t_ref:.0f}°C", color=col, fontsize=6.5)
        ax.axvline(0, color="white", lw=0.8, ls=":")
        style_ax(ax)
        ax.set_xlabel("X [mm]")
        ax.set_ylabel("Temperature [°C]")
        ax.set_ylim(0, 2600)
        st.pyplot(fig6, use_container_width=True)
        plt.close()

# ==============================
# FOOTER
# ==============================
st.divider()
st.caption(
    "Rosenthal, D. (1946). The theory of moving sources of heat and its application "
    "to metal treatments. *Trans. ASME*, 68, 849–866.  |  "
    "PhD Research — University of Stavanger  |  Even Englund"
)
