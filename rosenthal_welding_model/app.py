"""
================================================================================
  ROSENTHAL WELDING HEAT MODEL — Streamlit Application
  Run with:  streamlit run app.py
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
import streamlit as st

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
DARK_BG   = "#0d1117"
TEXT_COL  = "#c9d1d9"
TITLE_COL = "#f0f6fc"
SPINE_COL = "#30363d"

def style_ax(ax):
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
V   = st.sidebar.number_input("Voltage V (V)",          value=20.0,  step=1.0)
I   = st.sidebar.number_input("Current I (A)",          value=180.0, step=5.0)
te  = st.sidebar.slider("Thermal efficiency η",         min_value=0.5, max_value=1.0, value=0.8, step=0.05)

st.sidebar.subheader("Welding")
v_mm_min     = st.sidebar.number_input("Welding speed (mm/min)", value=180.0, step=10.0)
cooling_type = st.sidebar.selectbox("Cooling type", ["Natural", "Quenched"])

st.sidebar.subheader("Thermal Properties")
T0  = st.sidebar.number_input("Ambient temperature T₀ (°C)", value=25.0,  step=5.0)
k   = st.sidebar.number_input("Thermal conductivity k (W/m·K)", value=45.0, step=1.0)
rho = st.sidebar.number_input("Density ρ (kg/m³)",           value=7850.0, step=50.0)
c   = st.sidebar.number_input("Specific heat c (J/kg·K)",    value=470.0,  step=10.0)

st.sidebar.subheader("Mechanical Properties")
yt_MPa = st.sidebar.number_input("Tensile strength yt (MPa)", value=535.0, step=5.0)
ys_MPa = st.sidebar.number_input("Yield strength ys (MPa)",   value=385.0, step=5.0)

st.sidebar.subheader("Temperature Limits")
AC1_temp     = st.sidebar.number_input("AC1 — Lower transform (°C)", value=720.0, step=10.0)
AC3_temp     = st.sidebar.number_input("AC3 — Upper transform (°C)", value=900.0, step=10.0)
melting_temp = st.sidebar.number_input("Melting temperature (°C)",   value=1450.0, step=10.0)
max_elastic  = st.sidebar.number_input("Max elastic temp (°C)",      value=600.0,  step=10.0)

st.sidebar.subheader("Geometry")
thickness = st.sidebar.number_input("Plate thickness (mm)", value=8.1,  step=0.1)
ed        = st.sidebar.number_input("Electrode diameter (mm)", value=1.2, step=0.1)


# ==============================
# DERIVED QUANTITIES
# ==============================
v      = v_mm_min / 60 / 1000
Q_net  = te * V * I
alpha  = k / (rho * c)
HI_kJ  = Q_net / (v_mm_min / 60) / 1000
r_el   = (2 * ed) * 1e-3
T_elec = T0 + (Q_net / (2 * np.pi * k * r_el)) * np.exp(-(v * r_el) / (2 * alpha))
CR_800 = -2 * np.pi * k * v * (800 - T0)**2 / Q_net


# ==============================
# ROSENTHAL FUNCTION
# ==============================
def rosenthal(X, Y):
    r = np.sqrt(X**2 + Y**2) + 1e-9
    return T0 + (Q_net / (2 * np.pi * k * r)) * np.exp(-(v * (r + X)) / (2 * alpha))


# ==============================
# HAZ WIDTH
# ==============================
def haz_width(T_thresh, n=1000):
    y_scan = np.linspace(0, 60e-3, n)
    T_scan = rosenthal(np.zeros_like(y_scan), y_scan)
    below  = np.where(T_scan < T_thresh)[0]
    return y_scan[below[0]] * 1000 if len(below) > 0 else None

haz_w     = haz_width(melting_temp)
haz_w_ac1 = haz_width(AC1_temp)


# ==============================
# TENSILE & HARDNESS
# ==============================
y_profile  = np.linspace(-60, 60, 1000)
T_profile  = rosenthal(np.zeros_like(y_profile * 1e-3), y_profile * 1e-3)

yt_weld    = max(0.0799 * T_elec + 485.84, yt_MPa)
yt_linear  = 0.0799 * T_profile + 485.84
yt_w       = np.maximum(yt_linear, yt_MPa)
yt_w       = np.where(T_profile >= melting_temp, yt_weld, yt_w)
HV_profile = 0.3328 * yt_w - 14.144

# HAZ peak hardness (max HV in AC1–melting zone)
haz_mask   = (T_profile >= AC1_temp) & (T_profile < melting_temp)
HV_haz_max = HV_profile[haz_mask].max() if haz_mask.any() else 0.0
HV_weld    = 0.3328 * yt_weld - 14.144

ref_lines  = [
    (melting_temp, "#ff4040", "Melting"),
    (AC3_temp,     "#ff9500", "AC3"),
    (AC1_temp,     "#ffe066", "AC1"),
    (max_elastic,  "#7bc8f6", "Max elastic"),
]

# ==============================
# HEADER
# ==============================
st.title("🔥 Rosenthal Welding Heat Model")
st.caption(f"V={V}V  |  I={I}A  |  η={te}  |  v={v_mm_min}mm/min  |  k={k}W/m·K  |  Cooling: {cooling_type}")

# ==============================
# METRIC CARDS
# ==============================
c1,c2,c3,c4,c5,c6,c7,c8 = st.columns(8)
c1.metric("Net heat input",     f"{Q_net:.0f} W")
c2.metric("Heat input",         f"{HI_kJ:.2f} kJ/mm")
c3.metric("Fusion zone ½-width", f"{haz_w:.2f} mm"     if haz_w     else "N/A")
c4.metric("HAZ ½-width (AC1)",  f"{haz_w_ac1:.2f} mm"  if haz_w_ac1 else "N/A")
c5.metric("Cooling rate @800°C", f"{abs(CR_800):.0f} °C/s")
c6.metric("Max electrode T",    f"{T_elec:.0f} °C")
c7.metric("HAZ peak hardness",  f"{HV_haz_max:.0f} HV")
c8.metric("Weld hardness",      f"{HV_weld:.0f} HV")

st.divider()

# ==============================
# PLOT — 2D TEMPERATURE FIELD
# ==============================
st.subheader("2D Temperature Field")

x_mm = np.linspace(-80, 20, 400)
y_mm = np.linspace(-60, 60, 400)
X_mm, Y_mm = np.meshgrid(x_mm, y_mm)
T_field = rosenthal(X_mm * 1e-3, Y_mm * 1e-3)
T_field = np.clip(T_field, T0, T_elec)

iso_temps = {
    "Melting front":    melting_temp,
    "AC3":              AC3_temp,
    "AC1":              AC1_temp,
    "Max elastic":      max_elastic,
}
iso_colors = ["#ff4040", "#ff9500", "#ffe066", "#7bc8f6"]

cmap_weld = mcolors.LinearSegmentedColormap.from_list(
    "weld", [DARK_BG,"#1a2940","#1e4d6b","#e65c00","#f9d423","#ffffff"], N=512
)

fig1, ax = plt.subplots(figsize=(14, 5), facecolor=DARK_BG)
im = ax.contourf(X_mm, Y_mm, np.clip(T_field, T0, 2500), levels=256, cmap=cmap_weld)
cb = fig1.colorbar(im, ax=ax, pad=0.02)
cb.set_label("Temperature [°C]", color=TEXT_COL)
cb.ax.yaxis.set_tick_params(color=TEXT_COL)
plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT_COL)

iso_labels = []
for (label, Tiso), col in zip(iso_temps.items(), iso_colors):
    try:
        ax.contour(X_mm, Y_mm, T_field, levels=[Tiso],
                   colors=[col], linewidths=1.4, linestyles="--")
        iso_labels.append(Patch(facecolor=col, label=f"{Tiso:.0f}°C  {label}"))
    except Exception:
        pass

ax.plot(0, 0, "o", color="white", ms=7, zorder=10)
ax.annotate("Heat\nsource", (0,0), (5,8), color="white", fontsize=8,
            arrowprops=dict(arrowstyle="->", color="white", lw=0.8))
ax.legend(handles=iso_labels, loc="lower left", facecolor="#161b22",
          edgecolor=SPINE_COL, labelcolor=TEXT_COL, fontsize=8, framealpha=0.85)
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
T_scan    = rosenthal(np.zeros_like(y_scan_mm * 1e-3), y_scan_mm * 1e-3)

# ── TAB 1: HAZ WIDTH ──
with tab1:
    fig2, ax = plt.subplots(figsize=(10, 5), facecolor=DARK_BG)
    ax.plot(y_scan_mm, T_scan, color="#f9d423", lw=2, label="Temperature at X=0")
    ax.axvline(0, color="white", lw=0.8, ls=":", alpha=0.5)
    ax.axhspan(AC1_temp,    AC3_temp,    alpha=0.15, color="#ff9500",
               label=f"Intercritical HAZ ({AC1_temp:.0f}–{AC3_temp:.0f}°C)")
    ax.axhspan(AC3_temp,    melting_temp, alpha=0.15, color="#ff4040",
               label=f"Full austenite HAZ ({AC3_temp:.0f}–{melting_temp:.0f}°C)")
    ax.axhspan(melting_temp, T_elec,      alpha=0.20, color="#ffffff",
               label=f"Fusion zone (>{melting_temp:.0f}°C)")
    for Tref, col, lbl in ref_lines:
        ax.axhline(Tref, color=col, lw=1.0, ls="--", alpha=0.8)
        ax.text(-39, Tref+20, f"{Tref:.0f}°C  {lbl}", color=col, fontsize=8)
    if haz_w:
        ax.axvline( haz_w, color="#ff4040", lw=1.2, ls=":")
        ax.axvline(-haz_w, color="#ff4040", lw=1.2, ls=":")
        ax.text( haz_w+0.3, 100, f"Fusion\n+{haz_w:.2f}mm", color="#ff4040", fontsize=8)
        ax.text(-haz_w+0.3, 100, f"Fusion\n-{haz_w:.2f}mm", color="#ff4040", fontsize=8)
    if haz_w_ac1:
        ax.axvline( haz_w_ac1, color="#ffe066", lw=1.2, ls=":")
        ax.axvline(-haz_w_ac1, color="#ffe066", lw=1.2, ls=":")
        ax.text( haz_w_ac1+0.3, 250, f"HAZ\n+{haz_w_ac1:.2f}mm", color="#ffe066", fontsize=8)
        ax.text(-haz_w_ac1-5,   250, f"HAZ\n-{haz_w_ac1:.2f}mm", color="#ffe066", fontsize=8)
    style_ax(ax)
    ax.set_xlabel("Y — Transverse distance from centreline [mm]")
    ax.set_ylabel("Temperature [°C]")
    ax.set_title("HAZ Width Estimate", fontweight="bold")
    ax.set_xlim(-40, 40)
    ax.set_ylim(0, 4000)
    ax.legend(facecolor="#161b22", edgecolor=SPINE_COL, labelcolor=TEXT_COL, fontsize=8)
    st.pyplot(fig2, use_container_width=True)
    plt.close()

# ── TAB 2: TENSILE STRENGTH ──
with tab2:
    fig3, ax = plt.subplots(figsize=(10, 5), facecolor=DARK_BG)
    ax.plot(y_profile, yt_w, color="#f9d423", lw=2, label="Tensile strength yt_w")
    ax.axhline(yt_MPa, color="#ff4040", lw=1.2, ls="--",
               label=f"Base material yt = {yt_MPa:.0f} MPa")
    ax.axvline(0, color="white", lw=0.8, ls=":", alpha=0.5)
    if haz_w and haz_w_ac1:
        ax.axvspan(-haz_w,    -haz_w_ac1, alpha=0.10, color="#ff4040", label="Full austenite HAZ")
        ax.axvspan( haz_w_ac1, haz_w,     alpha=0.10, color="#ff4040")
        ax.axvspan(-haz_w_ac1, 0,         alpha=0.08, color="#ff9500", label="Intercritical HAZ")
        ax.axvspan( 0,         haz_w_ac1, alpha=0.08, color="#ff9500")
    if haz_w:
        ax.axvline( haz_w, color="#ff4040", lw=1.0, ls=":")
        ax.axvline(-haz_w, color="#ff4040", lw=1.0, ls=":")
        ax.text( haz_w+0.3, yt_MPa*0.75, f"Fusion\n+{haz_w:.2f}mm", color="#ff4040", fontsize=7.5)
        ax.text(-haz_w-5,   yt_MPa*0.75, f"Fusion\n-{haz_w:.2f}mm", color="#ff4040", fontsize=7.5)
    if haz_w_ac1:
        ax.axvline( haz_w_ac1, color="#ffe066", lw=1.0, ls=":")
        ax.axvline(-haz_w_ac1, color="#ffe066", lw=1.0, ls=":")
        ax.text( haz_w_ac1+0.3, yt_MPa*0.62, f"HAZ\n+{haz_w_ac1:.2f}mm", color="#ffe066", fontsize=7.5)
        ax.text(-haz_w_ac1-5,   yt_MPa*0.62, f"HAZ\n-{haz_w_ac1:.2f}mm", color="#ffe066", fontsize=7.5)
    style_ax(ax)
    ax.set_xlabel("Y — Transverse distance from centreline [mm]")
    ax.set_ylabel("Tensile Strength [MPa]")
    ax.set_title("Tensile Strength Profile across Weld", fontweight="bold")
    ax.set_xlim(-40, 40)
    ax.set_ylim(yt_MPa - 200, yt_MPa + 400)
    ax.legend(facecolor="#161b22", edgecolor=SPINE_COL, labelcolor=TEXT_COL, fontsize=8)
    st.pyplot(fig3, use_container_width=True)
    plt.close()

# ── TAB 3: HARDNESS ──
with tab3:
    fig4, ax = plt.subplots(figsize=(10, 5), facecolor=DARK_BG)
    ax.plot(y_profile, HV_profile, color="#7bc8f6", lw=2, label="Hardness profile HV")
    ax.axhline(HV_haz_max, color="#ffe066", lw=1.2, ls="--",
               label=f"HAZ peak HV = {HV_haz_max:.0f}")
    ax.axhline(HV_weld,    color="#f9d423", lw=1.2, ls="--",
               label=f"Weld metal HV = {HV_weld:.0f}")
    ax.axvline(0, color="white", lw=0.8, ls=":", alpha=0.5)
    if haz_w and haz_w_ac1:
        ax.axvspan(-haz_w,    -haz_w_ac1, alpha=0.10, color="#ff4040", label="Full austenite HAZ")
        ax.axvspan( haz_w_ac1, haz_w,     alpha=0.10, color="#ff4040")
        ax.axvspan(-haz_w_ac1, 0,         alpha=0.08, color="#ff9500", label="Intercritical HAZ")
        ax.axvspan( 0,         haz_w_ac1, alpha=0.08, color="#ff9500")
    if haz_w:
        ax.axvline( haz_w, color="#ff4040", lw=1.0, ls=":")
        ax.axvline(-haz_w, color="#ff4040", lw=1.0, ls=":")
        ax.text( haz_w+0.3, HV_weld*0.75, f"Fusion\n+{haz_w:.2f}mm", color="#ff4040", fontsize=7.5)
        ax.text(-haz_w-5,   HV_weld*0.75, f"Fusion\n-{haz_w:.2f}mm", color="#ff4040", fontsize=7.5)
    if haz_w_ac1:
        ax.axvline( haz_w_ac1, color="#ffe066", lw=1.0, ls=":")
        ax.axvline(-haz_w_ac1, color="#ffe066", lw=1.0, ls=":")
        ax.text( haz_w_ac1+0.3, HV_weld*0.62, f"HAZ\n+{haz_w_ac1:.2f}mm", color="#ffe066", fontsize=7.5)
        ax.text(-haz_w_ac1-5,   HV_weld*0.62, f"HAZ\n-{haz_w_ac1:.2f}mm", color="#ffe066", fontsize=7.5)
    style_ax(ax)
    ax.set_xlabel("Y — Transverse distance from centreline [mm]")
    ax.set_ylabel("Hardness [HV]")
    ax.set_title("Vickers Hardness Profile across Weld", fontweight="bold")
    ax.set_xlim(-40, 40)
    ax.set_ylim(HV_haz_max - 50, HV_weld + 80)
    ax.legend(facecolor="#161b22", edgecolor=SPINE_COL, labelcolor=TEXT_COL, fontsize=8)
    st.pyplot(fig4, use_container_width=True)
    plt.close()

# ── TAB 4: THERMAL CYCLES ──
with tab4:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Thermal cycles — Y direction**")
        fig5, ax = plt.subplots(figsize=(6, 4), facecolor=DARK_BG)
        y_positions  = [0, 2, 4, 8, 12, 20]
        cycle_colors = plt.cm.plasma(np.linspace(0.15, 0.95, len(y_positions)))
        x_arr = np.linspace(-100, 60, 600) * 1e-3
        t_arr = -x_arr / v
        for yp, col in zip(y_positions, cycle_colors):
            T_arr = rosenthal(x_arr, np.full_like(x_arr, yp*1e-3))
            ax.plot(t_arr, T_arr, color=col, lw=1.5, label=f"y={yp}mm")
        for Tref, col, lbl in ref_lines:
            ax.axhline(Tref, color=col, lw=0.8, ls="--", alpha=0.7)
        style_ax(ax)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Temperature [°C]")
        ax.set_xlim(-30, 120)
        ax.set_ylim(0, 2200)
        ax.legend(facecolor="#161b22", edgecolor=SPINE_COL, labelcolor=TEXT_COL,
                  fontsize=7, ncol=2)
        st.pyplot(fig5, use_container_width=True)
        plt.close()

    with col_b:
        st.markdown("**Centreline temperature profile**")
        fig6, ax = plt.subplots(figsize=(6, 4), facecolor=DARK_BG)
        x_cl    = np.linspace(-80, 10, 500) * 1e-3
        T_cl    = rosenthal(x_cl, np.zeros_like(x_cl))
        x_cl_mm = x_cl * 1000
        ax.plot(x_cl_mm, T_cl, color="#f9d423", lw=1.8)
        for Tref, col, lbl in ref_lines:
            ax.axhline(Tref, color=col, lw=0.9, ls="--", alpha=0.75)
            ax.text(-78, Tref+30, f"{Tref:.0f}°C", color=col, fontsize=6.5)
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
