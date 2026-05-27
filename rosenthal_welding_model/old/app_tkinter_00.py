"""
================================================================================
  ROSENTHAL WELDING HEAT MODEL — Tkinter Desktop Application
  Run with:  python app_tkinter.py
================================================================================
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
from tkinter import filedialog
import datetime

# ==============================
# COLOUR SCHEME
# ==============================
DARK_BG   = "#0d1117"
PANEL_BG  = "#161b22"
ENTRY_BG  = "#21262d"
TEXT_COL  = "#c9d1d9"
TITLE_COL = "#f0f6fc"
ACCENT    = "#f9d423"
BORDER    = "#30363d"
BTN_BG    = "#238636"
BTN_FG    = "#ffffff"


# ==============================
# ROSENTHAL CORE FUNCTIONS
# ==============================
def rosenthal(X, Y, T0, v, k, Q, alpha):
    r = np.sqrt(X**2 + Y**2) + 1e-9
    return T0 + (Q / (2 * np.pi * k * r)) * np.exp(-(v * (r + X)) / (2 * alpha))

def haz_width(T_thresh, T0, v, k, Q, alpha, n=1000):
    y_scan = np.linspace(0, 60e-3, n)
    T_scan = rosenthal(np.zeros_like(y_scan), y_scan, T0, v, k, Q, alpha)
    below  = np.where(T_scan < T_thresh)[0]
    return y_scan[below[0]] * 1000 if len(below) > 0 else None

def style_ax(ax):
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors=TEXT_COL, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.xaxis.label.set_color(TEXT_COL)
    ax.yaxis.label.set_color(TEXT_COL)
    ax.title.set_color(TITLE_COL)


# ==============================
# MAIN APPLICATION CLASS
# ==============================
class RosenthalApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Rosenthal Welding Heat Model — PhD Research Tool")
        self.configure(bg=DARK_BG)
        self.state("zoomed")   # start maximised

        self._build_ui()
        self.run_model()

    # ── UI LAYOUT ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        # top title bar
        title_bar = tk.Frame(self, bg=PANEL_BG, pady=8)
        title_bar.pack(fill="x", padx=0, pady=0)
        tk.Label(title_bar, text="🔥  Rosenthal Welding Heat Model",
                 bg=PANEL_BG, fg=TITLE_COL,
                 font=("Segoe UI", 14, "bold")).pack(side="left", padx=16)
        tk.Label(title_bar,
                 text="PhD Research Tool  |  Based on Rosenthal (1946)",
                 bg=PANEL_BG, fg=TEXT_COL,
                 font=("Segoe UI", 9)).pack(side="left", padx=8)

        # main area: left panel + right canvas
        main = tk.Frame(self, bg=DARK_BG)
        main.pack(fill="both", expand=True, padx=8, pady=8)

        # ── LEFT PANEL (inputs) ──
        left = tk.Frame(main, bg=PANEL_BG, width=240,
                        highlightbackground=BORDER, highlightthickness=1)
        left.pack(side="left", fill="y", padx=(0,8), pady=0)
        left.pack_propagate(False)

        # scrollable input frame
        canvas_scroll = tk.Canvas(left, bg=PANEL_BG, highlightthickness=0)
        scrollbar     = ttk.Scrollbar(left, orient="vertical",
                                      command=canvas_scroll.yview)
        self.scroll_frame = tk.Frame(canvas_scroll, bg=PANEL_BG)
        self.scroll_frame.bind("<Configure>",
            lambda e: canvas_scroll.configure(
                scrollregion=canvas_scroll.bbox("all")))
        canvas_scroll.create_window((0,0), window=self.scroll_frame, anchor="nw")
        canvas_scroll.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas_scroll.pack(side="left", fill="both", expand=True)

        # bind mousewheel
        canvas_scroll.bind_all("<MouseWheel>",
            lambda e: canvas_scroll.yview_scroll(-1*(e.delta//120), "units"))

        self.inputs = {}
        self._add_section("⚡ Electrical")
        self._add_input("Voltage V (V)",          "V",       20.0)
        self._add_input("Current I (A)",          "I",       180.0)
        self._add_input("Efficiency η",           "te",      0.8)

        self._add_section("🔧 Welding")
        self._add_input("Speed (mm/min)",         "v_mm",    180.0)
        self._add_combo("Cooling type",           "cooling", ["Natural","Quenched"])

        self._add_section("🧪 Thermal Properties")
        self._add_input("Ambient T₀ (°C)",        "T0",      25.0)
        self._add_input("Conductivity k (W/m·K)", "k",       45.0)
        self._add_input("Density ρ (kg/m³)",      "rho",     7850.0)
        self._add_input("Spec. heat c (J/kg·K)",  "c",       470.0)

        self._add_section("🔩 Mechanical")
        self._add_input("Tensile str. yt (MPa)",  "yt",      535.0)
        self._add_input("Yield str. ys (MPa)",    "ys",      385.0)

        self._add_section("🌡 Temperature Limits")
        self._add_input("AC1 (°C)",               "AC1",     720.0)
        self._add_input("AC3 (°C)",               "AC3",     900.0)
        self._add_input("Melting temp (°C)",       "Tm",      1450.0)
        self._add_input("Max elastic temp (°C)",   "Te",      600.0)

        self._add_section("📐 Geometry")
        self._add_input("Plate thickness (mm)",   "thick",   8.1)
        self._add_input("Electrode dia. (mm)",    "ed",      1.2)

        # run button
        btn_frame = tk.Frame(left, bg=PANEL_BG, pady=8)
        btn_frame.pack(fill="x", padx=8, pady=8)
        tk.Button(btn_frame, text="▶  Run Model",
          bg=BTN_BG, fg=BTN_FG,
          font=("Segoe UI", 11, "bold"),
          relief="flat", cursor="hand2",
          command=self.run_model).pack(fill="x")

        tk.Button(btn_frame, text="📄  Save PDF Report",
          bg="#1f6feb", fg=BTN_FG,
          font=("Segoe UI", 11, "bold"),
          relief="flat", cursor="hand2",
          command=self.save_pdf).pack(fill="x", pady=(6,0))

        # ── RIGHT PANEL (metrics + plots) ──
        right = tk.Frame(main, bg=DARK_BG)
        right.pack(side="left", fill="both", expand=True)

        # metrics row
        self.metric_frame = tk.Frame(right, bg=DARK_BG)
        self.metric_frame.pack(fill="x", pady=(0,8))
        self.metric_labels = {}
        metrics = [
            ("m_Q",    "Net heat input",      "W"),
            ("m_HI",   "Heat input",          "kJ/mm"),
            ("m_fz",   "Fusion zone ½-width", "mm"),
            ("m_haz",  "HAZ ½-width",         "mm"),
            ("m_CR",   "Cooling rate @800°C", "°C/s"),
            ("m_Tmax", "Max electrode T",     "°C"),
            ("m_hvhaz","HAZ peak hardness",   "HV"),
            ("m_hvw",  "Weld hardness",       "HV"),
        ]
        for i, (key, label, unit) in enumerate(metrics):
            f = tk.Frame(self.metric_frame, bg=PANEL_BG,
                         highlightbackground=BORDER, highlightthickness=1)
            f.grid(row=0, column=i, padx=4, pady=2, sticky="ew")
            self.metric_frame.columnconfigure(i, weight=1)
            tk.Label(f, text=label, bg=PANEL_BG, fg=TEXT_COL,
                     font=("Segoe UI", 8)).pack(pady=(6,0))
            lbl = tk.Label(f, text="—", bg=PANEL_BG, fg=ACCENT,
                           font=("Segoe UI", 14, "bold"))
            lbl.pack()
            tk.Label(f, text=unit, bg=PANEL_BG, fg=TEXT_COL,
                     font=("Segoe UI", 8)).pack(pady=(0,6))
            self.metric_labels[key] = lbl

        # notebook (tabs)
        style = ttk.Style()
        style.theme_use("default")
        style.configure("TNotebook",        background=DARK_BG, borderwidth=0)
        style.configure("TNotebook.Tab",    background=PANEL_BG,
                        foreground=TEXT_COL, padding=[12,6],
                        font=("Segoe UI", 9))
        style.map("TNotebook.Tab",
                  background=[("selected", ENTRY_BG)],
                  foreground=[("selected", TITLE_COL)])

        self.nb = ttk.Notebook(right)
        self.nb.pack(fill="both", expand=True)

        self.tabs = {}
        for name in ["2D Temperature Field", "HAZ Width",
                     "Tensile Strength", "Hardness Profile", "Thermal Cycles"]:
            frame = tk.Frame(self.nb, bg=DARK_BG)
            self.nb.add(frame, text=name)
            self.tabs[name] = frame

        self.canvases = {}

    # ── INPUT HELPERS ──────────────────────────────────────────────────────────
    def _add_section(self, title):
        tk.Label(self.scroll_frame, text=title,
                 bg=PANEL_BG, fg=ACCENT,
                 font=("Segoe UI", 9, "bold")).pack(
                     anchor="w", padx=10, pady=(12,2))

    def _add_input(self, label, key, default):
        f = tk.Frame(self.scroll_frame, bg=PANEL_BG)
        f.pack(fill="x", padx=10, pady=2)
        tk.Label(f, text=label, bg=PANEL_BG, fg=TEXT_COL,
                 font=("Segoe UI", 8), width=22, anchor="w").pack(side="left")
        var = tk.StringVar(value=str(default))
        tk.Entry(f, textvariable=var, width=8,
                 bg=ENTRY_BG, fg=TEXT_COL,
                 insertbackground=TEXT_COL,
                 relief="flat", font=("Segoe UI", 9)).pack(side="right")
        self.inputs[key] = var

    def _add_combo(self, label, key, options):
        f = tk.Frame(self.scroll_frame, bg=PANEL_BG)
        f.pack(fill="x", padx=10, pady=2)
        tk.Label(f, text=label, bg=PANEL_BG, fg=TEXT_COL,
                 font=("Segoe UI", 8), width=22, anchor="w").pack(side="left")
        var = tk.StringVar(value=options[0])
        ttk.Combobox(f, textvariable=var, values=options,
                     width=9, state="readonly").pack(side="right")
        self.inputs[key] = var

    def _get(self, key):
        try:
            return float(self.inputs[key].get())
        except ValueError:
            messagebox.showerror("Input error", f"Invalid value for {key}")
            return None

    # ── MODEL RUN ──────────────────────────────────────────────────────────────
    def run_model(self):
        # read inputs
        V    = self._get("V");    I   = self._get("I");   te  = self._get("te")
        v_mm = self._get("v_mm"); T0  = self._get("T0");  k   = self._get("k")
        rho  = self._get("rho");  c   = self._get("c");   yt  = self._get("yt")
        AC1  = self._get("AC1");  AC3 = self._get("AC3"); Tm  = self._get("Tm")
        Te   = self._get("Te");   ed  = self._get("ed")

        if None in [V,I,te,v_mm,T0,k,rho,c,yt,AC1,AC3,Tm,Te,ed]:
            return

        # derived
        v     = v_mm / 60 / 1000
        Q     = te * V * I
        alpha = k / (rho * c)
        HI    = Q / (v_mm / 60) / 1000
        r_el  = 2 * ed * 1e-3
        Tmax  = T0 + (Q/(2*np.pi*k*r_el)) * np.exp(-(v*r_el)/(2*alpha))
        CR    = -2*np.pi*k*v*(800-T0)**2/Q

        def R(X, Y): return rosenthal(X, Y, T0, v, k, Q, alpha)

        # HAZ widths
        fz_w  = haz_width(Tm,  T0, v, k, Q, alpha)
        haz_w_ = haz_width(AC1, T0, v, k, Q, alpha)

        # tensile & hardness profile
        y_p   = np.linspace(-60, 60, 1000)
        T_p   = R(np.zeros_like(y_p*1e-3), y_p*1e-3)
        yt_weld = max(0.0799*Tmax + 485.84, yt)
        yt_lin  = 0.0799*T_p + 485.84
        yt_w    = np.maximum(yt_lin, yt)
        yt_w    = np.where(T_p >= Tm, yt_weld, yt_w)
        HV_p    = 0.3328*yt_w - 14.144

        haz_mask   = (T_p >= AC1) & (T_p < Tm)
        HV_haz_max = HV_p[haz_mask].max() if haz_mask.any() else 0.0
        HV_weld    = 0.3328*yt_weld - 14.144

        ref_lines = [
            (Tm,  "#ff4040", "Melting"),
            (AC3, "#ff9500", "AC3"),
            (AC1, "#ffe066", "AC1"),
            (Te,  "#7bc8f6", "Max elastic"),
        ]

        # update metrics
        self.metric_labels["m_Q"].config(   text=f"{Q:.0f}")
        self.metric_labels["m_HI"].config(  text=f"{HI:.2f}")
        self.metric_labels["m_fz"].config(  text=f"{fz_w:.2f}"   if fz_w   else "N/A")
        self.metric_labels["m_haz"].config( text=f"{haz_w_:.2f}" if haz_w_ else "N/A")
        self.metric_labels["m_CR"].config(  text=f"{abs(CR):.0f}")
        self.metric_labels["m_Tmax"].config(text=f"{Tmax:.0f}")
        self.metric_labels["m_hvhaz"].config(text=f"{HV_haz_max:.0f}")
        self.metric_labels["m_hvw"].config( text=f"{HV_weld:.0f}")

        iso_temps  = {"Melting":Tm, "AC3":AC3, "AC1":AC1, "Max elastic":Te}
        iso_colors = ["#ff4040","#ff9500","#ffe066","#7bc8f6"]

        # ── PLOT 1: 2D Temperature field ──
        x_mm = np.linspace(-80, 20, 300)
        y_mm = np.linspace(-60, 60, 300)
        X_mm, Y_mm = np.meshgrid(x_mm, y_mm)
        T_field = R(X_mm*1e-3, Y_mm*1e-3)
        T_field = np.clip(T_field, T0, Tmax)

        cmap_weld = mcolors.LinearSegmentedColormap.from_list(
            "weld", ["#0d1117","#1a2940","#1e4d6b","#e65c00","#f9d423","#ffffff"], N=512)

        fig1, ax = plt.subplots(figsize=(10,4), facecolor=DARK_BG)
        im = ax.contourf(X_mm, Y_mm, np.clip(T_field,T0,2500), levels=256, cmap=cmap_weld)
        cb = fig1.colorbar(im, ax=ax, pad=0.02)
        cb.set_label("Temperature [°C]", color=TEXT_COL)
        cb.ax.yaxis.set_tick_params(color=TEXT_COL)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT_COL)
        iso_labels = []
        for (label, Tiso), col in zip(iso_temps.items(), iso_colors):
            try:
                ax.contour(X_mm, Y_mm, T_field, levels=[Tiso],
                           colors=[col], linewidths=1.2, linestyles="--")
                iso_labels.append(Patch(facecolor=col, label=f"{Tiso:.0f}°C  {label}"))
            except Exception:
                pass
        ax.plot(0, 0, "o", color="white", ms=6, zorder=10)
        ax.legend(handles=iso_labels, loc="lower left", facecolor="#161b22",
                  edgecolor=BORDER, labelcolor=TEXT_COL, fontsize=7, framealpha=0.85)
        style_ax(ax)
        ax.set_xlabel("X — Along welding direction [mm]")
        ax.set_ylabel("Y — Transverse [mm]")
        ax.set_title(f"Rosenthal Temperature Field  |  V={V}V  I={I}A  η={te}  v={v_mm}mm/min",
                     fontweight="bold", fontsize=10)
        fig1.tight_layout()
        self._embed(fig1, "2D Temperature Field")

        # ── PLOT 2: HAZ Width ──
        y_s = np.linspace(-60, 60, 800)
        T_s = R(np.zeros_like(y_s*1e-3), y_s*1e-3)
        fig2, ax = plt.subplots(figsize=(10,4), facecolor=DARK_BG)
        ax.plot(y_s, T_s, color="#f9d423", lw=2, label="Temperature at X=0")
        ax.axvline(0, color="white", lw=0.8, ls=":", alpha=0.5)
        ax.axhspan(AC1, AC3, alpha=0.15, color="#ff9500",
                   label=f"Intercritical HAZ ({AC1:.0f}–{AC3:.0f}°C)")
        ax.axhspan(AC3, Tm,  alpha=0.15, color="#ff4040",
                   label=f"Full austenite HAZ ({AC3:.0f}–{Tm:.0f}°C)")
        ax.axhspan(Tm, Tmax, alpha=0.20, color="#ffffff",
                   label=f"Fusion zone (>{Tm:.0f}°C)")
        for Tref, col, lbl in ref_lines:
            ax.axhline(Tref, color=col, lw=1.0, ls="--", alpha=0.8)
            ax.text(-39, Tref+20, f"{Tref:.0f}°C  {lbl}", color=col, fontsize=7)
        if fz_w:
            ax.axvline( fz_w, color="#ff4040", lw=1.2, ls=":")
            ax.axvline(-fz_w, color="#ff4040", lw=1.2, ls=":")
            ax.text( fz_w+0.3, 100, f"Fusion\n+{fz_w:.2f}mm", color="#ff4040", fontsize=7)
            ax.text(-fz_w+0.3, 100, f"Fusion\n-{fz_w:.2f}mm", color="#ff4040", fontsize=7)
        if haz_w_:
            ax.axvline( haz_w_, color="#ffe066", lw=1.2, ls=":")
            ax.axvline(-haz_w_, color="#ffe066", lw=1.2, ls=":")
            ax.text( haz_w_+0.3, 250, f"HAZ\n+{haz_w_:.2f}mm", color="#ffe066", fontsize=7)
            ax.text(-haz_w_-5,   250, f"HAZ\n-{haz_w_:.2f}mm", color="#ffe066", fontsize=7)
        style_ax(ax)
        ax.set_xlabel("Y — Transverse distance [mm]")
        ax.set_ylabel("Temperature [°C]")
        ax.set_title("HAZ Width Estimate", fontweight="bold", fontsize=10)
        ax.set_xlim(-40, 40); ax.set_ylim(0, 4000)
        ax.legend(facecolor="#161b22", edgecolor=BORDER, labelcolor=TEXT_COL, fontsize=7)
        fig2.tight_layout()
        self._embed(fig2, "HAZ Width")

        # ── PLOT 3: Tensile Strength ──
        fig3, ax = plt.subplots(figsize=(10,4), facecolor=DARK_BG)
        ax.plot(y_p, yt_w, color="#f9d423", lw=2, label="Tensile strength yt_w")
        ax.axhline(yt, color="#ff4040", lw=1.2, ls="--",
                   label=f"Base material yt = {yt:.0f} MPa")
        ax.axvline(0, color="white", lw=0.8, ls=":", alpha=0.5)
        if fz_w and haz_w_:
            ax.axvspan(-fz_w,   -haz_w_, alpha=0.10, color="#ff4040", label="Full austenite HAZ")
            ax.axvspan( haz_w_,  fz_w,   alpha=0.10, color="#ff4040")
            ax.axvspan(-haz_w_,  0,      alpha=0.08, color="#ff9500", label="Intercritical HAZ")
            ax.axvspan( 0,       haz_w_, alpha=0.08, color="#ff9500")
        if fz_w:
            ax.axvline( fz_w, color="#ff4040", lw=1.0, ls=":")
            ax.axvline(-fz_w, color="#ff4040", lw=1.0, ls=":")
        if haz_w_:
            ax.axvline( haz_w_, color="#ffe066", lw=1.0, ls=":")
            ax.axvline(-haz_w_, color="#ffe066", lw=1.0, ls=":")
        style_ax(ax)
        ax.set_xlabel("Y — Transverse distance [mm]")
        ax.set_ylabel("Tensile Strength [MPa]")
        ax.set_title("Tensile Strength Profile across Weld", fontweight="bold", fontsize=10)
        ax.set_xlim(-40, 40); ax.set_ylim(yt-200, yt+400)
        ax.legend(facecolor="#161b22", edgecolor=BORDER, labelcolor=TEXT_COL, fontsize=7)
        fig3.tight_layout()
        self._embed(fig3, "Tensile Strength")

        # ── PLOT 4: Hardness ──
        fig4, ax = plt.subplots(figsize=(10,4), facecolor=DARK_BG)
        ax.plot(y_p, HV_p, color="#7bc8f6", lw=2, label="Hardness profile HV")
        ax.axhline(HV_haz_max, color="#ffe066", lw=1.2, ls="--",
                   label=f"HAZ peak HV = {HV_haz_max:.0f}")
        ax.axhline(HV_weld,    color="#f9d423", lw=1.2, ls="--",
                   label=f"Weld metal HV = {HV_weld:.0f}")
        ax.axvline(0, color="white", lw=0.8, ls=":", alpha=0.5)
        if fz_w and haz_w_:
            ax.axvspan(-fz_w,   -haz_w_, alpha=0.10, color="#ff4040", label="Full austenite HAZ")
            ax.axvspan( haz_w_,  fz_w,   alpha=0.10, color="#ff4040")
            ax.axvspan(-haz_w_,  0,      alpha=0.08, color="#ff9500", label="Intercritical HAZ")
            ax.axvspan( 0,       haz_w_, alpha=0.08, color="#ff9500")
        if fz_w:
            ax.axvline( fz_w, color="#ff4040", lw=1.0, ls=":")
            ax.axvline(-fz_w, color="#ff4040", lw=1.0, ls=":")
        if haz_w_:
            ax.axvline( haz_w_, color="#ffe066", lw=1.0, ls=":")
            ax.axvline(-haz_w_, color="#ffe066", lw=1.0, ls=":")
        style_ax(ax)
        ax.set_xlabel("Y — Transverse distance [mm]")
        ax.set_ylabel("Hardness [HV]")
        ax.set_title("Vickers Hardness Profile across Weld", fontweight="bold", fontsize=10)
        ax.set_xlim(-40, 40); ax.set_ylim(HV_haz_max-50, HV_weld+80)
        ax.legend(facecolor="#161b22", edgecolor=BORDER, labelcolor=TEXT_COL, fontsize=7)
        fig4.tight_layout()
        self._embed(fig4, "Hardness Profile")

        # ── PLOT 5: Thermal Cycles ──
        fig5, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10,4), facecolor=DARK_BG)
        fig5.patch.set_facecolor(DARK_BG)

        y_positions  = [0, 2, 4, 8, 12, 20]
        cycle_colors = plt.cm.plasma(np.linspace(0.15, 0.95, len(y_positions)))
        x_arr = np.linspace(-100, 60, 500) * 1e-3
        t_arr = -x_arr / v
        for yp, col in zip(y_positions, cycle_colors):
            T_arr = R(x_arr, np.full_like(x_arr, yp*1e-3))
            ax_l.plot(t_arr, T_arr, color=col, lw=1.5, label=f"y={yp}mm")
        for Tref, col, lbl in ref_lines:
            ax_l.axhline(Tref, color=col, lw=0.8, ls="--", alpha=0.7)
            ax_l.text(-29, Tref+20, lbl, color=col, fontsize=7)
        style_ax(ax_l)
        ax_l.set_xlabel("Time [s]")
        ax_l.set_ylabel("Temperature [°C]")
        ax_l.set_title("Thermal Cycles (Y direction)", fontweight="bold", fontsize=9)
        ax_l.set_xlim(-30, 120); ax_l.set_ylim(0, 2200)
        ax_l.legend(facecolor="#161b22", edgecolor=BORDER, labelcolor=TEXT_COL,
                    fontsize=7, ncol=2)

        x_cl    = np.linspace(-80, 10, 400) * 1e-3
        T_cl    = R(x_cl, np.zeros_like(x_cl))
        x_cl_mm = x_cl * 1000
        ax_r.plot(x_cl_mm, T_cl, color="#f9d423", lw=1.8)
        for Tref, col, lbl in ref_lines:
            ax_r.axhline(Tref, color=col, lw=0.9, ls="--", alpha=0.75)
            ax_r.text(-78, Tref+30, f"{Tref:.0f}°C", color=col, fontsize=6.5)
        ax_r.axvline(0, color="white", lw=0.8, ls=":")
        style_ax(ax_r)
        ax_r.set_xlabel("X [mm]")
        ax_r.set_ylabel("Temperature [°C]")
        ax_r.set_title("Centreline Temperature Profile", fontweight="bold", fontsize=9)
        ax_r.set_ylim(0, 2600)
        fig5.tight_layout()
        self._embed(fig5, "Thermal Cycles")

        plt.close("all")


    def save_pdf(self):
         """Save a full PDF report with all plots and a summary page."""
         # ask user where to save
         filepath = filedialog.asksaveasfilename(
             defaultextension=".pdf",
             filetypes=[("PDF files", "*.pdf")],
             initialfile=f"rosenthal_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
             title="Save PDF Report"
         )
         if not filepath:
            return

         # read current inputs
         V    = self._get("V");    I   = self._get("I");   te  = self._get("te")
         v_mm = self._get("v_mm"); T0  = self._get("T0");  k   = self._get("k")
         rho  = self._get("rho");  c   = self._get("c");   yt  = self._get("yt")
         AC1  = self._get("AC1");  AC3 = self._get("AC3"); Tm  = self._get("Tm")
         Te   = self._get("Te");   ed  = self._get("ed");  ys  = self._get("ys")
         cooling = self.inputs["cooling"].get()

         v     = v_mm / 60 / 1000
         Q     = te * V * I
         alpha = k / (rho * c)
         HI    = Q / (v_mm / 60) / 1000
         r_el  = 2 * ed * 1e-3
         Tmax  = T0 + (Q/(2*np.pi*k*r_el)) * np.exp(-(v*r_el)/(2*alpha))
         CR    = -2*np.pi*k*v*(800-T0)**2/Q

         def R(X, Y): return rosenthal(X, Y, T0, v, k, Q, alpha)

         fz_w   = haz_width(Tm,  T0, v, k, Q, alpha)
         haz_w_ = haz_width(AC1, T0, v, k, Q, alpha)

         y_p     = np.linspace(-60, 60, 1000)
         T_p     = R(np.zeros_like(y_p*1e-3), y_p*1e-3)
         yt_weld = max(0.0799*Tmax + 485.84, yt)
         yt_w    = np.where(T_p >= Tm, yt_weld, np.maximum(0.0799*T_p+485.84, yt))
         HV_p    = 0.3328*yt_w - 14.144
         haz_mask    = (T_p >= AC1) & (T_p < Tm)
         HV_haz_max  = HV_p[haz_mask].max() if haz_mask.any() else 0.0
         HV_weld     = 0.3328*yt_weld - 14.144

         ref_lines = [
             (Tm,  "#ff4040", "Melting"),
             (AC3, "#ff9500", "AC3"),
             (AC1, "#ffe066", "AC1"),
             (Te,  "#7bc8f6", "Max elastic"),
         ]

         with PdfPages(filepath) as pdf:

             # ── PAGE 1: Summary ──
             fig_s = plt.figure(figsize=(11.69, 8.27), facecolor=DARK_BG)
             fig_s.patch.set_facecolor(DARK_BG)
             ax_s = fig_s.add_axes([0,0,1,1])
             ax_s.set_facecolor(DARK_BG)
             ax_s.axis("off")

             # title
             ax_s.text(0.5, 0.95, "Rosenthal Welding Heat Model — Results Report",
                       ha="center", va="top", color=TITLE_COL,
                       fontsize=18, fontweight="bold", transform=ax_s.transAxes)
             ax_s.text(0.5, 0.90,
                       f"Generated: {datetime.datetime.now().strftime('%d %B %Y  %H:%M')}",
                       ha="center", va="top", color=TEXT_COL,
                       fontsize=10, transform=ax_s.transAxes)

             # divider line
             ax_s.axhline(0.87, color=ACCENT, lw=1.5, xmin=0.05, xmax=0.95)

             # input parameters table
             ax_s.text(0.05, 0.84, "INPUT PARAMETERS",
                       color=ACCENT, fontsize=11, fontweight="bold",
                       transform=ax_s.transAxes)

             params = [
                 ("Voltage",              f"{V} V"),
                 ("Current",              f"{I} A"),
                 ("Thermal efficiency",   f"{te}"),
                 ("Welding speed",        f"{v_mm} mm/min  ({v*1000:.2f} mm/s)"),
                 ("Cooling type",         cooling),
                 ("Ambient temperature",  f"{T0} °C"),
                 ("Thermal conductivity", f"{k} W/m·K"),
                 ("Density",              f"{rho} kg/m3"),
                 ("Specific heat",        f"{c} J/kg·K"),
                 ("Tensile strength yt",  f"{yt} MPa"),
                 ("Yield strength ys",    f"{ys} MPa"),
                 ("AC1 temperature",      f"{AC1} °C"),
                 ("AC3 temperature",      f"{AC3} °C"),
                 ("Melting temperature",  f"{Tm} °C"),
                 ("Max elastic temp",     f"{Te} °C"),
                 ("Electrode diameter",   f"{ed} mm"),
             ]
             col1_x, col2_x = 0.05, 0.30
             col3_x, col4_x = 0.55, 0.80
             y_start = 0.80
             dy      = 0.034
             half    = len(params) // 2

             for i, (label, val) in enumerate(params[:half]):
                 y = y_start - i*dy
                 ax_s.text(col1_x, y, label+":", color=TEXT_COL,
                      fontsize=9, transform=ax_s.transAxes)
                 ax_s.text(col2_x, y, val, color=ACCENT,
                      fontsize=9, fontweight="bold", transform=ax_s.transAxes)
             for i, (label, val) in enumerate(params[half:]):
                 y = y_start - i*dy
                 ax_s.text(col3_x, y, label+":", color=TEXT_COL,
                      fontsize=9, transform=ax_s.transAxes)
                 ax_s.text(col4_x, y, val, color=ACCENT,
                      fontsize=9, fontweight="bold", transform=ax_s.transAxes)

             # divider
             ax_s.axhline(0.82 - half*dy, color=BORDER, lw=0.8, xmin=0.05, xmax=0.95)

             # results table
             y_res = 0.80 - half*dy - 0.06
             ax_s.text(0.05, y_res+0.04, "COMPUTED RESULTS",
                       color=ACCENT, fontsize=11, fontweight="bold",
                       transform=ax_s.transAxes)

             results = [
                 ("Net heat input Q",          f"{Q:.1f} W"),
                 ("Heat input per length",      f"{HI:.2f} kJ/mm"),
                 ("Thermal diffusivity alpha",  f"{alpha:.3e} m2/s"),
                 ("Max electrode temperature",  f"{Tmax:.0f} °C"),
                 ("Cooling rate at 800°C",      f"{abs(CR):.1f} °C/s"),
                 ("Fusion zone half-width",     f"{fz_w:.2f} mm"   if fz_w   else "N/A"),
                 ("HAZ half-width (AC1)",       f"{haz_w_:.2f} mm" if haz_w_ else "N/A"),
                 ("HAZ peak hardness",          f"{HV_haz_max:.0f} HV"),
                 ("Weld metal hardness",        f"{HV_weld:.0f} HV"),
                 ("Weld metal tensile str.",    f"{yt_weld:.1f} MPa"),
             ]
             half_r = len(results) // 2
             for i, (label, val) in enumerate(results[:half_r]):
                 y = y_res - i*dy
                 ax_s.text(col1_x, y, label+":", color=TEXT_COL,
                           fontsize=9, transform=ax_s.transAxes)
                 ax_s.text(col2_x, y, val, color="#6bcb77",
                           fontsize=9, fontweight="bold", transform=ax_s.transAxes)
             for i, (label, val) in enumerate(results[half_r:]):
                 y = y_res - i*dy
                 ax_s.text(col3_x, y, label+":", color=TEXT_COL,
                           fontsize=9, transform=ax_s.transAxes)
                 ax_s.text(col4_x, y, val, color="#6bcb77",
                           fontsize=9, fontweight="bold", transform=ax_s.transAxes)

             # footer
             ax_s.axhline(0.04, color=ACCENT, lw=1.0, xmin=0.05, xmax=0.95)
             ax_s.text(0.5, 0.02,
                       "Rosenthal, D. (1946). The theory of moving sources of heat. "
                       "Trans. ASME, 68, 849-866.  |  PhD Research — University of Stavanger  |  Even Englund",
                       ha="center", color=TEXT_COL, fontsize=7, transform=ax_s.transAxes)

             pdf.savefig(fig_s, facecolor=DARK_BG)
             plt.close(fig_s)

             # ── PAGE 2: 2D Temperature Field ──
             x_mm = np.linspace(-80, 20, 300)
             y_mm = np.linspace(-60, 60, 300)
             X_mm, Y_mm = np.meshgrid(x_mm, y_mm)
             T_field = R(X_mm*1e-3, Y_mm*1e-3)
             T_field = np.clip(T_field, T0, Tmax)
             cmap_weld = mcolors.LinearSegmentedColormap.from_list(
                 "weld", [DARK_BG,"#1a2940","#1e4d6b","#e65c00","#f9d423","#ffffff"], N=512)

             fig2, ax = plt.subplots(figsize=(11.69, 8.27), facecolor=DARK_BG)
             im = ax.contourf(X_mm, Y_mm, np.clip(T_field,T0,2500), levels=256, cmap=cmap_weld)
             cb = fig2.colorbar(im, ax=ax, pad=0.02)
             cb.set_label("Temperature [°C]", color=TEXT_COL)
             cb.ax.yaxis.set_tick_params(color=TEXT_COL)
             plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT_COL)
             iso_temps  = {"Melting":Tm,"AC3":AC3,"AC1":AC1,"Max elastic":Te}
             iso_colors = ["#ff4040","#ff9500","#ffe066","#7bc8f6"]
             iso_labels = []
             for (label, Tiso), col in zip(iso_temps.items(), iso_colors):
                 try:
                     ax.contour(X_mm, Y_mm, T_field, levels=[Tiso],
                                colors=[col], linewidths=1.2, linestyles="--")
                     iso_labels.append(Patch(facecolor=col, label=f"{Tiso:.0f}°C  {label}"))
                 except Exception:
                     pass
             ax.plot(0, 0, "o", color="white", ms=6, zorder=10)
             ax.legend(handles=iso_labels, loc="lower left", facecolor="#161b22",
                       edgecolor=BORDER, labelcolor=TEXT_COL, fontsize=8, framealpha=0.85)
             style_ax(ax)
             ax.set_xlabel("X — Along welding direction [mm]")
             ax.set_ylabel("Y — Transverse [mm]")
             ax.set_title(f"Rosenthal Temperature Field  |  V={V}V  I={I}A  η={te}  v={v_mm}mm/min",
                          fontweight="bold")
             fig2.tight_layout()
             pdf.savefig(fig2, facecolor=DARK_BG)
             plt.close(fig2)

             # ── PAGE 3: HAZ + Tensile + Hardness ──
             fig3, axes = plt.subplots(1, 3, figsize=(11.69, 8.27), facecolor=DARK_BG)
             fig3.patch.set_facecolor(DARK_BG)

             y_s = np.linspace(-60, 60, 800)
             T_s = R(np.zeros_like(y_s*1e-3), y_s*1e-3)

             # HAZ
             ax = axes[0]
             ax.plot(y_s, T_s, color="#f9d423", lw=2)
             ax.axhspan(AC1, AC3, alpha=0.15, color="#ff9500")
             ax.axhspan(AC3, Tm,  alpha=0.15, color="#ff4040")
             ax.axhspan(Tm, Tmax, alpha=0.20, color="#ffffff")
             for Tref, col, lbl in ref_lines:
                 ax.axhline(Tref, color=col, lw=1.0, ls="--", alpha=0.8)
             if fz_w:
                 ax.axvline( fz_w, color="#ff4040", lw=1.2, ls=":")
                 ax.axvline(-fz_w, color="#ff4040", lw=1.2, ls=":")
                 ax.text( fz_w+0.3, 100, f"+{fz_w:.2f}mm", color="#ff4040", fontsize=7)
                 ax.text(-fz_w+0.3, 100, f"-{fz_w:.2f}mm", color="#ff4040", fontsize=7)
             if haz_w_:
                 ax.axvline( haz_w_, color="#ffe066", lw=1.2, ls=":")
                 ax.axvline(-haz_w_, color="#ffe066", lw=1.2, ls=":")
             style_ax(ax)
             ax.set_xlabel("Y [mm]"); ax.set_ylabel("Temperature [°C]")
             ax.set_title("HAZ Width", fontweight="bold", fontsize=10)
             ax.set_xlim(-40, 40); ax.set_ylim(0, 4000)

             # Tensile
             ax = axes[1]
             ax.plot(y_p, yt_w, color="#f9d423", lw=2, label="yt_w")
             ax.axhline(yt, color="#ff4040", lw=1.2, ls="--", label=f"Base yt={yt:.0f}MPa")
             if fz_w and haz_w_:
                 ax.axvspan(-fz_w,-haz_w_,alpha=0.10,color="#ff4040")
                 ax.axvspan(haz_w_,fz_w,  alpha=0.10,color="#ff4040")
                 ax.axvspan(-haz_w_,0,    alpha=0.08,color="#ff9500")
                 ax.axvspan(0,haz_w_,     alpha=0.08,color="#ff9500")
             style_ax(ax)
             ax.set_xlabel("Y [mm]"); ax.set_ylabel("Tensile Strength [MPa]")
             ax.set_title("Tensile Strength", fontweight="bold", fontsize=10)
             ax.set_xlim(-40, 40); ax.set_ylim(yt-200, yt+400)
             ax.legend(facecolor="#161b22", edgecolor=BORDER, labelcolor=TEXT_COL, fontsize=7)

             # Hardness
             ax = axes[2]
             ax.plot(y_p, HV_p, color="#7bc8f6", lw=2, label="HV profile")
             ax.axhline(HV_haz_max, color="#ffe066", lw=1.2, ls="--",
                        label=f"HAZ peak={HV_haz_max:.0f}HV")
             ax.axhline(HV_weld,    color="#f9d423", lw=1.2, ls="--",
                        label=f"Weld={HV_weld:.0f}HV")
             if fz_w and haz_w_:
                 ax.axvspan(-fz_w,-haz_w_,alpha=0.10,color="#ff4040")
                 ax.axvspan(haz_w_,fz_w,  alpha=0.10,color="#ff4040")
                 ax.axvspan(-haz_w_,0,    alpha=0.08,color="#ff9500")
                 ax.axvspan(0,haz_w_,     alpha=0.08,color="#ff9500")
             style_ax(ax)
             ax.set_xlabel("Y [mm]"); ax.set_ylabel("Hardness [HV]")
             ax.set_title("Hardness Profile", fontweight="bold", fontsize=10)
             ax.set_xlim(-40, 40); ax.set_ylim(HV_haz_max-50, HV_weld+80)
             ax.legend(facecolor="#161b22", edgecolor=BORDER, labelcolor=TEXT_COL, fontsize=7)

             fig3.suptitle("HAZ Width  |  Tensile Strength  |  Hardness Profile",
                           color=TITLE_COL, fontweight="bold", fontsize=12)
             fig3.tight_layout()
             pdf.savefig(fig3, facecolor=DARK_BG)
             plt.close(fig3)

             # ── PAGE 4: Thermal Cycles ──
             fig4, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11.69, 8.27), facecolor=DARK_BG)
             fig4.patch.set_facecolor(DARK_BG)
             y_positions  = [0, 2, 4, 8, 12, 20]
             cycle_colors = plt.cm.plasma(np.linspace(0.15, 0.95, len(y_positions)))
             x_arr = np.linspace(-100, 60, 500) * 1e-3
             t_arr = -x_arr / v
             for yp, col in zip(y_positions, cycle_colors):
                 T_arr = R(x_arr, np.full_like(x_arr, yp*1e-3))
                 ax_l.plot(t_arr, T_arr, color=col, lw=1.5, label=f"y={yp}mm")
             for Tref, col, lbl in ref_lines:
                 ax_l.axhline(Tref, color=col, lw=0.8, ls="--", alpha=0.7)
                 ax_l.text(-29, Tref+20, lbl, color=col, fontsize=7)
             style_ax(ax_l)
             ax_l.set_xlabel("Time [s]"); ax_l.set_ylabel("Temperature [°C]")
             ax_l.set_title("Thermal Cycles (Y direction)", fontweight="bold")
             ax_l.set_xlim(-30, 120); ax_l.set_ylim(0, 2200)
             ax_l.legend(facecolor="#161b22", edgecolor=BORDER, labelcolor=TEXT_COL,
                         fontsize=7, ncol=2)

             x_cl = np.linspace(-80, 10, 400) * 1e-3
             T_cl = R(x_cl, np.zeros_like(x_cl))
             ax_r.plot(x_cl*1000, T_cl, color="#f9d423", lw=1.8)
             for Tref, col, lbl in ref_lines:
                 ax_r.axhline(Tref, color=col, lw=0.9, ls="--", alpha=0.75)
                 ax_r.text(-78, Tref+30, f"{Tref:.0f}°C", color=col, fontsize=6.5)
             ax_r.axvline(0, color="white", lw=0.8, ls=":")
             style_ax(ax_r)
             ax_r.set_xlabel("X [mm]"); ax_r.set_ylabel("Temperature [°C]")
             ax_r.set_title("Centreline Temperature Profile", fontweight="bold")
             ax_r.set_ylim(0, 2600)

             fig4.suptitle("Thermal Cycles  |  Centreline Profile",
                      color=TITLE_COL, fontweight="bold", fontsize=12)
             fig4.tight_layout()
             pdf.savefig(fig4, facecolor=DARK_BG)
             plt.close(fig4)

             # PDF metadata
             d = pdf.infodict()
             d["Title"]   = "Rosenthal Welding Heat Model Report"
             d["Author"]  = "Even Englund"
             d["Subject"] = "Welding thermal analysis — PhD Research"

         messagebox.showinfo("PDF Saved",
                             f"Report saved successfully!\n\n{filepath}")


    # ── EMBED FIGURE INTO TAB ──────────────────────────────────────────────────
    def _embed(self, fig, tab_name):
        frame = self.tabs[tab_name]
        for w in frame.winfo_children():
            w.destroy()
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvases[tab_name] = canvas


# ==============================
# ENTRY POINT
# ==============================
if __name__ == "__main__":
    app = RosenthalApp()
    app.mainloop()
