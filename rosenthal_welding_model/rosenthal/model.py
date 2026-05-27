"""
================================================================================
  ROSENTHAL WELDING HEAT MODEL — Core Functions
  Based on the Rosenthal (1946) Moving Heat Source Solution

  Rosenthal Equation (3D):
      T(X,Y) = T0 + (Q / (2π·k·r)) · exp(−v·(r + X) / (2α))

  Reference: Rosenthal, D. (1946). The theory of moving sources of heat
             and its application to metal treatments. Trans. ASME, 68, 849-866.
================================================================================
"""

import numpy as np
from rosenthal.parameters import (
    V, I, te, v_mm_min, T0, k, rho, c,
    yt, ed, AC1_temp, melting_temp
)


# ==============================
# DERIVED QUANTITIES
# ==============================
v      = v_mm_min / 60 / 1000          # Welding speed [m/s]
Q_net  = te * V * I                    # Net heat input [W]
alpha  = k / (rho * c)                 # Thermal diffusivity [m²/s]
HI_kJ_mm = (te * V * I) / (v_mm_min / 60) / 1000  # Heat input [kJ/mm]

r_electrode        = (2 * ed) * 1e-3   # Electrode radius [m]
max_electrode_temp = T0 + (Q_net / (2 * np.pi * k * r_electrode)) * \
                     np.exp(-(v * r_electrode) / (2 * alpha))  # [°C]


# ==============================
# ROSENTHAL TEMPERATURE FUNCTION
# ==============================
def rosenthal_temperature(X, Y, T0=T0, v=v, k=k, alpha=alpha, Q=Q_net):
    """
    Compute the Rosenthal moving heat source temperature field (3D).

    Parameters
    ----------
    X : float or ndarray  — coordinate along welding direction [m]
    Y : float or ndarray  — transverse coordinate [m]
    T0    : float — ambient temperature [°C]
    v     : float — welding speed [m/s]
    k     : float — thermal conductivity [W/m·K]
    alpha : float — thermal diffusivity [m²/s]
    Q     : float — net heat input [W]

    Returns
    -------
    T : float or ndarray — temperature field [°C]
    """
    r = np.sqrt(X**2 + Y**2) + 1e-9   # avoid division by zero
    T = T0 + (Q / (2 * np.pi * k * r)) * \
        np.exp(-(v * (r + X)) / (2 * alpha))
    return T


# ==============================
# THERMAL CYCLE — Y DIRECTION
# ==============================
def thermal_cycle(y_pos_mm, x_range_mm=(-100, 60), n=800):
    """
    Extract temperature-time cycle at transverse distance y_pos_mm from weld.

    Parameters
    ----------
    y_pos_mm   : float — transverse distance from centreline [mm]
    x_range_mm : tuple — (x_start, x_end) range [mm]
    n          : int   — number of points

    Returns
    -------
    t_arr : ndarray — time array [s]
    T_arr : ndarray — temperature array [°C]
    """
    x_arr = np.linspace(x_range_mm[0], x_range_mm[1], n) * 1e-3
    y_arr = np.full_like(x_arr, y_pos_mm * 1e-3)
    T_arr = rosenthal_temperature(x_arr, y_arr)
    t_arr = -x_arr / v
    return t_arr, T_arr


# ==============================
# THERMAL PROFILE — X SLICES
# ==============================
def thermal_cycle_xslice(x_pos_mm, y_range_mm=(-60, 60), n=800):
    """
    Extract temperature vs. transverse position Y at a fixed X position.

    Parameters
    ----------
    x_pos_mm   : float — fixed X position [mm]
    y_range_mm : tuple — (y_start, y_end) range [mm]
    n          : int   — number of points

    Returns
    -------
    y_arr_mm : ndarray — transverse position array [mm]
    T_arr    : ndarray — temperature array [°C]
    """
    y_arr = np.linspace(y_range_mm[0], y_range_mm[1], n) * 1e-3
    x_arr = np.full_like(y_arr, x_pos_mm * 1e-3)
    T_arr = rosenthal_temperature(x_arr, y_arr)
    return y_arr * 1000, T_arr


# ==============================
# COOLING RATE AT CENTRELINE
# ==============================
def cooling_rate_centreline(T_eval=800):
    """
    Analytical cooling rate dT/dt at centreline (Y=0).
    Rosenthal 3D:  dT/dt = −2π·k·v·(T−T0)² / Q

    Parameters
    ----------
    T_eval : float — temperature at which to evaluate cooling rate [°C]

    Returns
    -------
    CR : float — cooling rate [°C/s]  (negative = cooling)
    """
    CR = -2 * np.pi * k * v * (T_eval - T0)**2 / Q_net
    return CR


# ==============================
# HAZ WIDTH ESTIMATE
# ==============================
def haz_width_estimate(T_threshold, n_points=1000):
    """
    Estimate the transverse half-width of a given isotherm at X=0.

    Parameters
    ----------
    T_threshold : float — isotherm temperature [°C]
    n_points    : int   — number of scan points

    Returns
    -------
    width : float or None — half-width [mm], None if not found
    """
    y_scan = np.linspace(0, 60e-3, n_points)
    T_scan = rosenthal_temperature(np.zeros_like(y_scan), y_scan)
    below  = np.where(T_scan < T_threshold)[0]
    if len(below) == 0:
        return None
    return y_scan[below[0]] * 1000


# ==============================
# TENSILE STRENGTH PROFILE
# ==============================
def tensile_strength_profile(y_ts_mm):
    """
    Compute tensile strength profile across the weld in the Y direction.
    Formula: yt_w = MAX(0.0799*T + 485.84, yt)
    Above melting: weld metal strength based on max electrode temperature.

    Parameters
    ----------
    y_ts_mm : ndarray — transverse positions [mm]

    Returns
    -------
    yt_w          : ndarray — tensile strength profile [MPa]
    yt_weld_metal : float   — weld metal tensile strength [MPa]
    """
    y_ts_m = y_ts_mm * 1e-3
    T_ts   = rosenthal_temperature(np.zeros_like(y_ts_m), y_ts_m)

    yt_linear     = 0.0799 * T_ts + 485.84
    yt_w          = np.maximum(yt_linear, yt / 1e6)
    yt_weld_metal = max(0.0799 * max_electrode_temp + 485.84, yt / 1e6)
    yt_w          = np.where(T_ts >= melting_temp, yt_weld_metal, yt_w)

    return yt_w, yt_weld_metal


# ==============================
# HARDNESS PROFILE
# ==============================
def hardness_profile(yt_w):
    """
    Compute Vickers hardness profile from tensile strength.
    Formula: HV = 0.3328 * yt_w - 14.144

    Parameters
    ----------
    yt_w : ndarray — tensile strength profile [MPa]

    Returns
    -------
    HV : ndarray — Vickers hardness profile [HV]
    """
    return 0.3328 * yt_w - 14.144
