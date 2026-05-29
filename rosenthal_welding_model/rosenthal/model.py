"""
================================================================================
  ROSENTHAL WELDING HEAT MODEL — Core Functions
  Based on the Rosenthal (1946) Moving Heat Source Solution

  Rosenthal Equation (3D):
      T(X,Y) = T0 + (Q / (2π·k·r)) · exp(−v·(r + X) / (2α))

  Reference: Rosenthal, D. (1946). The theory of moving sources of heat
             and its application to metal treatments. Trans. ASME, 68, 849-866.

  All public functions accept a WeldingConditions instance so the model
  can be called with arbitrary parameter sets without reimporting.
================================================================================
"""

import numpy as np
from rosenthal.parameters import WeldingConditions, DEFAULT_CONDITIONS


def rosenthal_temperature(X, Y, cond=None):
    """
    Compute the Rosenthal moving heat source temperature field (3D).

    Parameters
    ----------
    X : float or ndarray
        Coordinate along welding direction [m].
    Y : float or ndarray
        Transverse coordinate [m].
    cond : WeldingConditions, optional
        Welding parameters. Uses DEFAULT_CONDITIONS if not provided.

    Returns
    -------
    T : float or ndarray
        Temperature field [°C].
    """
    if cond is None:
        cond = DEFAULT_CONDITIONS
    r = np.sqrt(X**2 + Y**2) + 1e-9
    return (
        cond.T0
        + (cond.Q_net / (2.0 * np.pi * cond.k * r))
        * np.exp(-(cond.v * (r + X)) / (2.0 * cond.alpha))
    )


def thermal_cycle(y_pos_mm, cond=None, x_range_mm=(-100, 60), n=800):
    """
    Extract temperature-time cycle at transverse distance y_pos_mm from weld.

    Parameters
    ----------
    y_pos_mm : float
        Transverse distance from centreline [mm].
    cond : WeldingConditions, optional
        Welding parameters. Uses DEFAULT_CONDITIONS if not provided.
    x_range_mm : tuple
        (x_start, x_end) range [mm].
    n : int
        Number of points.

    Returns
    -------
    t_arr : ndarray
        Time array [s].
    T_arr : ndarray
        Temperature array [°C].
    """
    if cond is None:
        cond = DEFAULT_CONDITIONS
    x_arr = np.linspace(x_range_mm[0], x_range_mm[1], n) * 1e-3
    y_arr = np.full_like(x_arr, y_pos_mm * 1e-3)
    T_arr = rosenthal_temperature(x_arr, y_arr, cond)
    t_arr = -x_arr / cond.v
    return t_arr, T_arr


def thermal_cycle_xslice(x_pos_mm, cond=None, y_range_mm=(-60, 60), n=800):
    """
    Extract temperature vs. transverse position Y at a fixed X position.

    Parameters
    ----------
    x_pos_mm : float
        Fixed X position [mm].
    cond : WeldingConditions, optional
        Welding parameters. Uses DEFAULT_CONDITIONS if not provided.
    y_range_mm : tuple
        (y_start, y_end) range [mm].
    n : int
        Number of points.

    Returns
    -------
    y_arr_mm : ndarray
        Transverse position array [mm].
    T_arr : ndarray
        Temperature array [°C].
    """
    if cond is None:
        cond = DEFAULT_CONDITIONS
    y_arr = np.linspace(y_range_mm[0], y_range_mm[1], n) * 1e-3
    x_arr = np.full_like(y_arr, x_pos_mm * 1e-3)
    T_arr = rosenthal_temperature(x_arr, y_arr, cond)
    return y_arr * 1000.0, T_arr


def cooling_rate_centreline(T_eval=800, cond=None):
    """
    Analytical cooling rate dT/dt at centreline (Y=0).

    Uses the Rosenthal 3D approximation:
        dT/dt = -2π·k·v·(T - T0)² / Q

    Parameters
    ----------
    T_eval : float
        Temperature at which to evaluate cooling rate [°C].
    cond : WeldingConditions, optional
        Welding parameters. Uses DEFAULT_CONDITIONS if not provided.

    Returns
    -------
    CR : float
        Cooling rate [°C/s]. Negative value indicates cooling.
    """
    if cond is None:
        cond = DEFAULT_CONDITIONS
    return (
        -2.0 * np.pi * cond.k * cond.v * (T_eval - cond.T0) ** 2 / cond.Q_net
    )


def haz_width_estimate(T_threshold, cond=None, n_points=2000):
    """
    Estimate the transverse half-width of a given isotherm at X=0.

    Scans from the weld centreline outward and returns the first Y
    position where temperature drops below T_threshold. Returns None
    if the temperature is below T_threshold everywhere in the scan
    range (i.e. the isotherm does not exist).

    Parameters
    ----------
    T_threshold : float
        Isotherm temperature [°C].
    cond : WeldingConditions, optional
        Welding parameters. Uses DEFAULT_CONDITIONS if not provided.
    n_points : int
        Number of scan points along Y.

    Returns
    -------
    width : float or None
        Half-width [mm], or None if the isotherm is not present.
    """
    if cond is None:
        cond = DEFAULT_CONDITIONS

    # Scan from a small offset to avoid the numerical singularity at Y=0,
    # then check whether the temperature at the first real point already
    # exceeds the threshold.  If not, the isotherm does not exist.
    y_scan = np.linspace(1e-6, 60e-3, n_points)
    T_scan = rosenthal_temperature(np.zeros_like(y_scan), y_scan, cond)

    # If the highest temperature in the scan is below the threshold,
    # the isotherm does not exist in the physical domain.
    if T_scan[0] < T_threshold:
        return None

    below = np.where(T_scan < T_threshold)[0]
    if len(below) == 0:
        return None
    return y_scan[below[0]] * 1000.0


def tensile_strength_profile(y_ts_mm, cond=None):
    """
    Compute tensile strength profile across the weld in the Y direction.

    Formula: yt_w = MAX(0.0799·T + 485.84, yt_base)
    Above melting: weld metal strength based on max electrode temperature.

    Parameters
    ----------
    y_ts_mm : ndarray
        Transverse positions [mm].
    cond : WeldingConditions, optional
        Welding parameters. Uses DEFAULT_CONDITIONS if not provided.

    Returns
    -------
    yt_w : ndarray
        Tensile strength profile [MPa].
    yt_weld_metal : float
        Weld metal tensile strength [MPa].
    """
    if cond is None:
        cond = DEFAULT_CONDITIONS
    yt_base_mpa = cond.yt / 1e6
    y_ts_m = y_ts_mm * 1e-3
    T_ts = rosenthal_temperature(np.zeros_like(y_ts_m), y_ts_m, cond)
    yt_linear = 0.0799 * T_ts + 485.84
    yt_w = np.maximum(yt_linear, yt_base_mpa)
    yt_weld_metal = max(
        0.0799 * cond.max_electrode_temp + 485.84, yt_base_mpa
    )
    yt_w = np.where(T_ts >= cond.melting_temp, yt_weld_metal, yt_w)
    return yt_w, yt_weld_metal


def hardness_profile(yt_w):
    """
    Compute Vickers hardness profile from tensile strength.

    Formula: HV = 0.3328 · yt_w - 14.144

    Parameters
    ----------
    yt_w : ndarray
        Tensile strength profile [MPa].

    Returns
    -------
    HV : ndarray
        Vickers hardness profile [HV].
    """
    return 0.3328 * yt_w - 14.144
