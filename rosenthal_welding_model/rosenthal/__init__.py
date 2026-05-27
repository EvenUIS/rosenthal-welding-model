"""
Rosenthal Welding Heat Model
============================
A PhD research tool for thermal analysis of welding processes.
Based on the Rosenthal (1946) moving heat source analytical solution.

Modules
-------
parameters : Input parameters (material, electrical, geometry)
model      : Core Rosenthal functions (temperature, HAZ, strength, hardness)
plots      : Plotting functions for all output figures

Reference
---------
Rosenthal, D. (1946). The theory of moving sources of heat and its
application to metal treatments. Trans. ASME, 68, 849-866.
"""

from rosenthal.model import (
    rosenthal_temperature,
    thermal_cycle,
    thermal_cycle_xslice,
    cooling_rate_centreline,
    haz_width_estimate,
    tensile_strength_profile,
    hardness_profile,
)

__version__ = "1.0.0"
__author__  = "Even Englund"
