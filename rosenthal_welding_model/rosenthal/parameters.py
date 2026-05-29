"""
================================================================================
  ROSENTHAL WELDING HEAT MODEL — Default Parameters
  All physical constants and default welding conditions are defined here.
  Import WeldingConditions and pass it to model functions instead of
  relying on module-level globals.
================================================================================
"""

from dataclasses import dataclass, field


@dataclass
class WeldingConditions:
    """
    Container for all welding process and material parameters.

    All model functions accept a WeldingConditions instance, making it
    possible to run the model with different parameter sets without
    editing this file or reimporting the module.

    Attributes
    ----------
    V : float
        Arc voltage [V].
    I : float
        Welding current [A].
    te : float
        Thermal efficiency [-].
    v_mm_min : float
        Welding speed [mm/min].
    cooling_type : str
        Cooling condition: "Natural" or "Quenched".
    T0 : float
        Ambient temperature [°C].
    k : float
        Thermal conductivity [W/m·K].
    rho : float
        Density [kg/m³].
    c : float
        Specific heat [J/kg·K].
    ys : float
        Yield strength [Pa].
    yt : float
        Tensile strength [Pa].
    E : float
        Young's modulus [Pa].
    be : float
        Poisson's ratio [-].
    thickness : float
        Plate thickness [mm].
    ed : float
        Electrode diameter [mm].
    AC1_temp : float
        Lower austenite transformation temperature [°C].
    AC3_temp : float
        Upper austenite transformation temperature [°C].
    max_elastic_temp : float
        Maximum elastic temperature [°C].
    melting_temp : float
        Melting temperature [°C].
    """

    # Electrical
    V: float = 20.0
    I: float = 180.0
    te: float = 0.8

    # Welding
    v_mm_min: float = 180.0
    cooling_type: str = "Natural"

    # Thermal properties
    T0: float = 25.0
    k: float = 45.0
    rho: float = 7850.0
    c: float = 470.0

    # Mechanical properties
    ys: float = 385e6
    yt: float = 535e6
    E: float = 210e9
    be: float = 0.3

    # Geometry
    thickness: float = 8.1
    ed: float = 1.2

    # Temperature limits
    AC1_temp: float = 720.0
    AC3_temp: float = 900.0
    max_elastic_temp: float = 600.0
    melting_temp: float = 1450.0

    # Derived quantities — computed automatically after init
    v: float = field(init=False)
    Q_net: float = field(init=False)
    alpha: float = field(init=False)
    HI_kJ_mm: float = field(init=False)
    r_electrode: float = field(init=False)
    max_electrode_temp: float = field(init=False)

    def __post_init__(self):
        """Compute derived quantities from primary parameters."""
        import numpy as np
        self.v = self.v_mm_min / 60.0 / 1000.0
        self.Q_net = self.te * self.V * self.I
        self.alpha = self.k / (self.rho * self.c)
        self.HI_kJ_mm = self.Q_net / (self.v_mm_min / 60.0) / 1000.0
        self.r_electrode = (2.0 * self.ed) * 1e-3
        self.max_electrode_temp = (
            self.T0
            + (self.Q_net / (2.0 * np.pi * self.k * self.r_electrode))
            * np.exp(-(self.v * self.r_electrode) / (2.0 * self.alpha))
        )


# Default preset — use this when no custom conditions are needed
DEFAULT_CONDITIONS = WeldingConditions()
