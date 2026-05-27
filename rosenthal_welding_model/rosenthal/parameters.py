# ==============================
# INPUT PARAMETERS
# ==============================

# Electrical parameters
V   = 20          # Voltage [V]
I   = 180         # Current [A]
te  = 0.8         # Thermal efficiency [-]

# Welding parameters
v_mm_min      = 180         # Welding speed [mm/min]
cooling_type  = "Natural"   # "Natural" or "Quenched"

# Thermal properties
T0  = 25          # Ambient temperature [°C]
k   = 45          # Thermal conductivity [W/m·K]
rho = 7850        # Density [kg/m³]
c   = 470         # Specific heat [J/kg·K]

# Mechanical properties
ys  = 385e6       # Yield strength [Pa]
yt  = 535e6       # Tensile strength [Pa]
E   = 210e9       # Young's modulus [Pa]
be  = 0.3         # Poisson's ratio [-]

# Thermal expansion coefficients
alpha_below = 1.20e-5   # Below melting [1/K]
alpha_above = 1.85e-5   # Above melting [1/K]

# Geometry
thickness = 8.1   # Plate thickness [mm]
ed        = 1.2   # Electrode diameter [mm]

# Temperature limits
AC1_temp           = 720    # Lower transformation temperature (A1) [°C]
AC3_temp           = 900    # Upper transformation temperature (A3) [°C]
max_elastic_temp   = 600    # Max elastic temperature [°C]
melting_temp       = 1450   # Melting temperature [°C]
