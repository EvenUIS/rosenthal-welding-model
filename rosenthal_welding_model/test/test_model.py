"""
================================================================================
  ROSENTHAL WELDING HEAT MODEL — Unit Tests
================================================================================

Run with:
    pytest test/ -v

Or with coverage:
    pytest --cov=rosenthal test/ -v
================================================================================
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rosenthal.parameters import WeldingConditions, DEFAULT_CONDITIONS
from rosenthal.model import (
    rosenthal_temperature,
    thermal_cycle,
    thermal_cycle_xslice,
    cooling_rate_centreline,
    haz_width_estimate,
    tensile_strength_profile,
    hardness_profile,
)

# Convenience aliases from the default condition preset
T0 = DEFAULT_CONDITIONS.T0
k = DEFAULT_CONDITIONS.k
melting_temp = DEFAULT_CONDITIONS.melting_temp
AC1_temp = DEFAULT_CONDITIONS.AC1_temp
yt = DEFAULT_CONDITIONS.yt


# ==============================
# TEST — WELDING CONDITIONS
# ==============================
class TestWeldingConditions:
    """Tests for the WeldingConditions dataclass and derived quantities."""

    def test_derived_quantities_computed(self):
        """Derived quantities should be computed automatically."""
        cond = WeldingConditions()
        assert cond.v > 0
        assert cond.Q_net > 0
        assert cond.alpha > 0

    def test_custom_conditions_independent(self):
        """Two instances with different parameters should be independent."""
        cond_a = WeldingConditions(V=20.0, I=180.0)
        cond_b = WeldingConditions(V=30.0, I=200.0)
        assert cond_b.Q_net > cond_a.Q_net

    def test_heat_input_formula(self):
        """Q_net = te * V * I."""
        cond = WeldingConditions(V=20.0, I=180.0, te=0.8)
        assert np.isclose(cond.Q_net, 0.8 * 20.0 * 180.0)

    def test_thermal_diffusivity_formula(self):
        """alpha = k / (rho * c)."""
        cond = WeldingConditions(k=45.0, rho=7850.0, c=470.0)
        expected = 45.0 / (7850.0 * 470.0)
        assert np.isclose(cond.alpha, expected)


# ==============================
# TEST — ROSENTHAL TEMPERATURE
# ==============================
class TestRosenthalTemperature:
    """Tests for the rosenthal_temperature function."""

    def test_ambient_far_from_source(self):
        """Temperature far from source should be at or above ambient T0.

        At extreme distances the Rosenthal solution decays to T0 to
        floating-point precision, so >= is the correct assertion.
        """
        T = rosenthal_temperature(np.array([-10.0]), np.array([10.0]))
        assert T[0] >= T0, "Temperature should be at or above ambient"

    def test_temperature_decreases_with_distance(self):
        """Temperature should decrease as Y distance from centreline increases."""
        y_vals = np.array([0.001, 0.005, 0.010, 0.020])
        T_vals = rosenthal_temperature(np.zeros_like(y_vals), y_vals)
        assert np.all(np.diff(T_vals) < 0), "Temperature should decrease with distance"

    def test_no_division_by_zero(self):
        """Should not raise at r=0 due to regularisation."""
        T = rosenthal_temperature(np.array([0.0]), np.array([0.0]))
        assert np.isfinite(T[0]), "Temperature at source should be finite"

    def test_returns_array_for_array_input(self):
        """Should return ndarray when given ndarray input."""
        X = np.linspace(-0.05, 0.01, 100)
        Y = np.zeros(100)
        T = rosenthal_temperature(X, Y)
        assert isinstance(T, np.ndarray)
        assert T.shape == (100,)

    def test_temperature_above_ambient(self):
        """All temperatures should be above ambient."""
        X = np.linspace(-0.05, 0.01, 50)
        Y = np.linspace(0, 0.03, 50)
        T = rosenthal_temperature(X, Y)
        assert np.all(T >= T0), "Temperature should never go below ambient"

    def test_custom_conditions_affect_temperature(self):
        """Higher heat input should produce higher temperatures."""
        cond_low = WeldingConditions(V=10.0, I=100.0)
        cond_high = WeldingConditions(V=30.0, I=300.0)
        T_low = rosenthal_temperature(np.array([0.0]), np.array([0.005]), cond_low)
        T_high = rosenthal_temperature(np.array([0.0]), np.array([0.005]), cond_high)
        assert T_high[0] > T_low[0]


# ==============================
# TEST — THERMAL CYCLE
# ==============================
class TestThermalCycle:
    """Tests for the thermal_cycle and thermal_cycle_xslice functions."""

    def test_returns_two_arrays(self):
        """thermal_cycle should return two equal-length arrays."""
        t, T = thermal_cycle(5.0)
        assert isinstance(t, np.ndarray)
        assert isinstance(T, np.ndarray)
        assert len(t) == len(T)

    def test_peak_temperature_at_centreline_higher(self):
        """Centreline (y=0) should have higher peak than y=10mm."""
        _, T0_arr = thermal_cycle(0.0)
        _, T10 = thermal_cycle(10.0)
        assert T0_arr.max() > T10.max()

    def test_xslice_returns_correct_shape(self):
        """thermal_cycle_xslice should return symmetric Y range."""
        y_mm, T = thermal_cycle_xslice(0.0)
        assert len(y_mm) == len(T)
        assert y_mm[0] < 0 and y_mm[-1] > 0


# ==============================
# TEST — COOLING RATE
# ==============================
class TestCoolingRate:
    """Tests for the cooling_rate_centreline function."""

    def test_cooling_rate_is_negative(self):
        """Cooling rate should be negative (temperature decreasing)."""
        cr = cooling_rate_centreline(800)
        assert cr < 0, "Cooling rate should be negative"

    def test_cooling_rate_increases_with_temperature(self):
        """Higher temperature difference → faster cooling."""
        cr_500 = cooling_rate_centreline(500)
        cr_800 = cooling_rate_centreline(800)
        assert abs(cr_800) > abs(cr_500)


# ==============================
# TEST — HAZ WIDTH
# ==============================
class TestHazWidth:
    """Tests for the haz_width_estimate function."""

    def test_haz_width_positive(self):
        """HAZ width should be a positive number for AC1 threshold."""
        w = haz_width_estimate(AC1_temp)
        assert w is not None
        assert w > 0

    def test_fusion_zone_narrower_than_haz(self):
        """Fusion zone should be narrower than HAZ."""
        w_fusion = haz_width_estimate(melting_temp)
        w_haz = haz_width_estimate(AC1_temp)
        assert w_fusion < w_haz

    def test_very_high_threshold_returns_none(self):
        """Threshold above peak temperature in the physical scan should return None.

        1e9 °C far exceeds any physically meaningful temperature in the model.
        The scan starts at Y=1µm (away from the singularity), so the peak
        temperature seen is the electrode temperature (~3200 °C), which is
        far below 1e9 °C.
        """
        result = haz_width_estimate(1e9)
        assert result is None, (
            "haz_width_estimate should return None when T_threshold "
            "exceeds the peak temperature in the scan domain"
        )


# ==============================
# TEST — TENSILE STRENGTH
# ==============================
class TestTensileStrength:
    """Tests for the tensile_strength_profile function."""

    def test_base_material_strength_minimum(self):
        """Away from weld, strength should equal base material yt."""
        y_mm = np.array([50.0, 55.0, 60.0])
        yt_w, _ = tensile_strength_profile(y_mm)
        assert np.allclose(yt_w, yt / 1e6, atol=1.0)

    def test_weld_metal_strength_above_base(self):
        """Weld metal strength should be above base material strength."""
        y_mm = np.linspace(-60, 60, 1000)
        yt_w, yt_weld = tensile_strength_profile(y_mm)
        assert yt_weld >= yt / 1e6

    def test_no_negative_strength(self):
        """Tensile strength should never be negative."""
        y_mm = np.linspace(-60, 60, 1000)
        yt_w, _ = tensile_strength_profile(y_mm)
        assert np.all(yt_w >= 0)


# ==============================
# TEST — HARDNESS
# ==============================
class TestHardness:
    """Tests for the hardness_profile function."""

    def test_hardness_formula(self):
        """HV = 0.3328 * yt_w - 14.144."""
        yt_w = np.array([535.0, 600.0, 740.0])
        hv = hardness_profile(yt_w)
        expected = 0.3328 * yt_w - 14.144
        assert np.allclose(hv, expected)

    def test_weld_harder_than_base(self):
        """Weld metal (higher yt) should be harder than base material."""
        hv_base = hardness_profile(np.array([yt / 1e6]))[0]
        hv_weld = hardness_profile(np.array([740.0]))[0]
        assert hv_weld > hv_base
