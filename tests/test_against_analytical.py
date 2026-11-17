"""Checks against textbook solutions. Run: pytest -q"""
import numpy as np
import pytest

from wellfield import (FlowField, Well, fit_plane, fit_theis, fit_cooper_jacob, theis_drawdown,
                       isochrones, polygon_area, connection, doublet_critical_distance,
                       ogata_banks, crank_nicolson_1d)

T, S, B, N = 600.0, 4e-4, 20.0, 0.25


def test_theis_well_function_table():
    # W(u) table values: W(1e-2) = 4.0379, W(1e-4) = 8.6332
    Q, r = 4 * np.pi * T, 1.0          # makes s = W(u)
    for u, W in [(1e-2, 4.0379), (1e-4, 8.6332)]:
        t = r * r * S / (4 * T * u)
        assert theis_drawdown([t], r, T, S, Q)[0] == pytest.approx(W, abs=1e-4)


def test_theis_fit_recovers_parameters():
    t = np.logspace(-3, 0, 30)
    s = theis_drawdown(t, 40.0, T, S, 1500.0)
    f = fit_theis(t, s, 1500.0, 40.0)
    assert f['T'] == pytest.approx(T, rel=1e-4)
    assert f['S'] == pytest.approx(S, rel=1e-4)
    assert f['cooper_jacob']['T'] == pytest.approx(T, rel=0.03)


def test_cooper_jacob_needs_late_data():
    t = np.logspace(-6, -5, 10)                         # far too short
    s = theis_drawdown(t, 400.0, T, S, 1500.0)
    with pytest.raises(ValueError):
        fit_cooper_jacob(t, s, 1500.0, 400.0, T=T, S=S)


def test_plane_fit():
    f = fit_plane([200, -200, 200], [200, 200, -200], [20.3, 20.1, 19.5])
    assert (f['A'], f['B'], f['C']) == pytest.approx((0.0005, 0.002, 19.8))
    assert f['flow_azimuth'] == pytest.approx(194.04, abs=0.01)


def test_radial_isochrone_area():
    # no regional flow: the t-day zone is a circle of area Q t / (n b)
    f = FlowField(T, B, N, wells=[Well('W', 0, 0, 1500.0)])
    zone = isochrones(f, 'W', days=[50])[50.0]
    assert polygon_area(zone) == pytest.approx(1500 * 50 / (N * B), rel=0.01)


def test_stagnation_point_and_capture_width():
    q0 = 1.25
    f = FlowField(T, B, N, qx=q0, wells=[Well('W', 0, 0, 1500.0)])
    stag = f.stagnation_points()
    assert len(stag) == 1
    assert stag[0][0] == pytest.approx(1500 / (2 * np.pi * q0), rel=1e-3)
    # far up-gradient the catchment is Q / q0 wide
    res = f.track(f.ring(f.well('W'), 180), backward=True, r_stop=40000, center=0j, ds_max=50, keep_paths=False)
    far = res['end'][res['status'] == 'out']
    width = np.ptp(far.imag[far.real < -30000])
    assert width == pytest.approx(1500 / q0, rel=0.02)


def test_rankine_oval_all_brine_captured():
    q0 = 1.25
    f = FlowField(T, B, N, qx=q0, wells=[Well('supply', 0, 0, 1500.0), Well('brine', -200, 0, -1500.0)])
    assert connection(f, 'brine', 'supply', n=120)['captured_fraction'] == 1.0


@pytest.mark.parametrize('factor,captured', [(0.85, True), (1.05, False)])
def test_doublet_recirculation_threshold(factor, captured):
    q0 = 1.25
    d = factor * doublet_critical_distance(1500.0, q0)
    f = FlowField(T, B, N, qx=q0, wells=[Well('supply', 0, 0, 1500.0), Well('brine', d, 0, -1500.0)])
    frac = connection(f, 'brine', 'supply', n=240)['captured_fraction']
    assert (frac > 0) == captured


def test_ogata_banks_matches_crank_nicolson():
    L, v, alpha = 200.0, 1.0, 20.0
    t, c = crank_nicolson_1d(L, v, alpha * v, t_end=3 * L / v, dt=0.05)
    assert np.max(np.abs(c - ogata_banks(t / (L / v), L / alpha))) < 0.005


def test_ogata_banks_large_peclet_is_finite():
    c = ogata_banks(np.array([0.5, 1.0, 2.0]), 5000.0)
    assert np.all(np.isfinite(c)) and c[1] == pytest.approx(0.5, abs=0.01)
