"""Pumping-test interpretation for confined aquifers.

Theis (1935) type-curve fit by least squares, and the Cooper-Jacob (1946)
straight-line method on the late data. Units: metres and days.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.special import exp1

TIME_UNITS = {'seconds': 1 / 86400, 'minutes': 1 / 1440, 'hours': 1 / 24, 'days': 1.0}


def theis_drawdown(t, r, T, S, Q):
    """Theis drawdown (m) at distance r (m) and time t (days).

    T transmissivity (m2/day), S storativity (-), Q pumping rate (m3/day).
    Returns 0 for t <= 0.
    """
    t = np.asarray(t, float)
    s = np.zeros_like(t)
    m = t > 0
    s[m] = Q / (4 * np.pi * T) * exp1(r * r * S / (4 * T * t[m]))
    return s


def fit_cooper_jacob(t, s, Q, r, T=None, S=None, u_max=0.05):
    """Cooper-Jacob straight line through the late data.

    Uses points with u = r^2 S / (4 T t) < u_max, where T and S are a first
    estimate (by default from a straight line through the later half of the data).
    Returns T, S, the slope per log cycle and the time window used.
    """
    t, s = np.asarray(t, float), np.asarray(s, float)
    if T is None or S is None:
        late = t >= np.median(t)
    else:
        late = r * r * S / (4 * T * t) < u_max
    if late.sum() < 3:
        raise ValueError('fewer than 3 points satisfy the Cooper-Jacob condition; the test is too short '
                         'or the observation well too far away')
    b1, b0 = np.polyfit(np.log10(t[late]), s[late], 1)
    if b1 <= 0:
        raise ValueError('drawdown does not increase with time in the late data')
    T_cj = 2.3 * Q / (4 * np.pi * b1)
    t0 = 10 ** (-b0 / b1)
    S_cj = 2.25 * T_cj * t0 / r ** 2
    return dict(T=T_cj, S=S_cj, slope_per_log_cycle=b1, t0=t0,
                n_points=int(late.sum()), t_from=float(t[late].min()))


def fit_theis(t, s, Q, r):
    """Least-squares Theis fit on log10 T and log10 S.

    Returns T, S, their approximate 95 % ranges, the RMSE (m), and the
    Cooper-Jacob check computed with the Theis estimates.
    """
    t, s = np.asarray(t, float), np.asarray(s, float)
    keep = (t > 0) & np.isfinite(s)
    t, s = t[keep], s[keep]
    first = fit_cooper_jacob(t, s, Q, r)
    x0 = [np.log10(first['T']), np.log10(max(first['S'], 1e-7))]

    def resid(p):
        return theis_drawdown(t, r, 10 ** p[0], 10 ** p[1], Q) - s

    fit = least_squares(resid, x0=x0)
    T, S = 10 ** fit.x[0], 10 ** fit.x[1]
    dof = max(t.size - 2, 1)
    rmse = float(np.sqrt(np.sum(fit.fun ** 2) / t.size))
    try:
        cov = np.linalg.inv(fit.jac.T @ fit.jac) * np.sum(fit.fun ** 2) / dof
        sd = np.sqrt(np.diag(cov))
        T_rng = [10 ** (fit.x[0] - 1.96 * sd[0]), 10 ** (fit.x[0] + 1.96 * sd[0])]
        S_rng = [10 ** (fit.x[1] - 1.96 * sd[1]), 10 ** (fit.x[1] + 1.96 * sd[1])]
    except np.linalg.LinAlgError:
        T_rng = S_rng = [float('nan'), float('nan')]
    cj = fit_cooper_jacob(t, s, Q, r, T=T, S=S)
    return dict(T=T, S=S, T_95=T_rng, S_95=S_rng, rmse=rmse, n_points=int(t.size), cooper_jacob=cj)
