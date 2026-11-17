"""Advection-dispersion along flow paths.

Ogata-Banks (1961) solution for a constant-concentration inlet, applied
streamtube by streamtube; a Crank-Nicolson finite-difference solver checks it.
"""
import numpy as np
from scipy.special import erfc, erfcx
from scipy.sparse import diags
from scipy.sparse.linalg import splu


def ogata_banks(tR, Pe):
    """Relative concentration C/C0 at the end of a 1D tube.

    tR = t / tau (time over advective travel time), Pe = L / alpha_L.
    exp(Pe) * erfc(b) is evaluated as exp(Pe - b^2) * erfcx(b), so large
    Peclet numbers do not overflow.
    """
    tR = np.asarray(tR, float)
    out = np.zeros_like(tR)
    m = tR > 0
    a = (1 - tR[m]) * np.sqrt(Pe / (4 * tR[m]))
    b = (1 + tR[m]) * np.sqrt(Pe / (4 * tR[m]))
    out[m] = 0.5 * (erfc(a) + np.exp(Pe - b * b) * erfcx(b))
    return out


def streamtube_breakthrough(times, lengths, alpha_L, t):
    """Mean C/C0 over streamtubes that carry equal flow.

    times, lengths: advective travel time (days) and length (m) of each tube.
    """
    t = np.asarray(t, float)
    c = np.zeros_like(t)
    for tau, L in zip(times, lengths):
        c += ogata_banks(t / tau, L / alpha_L)
    return c / max(len(times), 1)


def crank_nicolson_1d(L, v, D, t_end, nx=800, dt=None, domain=3.0, save_every=10):
    """Numerical check: 1D advection-dispersion, C(0,t) = 1, zero gradient at the far end.

    Central differences in space, Crank-Nicolson in time. Returns times and C/C0 at x = L.
    """
    x = np.linspace(0, domain * L, nx + 1)
    dx = x[1] - x[0]
    dt = dt or min(0.5 * dx / v, t_end / 2000)
    nsteps = int(np.ceil(t_end / dt))
    ld, la = D * dt / dx ** 2, v * dt / (2 * dx)
    main = np.full(nx + 1, 1 + ld)
    low = np.full(nx, -0.5 * ld - 0.5 * la)
    up = np.full(nx, -0.5 * ld + 0.5 * la)
    A = diags([low, main, up], [-1, 0, 1], format='lil')
    B = diags([-low, 2 - main, -up], [-1, 0, 1], format='lil')
    for M in (A, B):
        M[0, :] = 0
        M[-1, :] = 0
    A[0, 0] = 1
    A[-1, -1], A[-1, -2] = 1, -1
    lu, Bc = splu(A.tocsc()), B.tocsr()
    c = np.zeros(nx + 1)
    c[0] = 1.0
    iL = int(round(L / dx))
    ts, cs = [], []
    for k in range(1, nsteps + 1):
        rhs = Bc @ c
        rhs[0], rhs[-1] = 1.0, 0.0
        c = lu.solve(rhs)
        if k % save_every == 0:
            ts.append(k * dt)
            cs.append(c[iL])
    return np.array(ts), np.array(cs)
