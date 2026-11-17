"""Steady 2D flow of wells in a confined aquifer with uniform regional flow.

Analytic element superposition: uniform flow plus one point sink/source per
well, in an aquifer without boundaries. Particle tracking uses 4th-order
Runge-Kutta with steps shortened near wells.

Sign convention: well rate Q > 0 extracts (pumping), Q < 0 injects.
Units: metres and days. Coordinates: any projected system in metres.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import root


@dataclass
class Well:
    name: str
    x: float
    y: float
    Q: float          # m3/day, + extraction, - injection

    @property
    def z(self):
        return complex(self.x, self.y)


class FlowField:
    def __init__(self, T, b, n, qx=0.0, qy=0.0, wells=()):
        """T transmissivity (m2/day), b aquifer thickness (m), n effective porosity,
        qx, qy regional discharge per unit width (m2/day) = -T * dh/dx, -T * dh/dy."""
        self.T, self.b, self.n = float(T), float(b), float(n)
        self.qx, self.qy = float(qx), float(qy)
        self.wells = list(wells)

    @classmethod
    def from_gradient(cls, T, b, n, A, B, wells=()):
        """Regional flow from the head plane h = A x + B y + C."""
        return cls(T, b, n, qx=-T * A, qy=-T * B, wells=wells)

    @property
    def q0(self):
        return float(np.hypot(self.qx, self.qy))

    @property
    def flow_dir(self):
        """Unit vector of regional flow as a complex number."""
        return complex(self.qx, self.qy) / self.q0 if self.q0 > 0 else 1 + 0j

    def well(self, name):
        for w in self.wells:
            if w.name == name:
                return w
        raise KeyError(f'no well named {name!r}')

    def with_wells(self, wells):
        return FlowField(self.T, self.b, self.n, self.qx, self.qy, wells)

    # -- field ---------------------------------------------------------------
    def discharge(self, z):
        """Complex discharge Qx - i Qy (m2/day) at complex points z."""
        z = np.asarray(z, complex)
        w = np.full(z.shape, self.qx - 1j * self.qy, dtype=complex)
        with np.errstate(divide='ignore', invalid='ignore'):
            for wl in self.wells:
                w = w - wl.Q / (2 * np.pi) / (z - wl.z)
        return w

    def velocity(self, z):
        """Seepage velocity vx + i vy (m/day)."""
        return np.conj(self.discharge(z)) / (self.n * self.b)

    def stagnation_points(self, extent=None, grid=120):
        """Points where the groundwater stands still, found from a grid of starts."""
        if not self.wells:
            return []
        xs = [w.x for w in self.wells]
        ys = [w.y for w in self.wells]
        if extent is None:
            pad = 3 * max(abs(w.Q) for w in self.wells) / (2 * np.pi * max(self.q0, 1e-9))
            pad = min(max(pad, 200.0), 20000.0)
            extent = (min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad)
        gx, gy = np.meshgrid(np.linspace(extent[0], extent[1], grid), np.linspace(extent[2], extent[3], grid))
        mag = np.abs(self.discharge(gx + 1j * gy))
        found = []
        for i in range(1, grid - 1):
            for j in range(1, grid - 1):
                v = mag[i, j]
                if not np.isfinite(v) or v > np.nanmin(mag[i - 1:i + 2, j - 1:j + 2]):
                    continue
                sol = root(lambda p: [self.discharge(p[0] + 1j * p[1]).real,
                                      self.discharge(p[0] + 1j * p[1]).imag], [gx[i, j], gy[i, j]])
                if not sol.success:
                    continue
                zc = complex(*sol.x)
                if min(abs(zc - w.z) for w in self.wells) < 1.0:
                    continue
                if abs(self.discharge(zc)) > 1e-6 * max(self.q0, 1e-3):
                    continue
                if all(abs(zc - f) > 1.0 for f in found):
                    found.append(zc)
        return [(round(f.real, 1), round(f.imag, 1)) for f in found]

    # -- particle tracking ---------------------------------------------------
    def track(self, z0, backward=False, t_max=np.inf, ds_max=5.0, r_stop=5000.0,
              center=None, capture_radius=1.0, max_steps=200000, keep_paths=True):
        """Track particles from complex starts z0.

        Forward particles end in extraction wells, backward ones in injection
        wells. Stops at t_max (days), beyond r_stop (m) from `center` (default:
        mean well position), or at a well.
        Returns dict with paths, end status ('well:<name>', 'time', 'out',
        'stalled'), travel time (days) and path length (m) per particle.
        """
        z = np.atleast_1d(np.asarray(z0, complex)).copy()
        n = z.size
        sign = -1.0 if backward else 1.0
        if center is None:
            center = np.mean([w.z for w in self.wells]) if self.wells else 0j
        sinks = [w for w in self.wells if (w.Q > 0) != backward and w.Q != 0]
        zw_all = np.array([w.z for w in self.wells]) if self.wells else np.array([np.inf + 0j])
        t = np.zeros(n)
        L = np.zeros(n)
        status = np.array([''] * n, dtype=object)
        alive = np.ones(n, bool)
        paths = [[p] for p in z] if keep_paths else None

        def vel(zz):
            return sign * self.velocity(zz)

        for _ in range(max_steps):
            if not alive.any():
                break
            idx = np.where(alive)[0]
            zi = z[idx]
            dmin = np.min(np.abs(zi[:, None] - zw_all[None, :]), axis=1)
            step = np.clip(0.15 * dmin, 0.02, ds_max)
            k1 = vel(zi)
            sp = np.abs(k1)
            stalled = sp < 1e-12
            h = np.where(stalled, 0.0, step / np.where(stalled, 1, sp))
            h = np.minimum(h, t_max - t[idx])
            k2 = vel(zi + 0.5 * h * k1)
            k3 = vel(zi + 0.5 * h * k2)
            k4 = vel(zi + h * k3)
            znew = zi + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            bad = ~np.isfinite(znew)
            znew = np.where(bad, zi, znew)
            z[idx] = znew
            t[idx] += h
            L[idx] += np.abs(znew - zi)
            if keep_paths:
                for j, p in zip(idx, znew):
                    paths[j].append(p)
            done = np.zeros(idx.size, bool)
            for w in sinks:
                hit = (np.abs(znew - w.z) < capture_radius) & ~done
                status[idx[hit]] = 'well:' + w.name
                done |= hit
            out = (np.abs(znew - center) > r_stop) & ~done
            status[idx[out]] = 'out'
            done |= out
            tm = (t[idx] >= t_max - 1e-9) & ~done
            status[idx[tm]] = 'time'
            done |= tm
            st = (stalled | bad) & ~done
            status[idx[st]] = 'stalled'
            done |= st
            alive[idx[done]] = False
        status[alive] = 'stalled'
        return dict(paths=paths, status=status, time=t, length=L, end=z)

    def ring(self, well, n=72, r=0.5):
        a = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / n
        return well.z + r * np.exp(1j * a)
