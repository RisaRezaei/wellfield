"""Protection zones, injection-to-supply connections and well-siting scans."""
import numpy as np

from .flowfield import FlowField, Well


def isochrones(field: FlowField, well_name, days=(50, 365), n=180, r_stop=None):
    """Time-of-travel zones around an extraction well.

    Each zone is the area from which groundwater reaches the well within the
    given number of days (e.g. 50 days for a drinking-water protection zone).
    Built by tracking particles backward from the well for that time.
    Returns {days: [(x, y), ...]} closed polygons.
    """
    w = field.well(well_name)
    if w.Q <= 0:
        raise ValueError(f'{well_name} is not an extraction well')
    out = {}
    for d in days:
        res = field.track(field.ring(w, n), backward=True, t_max=float(d), keep_paths=False,
                          r_stop=r_stop or 1e6, center=w.z)
        poly = [(float(p.real), float(p.imag)) for p in res['end']]
        poly.append(poly[0])
        out[float(d)] = poly
    return out


def polygon_area(poly):
    p = np.asarray(poly)
    x, y = p[:, 0], p[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def connection(field: FlowField, source, sink, n=360, r_stop=None):
    """Does water injected at `source` reach the extraction well `sink`?

    Particles are released evenly around the injection well, so each carries
    the same share of the injected flow. Returns the share of injected water
    captured, the share of the sink's water that comes from the source, and
    the travel times and path lengths of the captured particles.
    """
    s, k = field.well(source), field.well(sink)
    if s.Q >= 0:
        raise ValueError(f'{source} is not an injection well')
    if k.Q <= 0:
        raise ValueError(f'{sink} is not an extraction well')
    span = abs(s.z - k.z)
    res = field.track(field.ring(s, n), r_stop=r_stop or max(5000.0, 4 * span), center=k.z, keep_paths=False)
    cap = res['status'] == 'well:' + k.name
    frac = float(cap.mean())
    return dict(captured_fraction=frac,
                share_of_sink_water=frac * abs(s.Q) / k.Q,
                n_particles=n,
                unresolved=int((res['status'] == 'stalled').sum()),
                times=res['time'][cap].tolist(),
                lengths=res['length'][cap].tolist(),
                first_arrival=float(res['time'][cap].min()) if cap.any() else None,
                median_travel_time=float(np.median(res['time'][cap])) if cap.any() else None)


def siting_scan(field: FlowField, sink, injection_rate, distances, angle=0.0, n=120):
    """Share of the sink's water that would come from a new injection well,
    placed at each distance from the sink.

    angle: direction of the new well relative to regional flow, in degrees
    (0 = down-gradient, 90 = cross-gradient to the left, 180 = up-gradient).
    Returns the curve and the smallest distance beyond which none is captured.
    """
    k = field.well(sink)
    direction = field.flow_dir * np.exp(1j * np.radians(angle))
    shares = []
    for d in distances:
        zi = k.z + d * direction
        new = Well('new injection', zi.real, zi.imag, -abs(injection_rate))
        f2 = field.with_wells(field.wells + [new])
        c = connection(f2, 'new injection', sink, n=n)
        shares.append(c['share_of_sink_water'])
    shares = np.array(shares)
    safe = None
    for i in range(len(distances)):
        if np.all(shares[i:] == 0):
            safe = float(distances[i])
            break
    return dict(distances=[float(d) for d in distances], share=shares.tolist(), safe_distance=safe, angle=angle)


def doublet_critical_distance(Q, q0):
    """Spacing above which an equal-rate injection well placed straight
    down-gradient of an extraction well does not recirculate: d = 2Q / (pi q0)."""
    return 2 * Q / (np.pi * q0)
