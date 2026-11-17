"""Regional groundwater flow from observation-well heads.

Fits the plane h = A x + B y + C (exactly through 3 wells, by least squares
through more) and returns the hydraulic gradient and the flow direction.
"""
import numpy as np


def fit_plane(x, y, h):
    """Return A, B, C, gradient, flow azimuth and residuals.

    x, y in metres (any projected system), h in metres.
    Flow azimuth is the direction groundwater flows to, in degrees
    clockwise from north (grid north of the coordinate system).
    """
    x, y, h = (np.asarray(v, float) for v in (x, y, h))
    if x.size < 3:
        raise ValueError('at least 3 observation wells are needed')
    M = np.column_stack([x, y, np.ones_like(x)])
    if np.linalg.matrix_rank(M) < 3:
        raise ValueError('observation wells are on one line; the gradient is undefined')
    (A, B, C), *_ = np.linalg.lstsq(M, h, rcond=None)
    res = h - M @ np.array([A, B, C])
    grad = float(np.hypot(A, B))
    azimuth = float((np.degrees(np.arctan2(-A, -B)) + 360) % 360)
    return dict(A=float(A), B=float(B), C=float(C), gradient=grad, flow_azimuth=azimuth,
                rmse=float(np.sqrt(np.mean(res ** 2))) if x.size > 3 else 0.0,
                residuals=res.tolist())


def compass(azimuth):
    names = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    return names[int((azimuth % 360) / 22.5 + 0.5) % 16]
