"""Command line: python -m wellfield config.toml

Reads the CSV files named in the config, runs every analysis the config asks
for, and writes results.json, report.md, figures and zones.geojson.
"""
import argparse
import csv
import json
import os
import sys
import tomllib

import numpy as np

from .pumping_test import fit_theis, theis_drawdown, TIME_UNITS
from .regional_flow import fit_plane, compass
from .flowfield import FlowField, Well
from .zones import isochrones, polygon_area, connection, siting_scan, doublet_critical_distance
from .transport import streamtube_breakthrough

C_BLUE, C_ORANGE, C_AQUA, C_GREY = '#2a78d6', '#eb6834', '#1baf7a', '#9aa3b2'


def read_csv(path, required):
    with open(path, newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f'{path}: no rows')
    cols = {c.strip().lower(): c for c in rows[0].keys()}
    missing = [c for c in required if c not in cols]
    if missing:
        raise SystemExit(f'{path}: missing column(s) {missing}; found {list(cols)}')
    return [{k: r[cols[k]].strip() for k in cols} for r in rows]


def run(cfg_path):
    base = os.path.dirname(os.path.abspath(cfg_path))
    with open(cfg_path, 'rb') as f:
        cfg = tomllib.load(f)
    p = lambda name: os.path.join(base, name)
    out_dir = p(cfg.get('output', 'out'))
    os.makedirs(out_dir, exist_ok=True)
    R = {'title': cfg.get('title', '')}
    report = [f"# {cfg.get('title', 'wellfield report')}", '']

    aq = cfg['aquifer']
    b, n = float(aq['thickness']), float(aq['porosity'])

    # 1 pumping test -----------------------------------------------------------
    T = aq.get('transmissivity')
    pt = cfg.get('pumping_test')
    if pt:
        rows = read_csv(p(pt['file']), ['time', 'drawdown'])
        k = TIME_UNITS[pt.get('time_unit', 'minutes')]
        t = np.array([float(r['time']) for r in rows]) * k
        s = np.array([float(r['drawdown']) for r in rows])
        fit = fit_theis(t, s, float(pt['rate']), float(pt['distance']))
        R['pumping_test'] = fit
        if T is None:
            T = fit['T']
        report += ['## Pumping test',
                   f"- Theis: T = {fit['T']:.0f} m²/day (95 %: {fit['T_95'][0]:.0f}–{fit['T_95'][1]:.0f}), "
                   f"S = {fit['S']:.2e}, RMSE = {fit['rmse'] * 1000:.0f} mm",
                   f"- Cooper–Jacob check: T = {fit['cooper_jacob']['T']:.0f} m²/day, S = {fit['cooper_jacob']['S']:.2e}",
                   f"- Hydraulic conductivity K = T / b = {fit['T'] / b:.1f} m/day", '']
        plot_pumping_test(t / k, s, fit, float(pt['rate']), float(pt['distance']), k, pt.get('time_unit', 'minutes'),
                          os.path.join(out_dir, 'pumping_test.png'))
    if T is None:
        raise SystemExit('give aquifer.transmissivity or a [pumping_test] section')
    T = float(T)
    R['aquifer'] = dict(T=T, b=b, n=n, K=T / b)

    # 2 regional flow ----------------------------------------------------------
    rf = cfg.get('regional_flow', {})
    if 'file' in rf:
        rows = read_csv(p(rf['file']), ['name', 'x', 'y', 'head'])
        plane = fit_plane([float(r['x']) for r in rows], [float(r['y']) for r in rows], [float(r['head']) for r in rows])
        A, B = plane['A'], plane['B']
    else:
        i, az = float(rf.get('gradient', 0.0)), float(rf.get('flow_azimuth', 0.0))
        A, B = -i * np.sin(np.radians(az)), -i * np.cos(np.radians(az))
        plane = dict(A=A, B=B, C=None, gradient=i, flow_azimuth=az, rmse=0.0)
    q0 = T * plane['gradient']
    v0 = T / b * plane['gradient'] / n
    plane.update(q0=q0, seepage_velocity=v0, direction=compass(plane['flow_azimuth']))
    R['regional_flow'] = plane
    report += ['## Regional flow',
               f"- Gradient {plane['gradient']:.4f} toward {plane['flow_azimuth']:.0f}° ({plane['direction']})",
               f"- Seepage velocity {v0:.3f} m/day ({v0 * 365:.0f} m/year)", '']

    # wells and flow field -----------------------------------------------------
    rows = read_csv(p(cfg['wells']['file']), ['name', 'x', 'y', 'rate'])
    wells = [Well(r['name'], float(r['x']), float(r['y']), float(r['rate'])) for r in rows]
    field = FlowField.from_gradient(T, b, n, A, B, wells)
    stag = field.stagnation_points()
    R['wells'] = [dict(name=w.name, x=w.x, y=w.y, rate=w.Q) for w in wells]
    R['stagnation_points'] = stag

    # 3 protection zones -------------------------------------------------------
    zones = {}
    pz = cfg.get('protection_zones')
    if pz:
        report.append('## Protection zones (time of travel)')
        for name in pz['wells']:
            iso = isochrones(field, name, days=pz.get('days', [50, 365]))
            zones[name] = iso
            for d, poly in iso.items():
                report.append(f"- {name}, {d:g} days: {polygon_area(poly) / 1e4:.1f} ha, reaches "
                              f"{max(np.hypot(x - field.well(name).x, y - field.well(name).y) for x, y in poly):.0f} m from the well")
        report.append('')
    R['protection_zones'] = {k: {f'{d:g}': dict(area_ha=polygon_area(pl) / 1e4) for d, pl in v.items()} for k, v in zones.items()}

    # 4 connections + breakthrough --------------------------------------------
    R['connections'] = []
    curves = []
    for c in cfg.get('connection', []):
        con = connection(field, c['source'], c['sink'])
        entry = {k: con[k] for k in ['captured_fraction', 'share_of_sink_water', 'first_arrival',
                                     'median_travel_time', 'unresolved']}
        entry.update(source=c['source'], sink=c['sink'])
        line = (f"- {c['source']} → {c['sink']}: {con['captured_fraction'] * 100:.0f} % of the injected water reaches "
                f"the well ({con['share_of_sink_water'] * 100:.0f} % of what it pumps)")
        if con['times']:
            alpha = float(c.get('dispersivity', 0.1 * np.median(con['lengths'])))
            t_end = max(5 * np.median(con['times']), 365.0)
            tt = np.linspace(0, t_end, 1500)
            cr = streamtube_breakthrough(con['times'], con['lengths'], alpha, tt) * con['share_of_sink_water']
            c0, cb = c.get('source_concentration'), c.get('background', 0.0)
            entry.update(dispersivity=alpha)
            line += f"; first water after {con['first_arrival']:.0f} days (advection)"
            if c0 is not None:
                conc = cb + (c0 - cb) * cr
                lim = c.get('limit')
                if lim is not None:
                    hit = np.argmax(conc >= lim) if (conc >= lim).any() else None
                    entry['days_to_limit'] = float(tt[hit]) if hit is not None else None
                    unit = c.get('unit', '')
                    line += (f"; above {lim:g} {unit} after {tt[hit]:.0f} days" if hit is not None
                             else f"; stays below {lim:g} {unit}")
                curves.append((c, tt, conc, lim))
            else:
                curves.append((c, tt, cr, None))
        R['connections'].append(entry)
        report.append(line)
    if cfg.get('connection'):
        report.insert(len(report) - len(cfg['connection']), '## Injection → extraction')
        report.append('')
    if curves:
        plot_breakthrough(curves, os.path.join(out_dir, 'breakthrough.png'))

    # 5 siting scan ------------------------------------------------------------
    st = cfg.get('siting')
    if st:
        dists = np.arange(st.get('step', 50), st.get('max_distance', 1500) + 1e-9, st.get('step', 50))
        scans = [siting_scan(field.with_wells([w for w in wells if w.name == st['sink']]), st['sink'],
                             st['injection_rate'], dists, angle=a) for a in st.get('directions', [0, 90, 180])]
        R['siting'] = scans
        Qs = field.well(st['sink']).Q
        report.append('## Where can a new injection well go?')
        names = {0: 'down-gradient', 90: 'cross-gradient', 180: 'up-gradient'}
        for sc in scans:
            nm = names.get(int(sc['angle']), f"{sc['angle']:g}° from flow")
            report.append(f"- {nm}: " + (f"safe from {sc['safe_distance']:.0f} m" if sc['safe_distance']
                                         else 'not safe within the scanned distances'))
        if abs(Qs - st['injection_rate']) < 1e-9:
            report.append(f"- equal-rate doublet check: 2Q/(π q₀) = {doublet_critical_distance(Qs, q0):.0f} m")
        report.append('')
        plot_siting(scans, os.path.join(out_dir, 'siting.png'))

    plot_map(field, zones, stag, os.path.join(out_dir, 'flow_map.png'), title=cfg.get('title', ''))
    write_geojson(field, zones, os.path.join(out_dir, 'zones.geojson'), cfg.get('crs'))
    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(R, f, indent=1, default=float)
    with open(os.path.join(out_dir, 'report.md'), 'w') as f:
        f.write('\n'.join(report) + '\n')
    print('\n'.join(report))
    print(f'outputs in {out_dir}')
    return R


# -- figures ------------------------------------------------------------------
def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.color': '#e6e8ec', 'grid.linewidth': 0.8})
    return plt


def plot_pumping_test(t_raw, s, fit, Q, r, k, unit, path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.4, 4))
    tt = np.logspace(np.log10(t_raw.min()), np.log10(t_raw.max()), 200)
    ax.plot(t_raw, s, 'o', ms=5, color=C_BLUE, label='Measured')
    ax.plot(tt, theis_drawdown(tt * k, r, fit['T'], fit['S'], Q), color=C_ORANGE, lw=2,
            label=f"Theis: T = {fit['T']:.0f} m²/d, S = {fit['S']:.1e}")
    cj = fit['cooper_jacob']
    line = cj['slope_per_log_cycle'] * np.log10(tt * k / cj['t0'])
    ax.plot(tt[line > 0], line[line > 0], '--', color=C_AQUA, lw=1.8, label=f"Cooper–Jacob: T = {cj['T']:.0f} m²/d")
    ax.set_xscale('log')
    ax.set_xlabel(f'Time ({unit})')
    ax.set_ylabel('Drawdown (m)')
    ax.set_title(f'Pumping test, observation well at {r:g} m', loc='left', fontsize=11)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_map(field, zones, stag, path, title=''):
    plt = _plt()
    xs, ys = [w.x for w in field.wells], [w.y for w in field.wells]
    big = max(abs(w.Q) for w in field.wells)
    spread = max(max(xs) - min(xs), max(ys) - min(ys))
    pad = float(np.clip(max(3 * big / (2 * np.pi * max(field.q0, 1e-6)), 0.6 * spread), 400, 3000))
    ext = (min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad)
    fig, ax = plt.subplots(figsize=(7, 7 * (ext[3] - ext[2]) / (ext[1] - ext[0])))
    ax.grid(False)
    ctr = complex((ext[0] + ext[1]) / 2, (ext[2] + ext[3]) / 2)
    rr = 1.6 * max(ext[1] - ext[0], ext[3] - ext[2])

    def draw(res, color, lw, alpha):
        for path_ in res['paths']:
            q = np.array(path_)
            ax.plot(q.real, q.imag, color=color, lw=lw, alpha=alpha, solid_capstyle='round')

    # regional flow
    u = field.flow_dir
    perp = u * 1j
    starts = [ctr - u * rr / 2 + perp * s for s in np.linspace(-rr / 2, rr / 2, 25)]
    draw(field.track(np.array(starts), r_stop=rr, center=ctr, ds_max=15), C_GREY, 0.7, 0.6)
    for w in field.wells:
        if w.Q > 0:
            draw(field.track(field.ring(w, 36), backward=True, r_stop=rr, center=ctr, ds_max=8), C_BLUE, 0.9, 0.85)
        elif w.Q < 0:
            draw(field.track(field.ring(w, 36), r_stop=rr, center=ctr, ds_max=8), C_ORANGE, 0.9, 0.9)
    for name, iso in zones.items():
        for d, poly in sorted(iso.items(), reverse=True):
            q = np.array(poly)
            ax.fill(q[:, 0], q[:, 1], color=C_AQUA, alpha=0.18 if d == min(iso) else 0.08, lw=0)
            ax.plot(q[:, 0], q[:, 1], color=C_AQUA, lw=1.2)
            right = q[np.argmax(q[:, 0])]
            ax.annotate(f'{d:g} d', right, xytext=(4, 0), textcoords='offset points', va='center', fontsize=8,
                        color='#137a55', fontweight='bold')
    for w in field.wells:
        ax.plot(w.x, w.y, 'o', ms=9, mfc=C_BLUE if w.Q > 0 else C_ORANGE, mec='white', mew=1.5, zorder=5)
        ax.annotate(w.name, (w.x, w.y), xytext=(0, 9), textcoords='offset points', ha='center', fontsize=9,
                    fontweight='bold', zorder=6)
    for x, y in stag:
        ax.plot(x, y, 'D', ms=6, color='#333', zorder=5)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=C_GREY, label='Regional flow'), Line2D([], [], color=C_BLUE, label='Water reaching extraction wells'),
         Line2D([], [], color=C_ORANGE, label='Water from injection wells'),
         Line2D([], [], color=C_AQUA, label='Time-of-travel zones'), Line2D([], [], marker='D', ls='', color='#333', label='Stagnation point')]
    ax.legend(handles=h, loc='lower left', fontsize=8, framealpha=0.9)
    ax.set_title(title, loc='left', fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_breakthrough(curves, path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.4, 4))
    colors = [C_ORANGE, C_BLUE, C_AQUA]
    for i, (c, tt, conc, lim) in enumerate(curves):
        ax.plot(tt, conc, color=colors[i % 3], lw=2, label=f"{c['source']} → {c['sink']}")
        if lim is not None:
            ax.axhline(lim, color='#555', ls='--', lw=1)
            ax.annotate(f"limit {lim:g} {c.get('unit', '')}", (tt[-1], lim), xytext=(-4, 4), textcoords='offset points', ha='right', fontsize=8)
    ax.set_xlabel('Days since injection started')
    unit = curves[0][0].get('unit', '')
    ax.set_ylabel(f'Concentration in pumped water ({unit})' if unit else 'Concentration in pumped water')
    ax.set_title('Breakthrough at the extraction well', loc='left', fontsize=11)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_siting(scans, path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.4, 4))
    names = {0: 'Down-gradient', 90: 'Cross-gradient', 180: 'Up-gradient'}
    colors = [C_BLUE, C_AQUA, C_ORANGE]
    for i, sc in enumerate(scans):
        ax.plot(sc['distances'], np.array(sc['share']) * 100, color=colors[i % 3], lw=2,
                label=names.get(int(sc['angle']), f"{sc['angle']:g}°"))
    ax.set_xlabel('Distance from the extraction well (m)')
    ax.set_ylabel('Share of pumped water from the new well (%)')
    ax.set_ylim(-3, 105)
    ax.set_title('Where can a new injection well go?', loc='left', fontsize=11)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_geojson(field, zones, path, crs=None):
    feats = []
    for w in field.wells:
        feats.append({'type': 'Feature', 'properties': {'kind': 'well', 'name': w.name, 'rate_m3_day': w.Q},
                      'geometry': {'type': 'Point', 'coordinates': [w.x, w.y]}})
    for name, iso in zones.items():
        for d, poly in iso.items():
            feats.append({'type': 'Feature',
                          'properties': {'kind': 'time_of_travel_zone', 'well': name, 'days': d,
                                         'area_ha': round(polygon_area(poly) / 1e4, 2)},
                          'geometry': {'type': 'Polygon', 'coordinates': [[[round(x, 2), round(y, 2)] for x, y in poly]]}})
    gj = {'type': 'FeatureCollection', 'features': feats}
    if crs:
        gj['crs'] = {'type': 'name', 'properties': {'name': crs}}
    with open(path, 'w') as f:
        json.dump(gj, f)


def main(argv=None):
    ap = argparse.ArgumentParser(prog='wellfield', description=__doc__.split('\n')[0])
    ap.add_argument('config', help='TOML file describing the site')
    a = ap.parse_args(argv)
    run(a.config)


if __name__ == '__main__':
    main()
