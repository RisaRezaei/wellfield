"""wellfield: pumping tests, protection zones and injection-well screening
for confined aquifers, in plain NumPy/SciPy."""
from .pumping_test import theis_drawdown, fit_theis, fit_cooper_jacob
from .regional_flow import fit_plane
from .flowfield import FlowField, Well
from .zones import isochrones, connection, siting_scan, doublet_critical_distance, polygon_area
from .transport import ogata_banks, streamtube_breakthrough, crank_nicolson_1d

__version__ = '0.1.0'
