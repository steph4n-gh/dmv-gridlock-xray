"""
DMV Gridlock X-Ray - Consolidated Shared Utilities Module.

Contains Potomac and Anacostia River Barrier geometries, coordinate math
for intersection detection, and the smooth_floor Softplus formulation.
"""

import math
from typing import Union, Tuple, List
import numpy as np

# --- River Geometries ---
POTOMAC_BARRIER: List[Tuple[Tuple[float, float], Tuple[float, float]]] = [
    ((38.995, -77.162), (38.960, -77.130)),
    ((38.960, -77.130), (38.930, -77.115)),
    ((38.930, -77.115), (38.900, -77.070)),
    ((38.900, -77.070), (38.888, -77.060)),
    ((38.888, -77.060), (38.875, -77.043)),
    ((38.875, -77.043), (38.850, -77.040)),
    ((38.850, -77.040), (38.790, -77.035))
]

ANACOSTIA_BARRIER: List[Tuple[Tuple[float, float], Tuple[float, float]]] = [
    ((38.935, -76.940), (38.915, -76.955)),
    ((38.915, -76.955), (38.900, -76.965)),
    ((38.900, -76.965), (38.875, -76.980)),
    ((38.875, -76.980), (38.860, -77.010)),
    ((38.860, -77.010), (38.858, -77.025))
]

# --- Coordinate Math ---

def ccw(
    A: Tuple[float, float], 
    B: Tuple[float, float], 
    C: Tuple[float, float]
) -> bool:
    """
    Evaluates the counter-clockwise orientation of coordinates A, B, and C.
    
    Formula:
        ccw(A, B, C) = (C_lat - A_lat)*(B_lon - A_lon) > (B_lat - A_lat)*(C_lon - A_lon)
    """
    return (C[0] - A[0]) * (B[1] - A[1]) > (B[0] - A[0]) * (C[1] - A[1])


def segments_intersect(
    A: Tuple[float, float], 
    B: Tuple[float, float], 
    C: Tuple[float, float], 
    D: Tuple[float, float]
) -> bool:
    """
    Determines if segment AB intersects segment CD.
    """
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def crosses_river(
    lat1: float, 
    lon1: float, 
    lat2: float, 
    lon2: float
) -> bool:
    """
    Checks if the path between coordinate (lat1, lon1) and (lat2, lon2)
    crosses either the Potomac or Anacostia River barrier segments.
    """
    p1 = (lat1, lon1)
    p2 = (lat2, lon2)
    for seg in POTOMAC_BARRIER:
        if segments_intersect(p1, p2, seg[0], seg[1]):
            return True
    for seg in ANACOSTIA_BARRIER:
        if segments_intersect(p1, p2, seg[0], seg[1]):
            return True
    return False


# --- Softplus Decay Formulation ---

def smooth_floor(
    x: Union[float, np.ndarray, List, Tuple], 
    k: float = 100.0
) -> Union[float, np.ndarray]:
    """
    Piecewise softplus decay formulation ensuring numerical stability floor.
    Accepts float scalars, NumPy arrays, lists, and tuples.
    """
    if isinstance(x, (list, tuple)):
        x = np.array(x)

    if isinstance(x, np.ndarray):
        kx = k * x
        return np.where(kx > 50.0, x, np.log1p(np.exp(np.clip(kx, -50.0, 50.0))) / k)
    else:
        kx = k * x
        if kx > 50.0:
            return x
        return math.log1p(math.exp(kx)) / k


# --- Coordinate Bounding Box Validation ---

def validate_coordinates(
    lat: Union[float, int, str, None, List, dict], 
    lon: Union[float, int, str, None, List, dict]
) -> bool:
    """
    Validates if the given coordinates lie within the DMV metropolitan bounding box.
    Enforces DMV limits: 38.0 <= lat <= 40.0 and -78.0 <= lon <= -76.0.
    Handles invalid types, None, lists, dicts, NaN, Inf, and returns False cleanly.
    """
    if isinstance(lat, bool) or isinstance(lon, bool):
        return False
    try:
        lat_f = float(lat)
        lon_f = float(lon)
    except (ValueError, TypeError):
        return False

    if math.isnan(lat_f) or math.isnan(lon_f) or math.isinf(lat_f) or math.isinf(lon_f):
        return False

    return (38.0 <= lat_f <= 40.0) and (-78.0 <= lon_f <= -76.0)
