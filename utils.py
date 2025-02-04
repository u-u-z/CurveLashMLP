import numpy as np
from svg.path import parse_path, CubicBezier, Line

def parse_svg_path(d):
    """Parse SVG path into Bezier curve segments
    
    Args:
        d (str): SVG path string
    
    Returns:
        list: List of dictionaries containing curve segment information
    """
    path = parse_path(d)
    segments = []
    for e in path:
        if isinstance(e, CubicBezier):
            segments.append({
                'start': (e.start.real, e.start.imag),
                'c1': (e.control1.real, e.control1.imag),
                'c2': (e.control2.real, e.control2.imag),
                'end': (e.end.real, e.end.imag)
            })
        elif isinstance(e, Line):
            segments.append({
                'start': (e.start.real, e.start.imag),
                'end': (e.end.real, e.end.imag)
            })
    return segments


def bezier_sample(segments, n_samples=50):
    """Sample points from Bezier curves
    
    Args:
        segments (list): List of curve segments
        n_samples (int): Number of sample points per curve segment
    
    Returns:
        np.array: Array of sampled points with shape (n_points, 2)
    """
    points = []
    for seg in segments:
        if 'c1' in seg:  # Cubic Bezier curve
            t = np.linspace(0, 1, n_samples)
            x = (1-t)**3*seg['start'][0] + 3*(1-t)**2*t*seg['c1'][0] + 3*(1-t)*t**2*seg['c2'][0] + t**3*seg['end'][0]
            y = (1-t)**3*seg['start'][1] + 3*(1-t)**2*t*seg['c1'][1] + 3*(1-t)*t**2*seg['c2'][1] + t**3*seg['end'][1]
            points.extend(list(zip(x, y)))
        else:  # Line segment
            x = np.linspace(seg['start'][0], seg['end'][0], n_samples)
            y = np.linspace(seg['start'][1], seg['end'][1], n_samples)
            points.extend(list(zip(x, y)))
    return np.array(points)