import numpy as np
from pathlib import Path
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


def load_data(data_dir):
    """Load curve data from data directory
    
    Args:
        data_dir (Path): Path to data directory
        
    Returns:
        tuple: (A_points, B_points) arrays of curve points
    """
    data_dir = Path(data_dir)
    
    # Load curve A data
    with open(data_dir / 'curve_A.txt', 'r') as f:
        A_path = f.read().strip()
    A_segments = parse_svg_path(A_path)
    A_points = bezier_sample(A_segments)
    
    # Load curve B data
    with open(data_dir / 'curve_B.txt', 'r') as f:
        B_path = f.read().strip()
    B_segments = parse_svg_path(B_path)
    B_points = bezier_sample(B_segments)
    
    return A_points, B_points


def preprocess_data(A_points, B_points):
    """Preprocess curve data for model input/output
    
    Args:
        A_points (np.array): Points from curve A
        B_points (np.array): Points from curve B
        
    Returns:
        tuple: (X, y) arrays for model training
    """
    X = A_points.reshape(1, -1, 2)
    y = B_points.reshape(1, -1, 2)
    return X, y