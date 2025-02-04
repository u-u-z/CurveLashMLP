from typing import List, Tuple, Union, Optional
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import argparse
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from utils import parse_svg_path, bezier_sample
import onnxruntime as ort
import joblib
import os

@dataclass
class CurveData:
    """Data class for storing curve points and parameters"""
    points: np.ndarray
    params: Optional[np.ndarray] = None
    
    def compute_parameters(self) -> np.ndarray:
        """Compute curve parameters using arc length parameterization"""
        diffs = np.diff(self.points, axis=0)
        lengths = np.sqrt(np.sum(diffs**2, axis=1))
        cumsum = np.cumsum(lengths)
        cumsum = np.insert(cumsum, 0, 0)
        
        total_length = cumsum[-1]
        if total_length < 1e-10:
            print("Warning: Curve total length is close to zero")
            total_length = 1e-10
            
        self.params = cumsum / total_length
        return self.params.reshape(-1, 1)
    
    def get_tangent_normal(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get tangent and normal vectors at given index"""
        if index == len(self.points) - 1:
            tangent = self.points[index] - self.points[index-1]
        else:
            tangent = self.points[index+1] - self.points[index]
            
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm < 1e-10:
            print(f"Warning: Tangent vector at point {index} is close to zero")
            tangent = np.array([1e-10, 0])
            tangent_norm = np.linalg.norm(tangent)
            
        tangent = tangent / tangent_norm
        normal = np.array([-tangent[1], tangent[0]])
        
        return tangent, normal

class CurvePredictor:
    """Base class for curve prediction models"""
    def __init__(self, model, x_scaler: StandardScaler, y_scaler: StandardScaler):
        self.model = model
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
    
    def predict(self, curve_a: CurveData) -> np.ndarray:
        """Predict curve B points from curve A"""
        raise NotImplementedError

class TFPredictor(CurvePredictor):
    """TensorFlow model predictor"""
    def predict(self, curve_a: CurveData) -> np.ndarray:
        # Get parameters
        a_params = curve_a.compute_parameters()
        
        # Standardize input
        a_params_scaled = self.x_scaler.transform(a_params)
        
        # Predict offsets
        offsets_scaled = self.model.predict(a_params_scaled, verbose=0)
        offsets = self.y_scaler.inverse_transform(offsets_scaled)
        
        # Calculate curve B points
        b_points = np.zeros_like(curve_a.points)
        for i in range(len(curve_a.points)):
            tangent, normal = curve_a.get_tangent_normal(i)
            b_points[i] = curve_a.points[i] + offsets[i][0] * tangent + offsets[i][1] * normal
        
        return b_points

class ONNXPredictor(CurvePredictor):
    """ONNX model predictor"""
    def predict(self, curve_a: CurveData) -> np.ndarray:
        # Get parameters
        a_params = curve_a.compute_parameters()
        
        # Standardize input
        a_params_scaled = self.x_scaler.transform(a_params)
        
        # Run inference with ONNX model
        input_name = self.model.get_inputs()[0].name
        ort_inputs = {input_name: a_params_scaled.astype(np.float32)}
        offsets_scaled = self.model.run(None, ort_inputs)[0]
        
        # Inverse transform predictions
        offsets = self.y_scaler.inverse_transform(offsets_scaled)
        
        # Calculate curve B points
        b_points = np.zeros_like(curve_a.points)
        for i in range(len(curve_a.points)):
            tangent, normal = curve_a.get_tangent_normal(i)
            b_points[i] = curve_a.points[i] + offsets[i][0] * tangent + offsets[i][1] * normal
        
        return b_points



def transform_to_local_coordinates(point, origin, tangent, normal):
    """Transform point to local coordinate system"""
    translated = point - origin
    x_local = np.dot(translated, tangent)
    y_local = np.dot(translated, normal)
    return np.array([x_local, y_local])

def prepare_curve_data(A_points, B_points):
    """Prepare curve training data"""
    curve_a = CurveData(A_points)
    A_params = curve_a.compute_parameters()
    
    # Calculate offsets of curve B relative to curve A
    offsets = []
    for i in range(len(A_points)):
        tangent, normal = curve_a.get_tangent_normal(i)
        
        # Calculate offset of point B relative to point A
        diff = B_points[i] - A_points[i]
        
        # Decompose into tangential and normal components
        tangential_offset = np.dot(diff, tangent)
        normal_offset = np.dot(diff, normal)
        
        offsets.append([tangential_offset, normal_offset])
    
    offsets = np.array(offsets)
    return A_params, offsets

def predict_B_curve_tf(model, A_points, X_scaler, y_scaler):
    """Predict curve B from curve A using the trained model"""
    # Create CurveData object and get parameters
    curve_a = CurveData(A_points)
    A_params = curve_a.compute_parameters()
    
    # Standardize input
    X_scaled = X_scaler.transform(A_params.reshape(-1, 1))
    
    # Predict offsets
    offsets_scaled = model.predict(X_scaled)
    offsets = y_scaler.inverse_transform(offsets_scaled)
    
    # Reconstruct curve B points
    B_points = []
    for i in range(len(A_points)):
        if i == len(A_points) - 1:
            tangent = A_points[i] - A_points[i-1]
        else:
            tangent = A_points[i+1] - A_points[i]
            
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm < 1e-10:
            print(f"Warning: Tangent vector at point {i} is close to zero")
            tangent = np.array([1e-10, 0])
            tangent_norm = np.linalg.norm(tangent)
            
        tangent = tangent / tangent_norm
        normal = np.array([-tangent[1], tangent[0]])
        
        B_point = (A_points[i] + 
                  offsets[i][0] * tangent + 
                  offsets[i][1] * normal)
        B_points.append(B_point)
    
    return np.array(B_points)

def predict_B_curve_onnx(onnx_session, A_points, X_scaler, y_scaler):
    """Predict curve B from curve A using the ONNX model"""
    # Prepare curve A parameters
    A_diffs = np.diff(A_points, axis=0)
    A_lengths = np.sqrt(np.sum(A_diffs**2, axis=1))
    A_cumsum = np.cumsum(A_lengths)
    A_cumsum = np.insert(A_cumsum, 0, 0)
    
    total_length = A_cumsum[-1]
    if total_length < 1e-10:
        print("Warning: Curve total length is close to zero during prediction")
        total_length = 1e-10
        
    A_params = A_cumsum / total_length
    
    # Standardize input
    A_params_scaled = X_scaler.transform(A_params.reshape(-1, 1))
    
    # Run inference with ONNX model
    input_name = onnx_session.get_inputs()[0].name
    print(f"Using input name: {input_name}")
    ort_inputs = {input_name: A_params_scaled.astype(np.float32)}
    offsets_scaled = onnx_session.run(None, ort_inputs)[0]
    
    # Inverse transform the predictions
    offsets = y_scaler.inverse_transform(offsets_scaled)
    
    # Calculate curve B points
    B_points = np.zeros_like(A_points)
    for i in range(len(A_points)):
        if i == len(A_points) - 1:
            tangent = A_points[i] - A_points[i-1]
        else:
            tangent = A_points[i+1] - A_points[i]
            
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm < 1e-10:
            tangent = np.array([1e-10, 0])
            tangent_norm = np.linalg.norm(tangent)
            
        tangent = tangent / tangent_norm
        normal = np.array([-tangent[1], tangent[0]])
        
        B_points[i] = A_points[i] + offsets[i][0] * tangent + offsets[i][1] * normal
    
    return B_points

class ValidationVisualizer:
    """Class for visualizing validation results"""
    def __init__(self, original_a: CurveData, original_b: CurveData):
        self.original_a = original_a
        self.original_b = original_b
        
    def plot_validation_results(self, 
                               transformed_as: List[np.ndarray],
                               transformed_bs: List[np.ndarray],
                               predicted_bs: List[np.ndarray],
                               errors: List[float],
                               n_samples: int = 10):
        """Plot validation results"""
        # Individual validation plots
        plt.figure(figsize=(20, 10))
        for i in range(min(n_samples, len(transformed_as))):
            ax = plt.subplot(2, 5, i + 1)
            
            plt.plot(self.original_a.points[:, 0], -self.original_a.points[:, 1], 
                     'b-', alpha=0.3, label='Original A')
            plt.plot(self.original_b.points[:, 0], -self.original_b.points[:, 1], 
                     'r-', alpha=0.3, label='Original B')
            
            plt.plot(transformed_as[i][:, 0], -transformed_as[i][:, 1], 'b--', 
                     label=f'A {i+1}')
            plt.plot(transformed_bs[i][:, 0], -transformed_bs[i][:, 1], 'r--', 
                     label=f'B {i+1}')
            plt.plot(predicted_bs[i][:, 0], -predicted_bs[i][:, 1], 'g--', 
                     label=f'Pred B\n(err:{errors[i]:.2f})')
            
            plt.title(f'Test {i+1}')
            plt.grid(True)
            if i == 0:
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.show()
        
        # Overlap plot
        self._plot_overlap(transformed_as, transformed_bs, predicted_bs)
        
        # 3D visualization
        self._plot_3d_visualization(transformed_as, predicted_bs)
    
    def _plot_overlap(self, transformed_as, transformed_bs, predicted_bs):
        plt.figure(figsize=(12, 8))
        plt.plot(self.original_a.points[:, 0], -self.original_a.points[:, 1], 
                 'b-', linewidth=2, label='Original A', alpha=0.8)
        plt.plot(self.original_b.points[:, 0], -self.original_b.points[:, 1], 
                 'r-', linewidth=2, label='Original B', alpha=0.8)
        
        for i in range(len(transformed_as)):
            plt.plot(transformed_as[i][:, 0], -transformed_as[i][:, 1], 'b-', alpha=0.1)
            plt.plot(transformed_bs[i][:, 0], -transformed_bs[i][:, 1], 'r-', alpha=0.1)
            plt.plot(predicted_bs[i][:, 0], -predicted_bs[i][:, 1], 'g-', alpha=0.1)
        
        plt.title('Overlap of All Validation Results')
        plt.grid(True)
        plt.legend(['Original A', 'Original B', 'Transformed A', 'Transformed B', 'Predicted B'])
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()
    
    def _plot_3d_visualization(self, transformed_as, predicted_bs):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        for i in range(len(transformed_as)):
            s_values, local_coords = process_curves(transformed_as[i], predicted_bs[i])
            ax.plot(s_values, local_coords[:,0], local_coords[:,1],
                    alpha=0.5, linewidth=1,
                    color=plt.cm.viridis(i/len(transformed_as)))
        
        plt.title('3D Visualization of Local Coordinates')
        ax.set_xlabel('Arc Length (s)')
        ax.set_ylabel('Tangential Offset')
        ax.set_zlabel('Normal Offset')
        plt.show()

def validate_model(predictor: CurvePredictor, 
                  curve_a: CurveData,
                  curve_b: CurveData,
                  n_validations: int = 100,
                  show_plots: bool = False) -> Tuple[float, float]:
    """
    Validate model multiple times with random transformations
    
    Args:
        predictor: The curve predictor (TensorFlow or ONNX)
        curve_a: Original curve A data
        curve_b: Original curve B data
        n_validations: Number of validations
        show_plots: Whether to display plots
        
    Returns:
        Tuple of (mean_error, std_error)
    """
    errors = []
    transformed_as = []
    transformed_bs = []
    predicted_bs = []
    
    for i in range(n_validations):
        # Translation parameters
        shift_x = np.random.normal(0, 2.0)
        shift_y = np.random.normal(0, 2.0)
        
        # Scaling parameters
        curve_scale_a = np.random.normal(1.0, 0.1)
        curve_scale_b = np.random.normal(1.0, 0.1)
        
        # Transform curve A
        a_points = curve_a.points.copy()
        if np.linalg.norm(a_points[-1] - a_points[-2]) < 1e-6:
            direction = a_points[-1] - a_points[-2]
            direction = direction / np.linalg.norm(direction)
            a_points[-1] = a_points[-2] + direction * 0.1
        
        a_points[:, 0] += shift_x
        a_points[:, 1] = curve_a.points[:, 1] * curve_scale_a + shift_y
        
        # Transform curve B
        b_points = curve_b.points.copy()
        if np.linalg.norm(b_points[-1] - b_points[-2]) < 1e-6:
            direction = b_points[-1] - b_points[-2]
            direction = direction / np.linalg.norm(direction)
            b_points[-1] = b_points[-2] + direction * 0.1
        
        b_points[:, 0] += shift_x
        b_points[:, 1] = curve_b.points[:, 1] * curve_scale_b + shift_y
        
        # Create transformed curve A and predict B
        curve_a_transformed = CurveData(a_points)
        predicted_b = predictor.predict(curve_a_transformed)
        
        # Store curves for visualization
        transformed_as.append(a_points)
        transformed_bs.append(b_points)
        predicted_bs.append(predicted_b)
        
        # Calculate error
        error = np.mean(np.sqrt(np.sum((predicted_b - b_points)**2, axis=1)))
        errors.append(error)
        
        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1}/{n_validations} validations")
    
    # Calculate statistics
    errors = np.array(errors)
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    
    print(f"\nValidation Statistics:")
    print(f"Mean Error: {mean_error:.4f} ± {std_error:.4f} pixels")
    print(f"Min Error: {np.min(errors):.4f} pixels")
    print(f"Max Error: {np.max(errors):.4f} pixels")
    
    if show_plots:
        visualizer = ValidationVisualizer(curve_a, curve_b)
        visualizer.plot_validation_results(
            transformed_as,
            transformed_bs,
            predicted_bs,
            errors,
            n_samples=10
        )
    
    return mean_error, std_error

def process_curves(curve_A, curve_B):
    """Process curves for 3D visualization"""
    # Create CurveData objects
    curve_a_data = CurveData(curve_A)
    s_values = curve_a_data.compute_parameters().flatten()
    
    # Create dense sampling of curve B
    num_samples = 1000
    t_dense = np.linspace(0, 1, num_samples)
    B_dense = np.column_stack([
        np.interp(t_dense, np.linspace(0, 1, len(curve_B)), curve_B[:,0]),
        np.interp(t_dense, np.linspace(0, 1, len(curve_B)), curve_B[:,1])
    ])
    
    # Create CurveData object for dense B curve
    curve_b_dense = CurveData(B_dense)
    s_B_dense = curve_b_dense.compute_parameters().flatten()
    
    # Sample points on curve A
    num_A_samples = 200
    s_values = np.linspace(0, 1, num_A_samples)
    A_sampled = np.column_stack([
        np.interp(s_values, np.linspace(0, 1, len(curve_A)), curve_A[:,0]),
        np.interp(s_values, np.linspace(0, 1, len(curve_A)), curve_A[:,1])
    ])
    
    # Create CurveData object for sampled points
    curve_a_sampled = CurveData(A_sampled)
    
    # Calculate local coordinates
    local_coordinates = []
    for i in range(len(s_values)):
        idx = np.argmin(np.abs(s_B_dense - s_values[i]))
        B_point = B_dense[idx]
        
        tangent, normal = curve_a_sampled.get_tangent_normal(i)
        
        local_coord = transform_to_local_coordinates(
            B_point,
            A_sampled[i],
            tangent,
            normal
        )
        local_coordinates.append(local_coord)
        
    return s_values, np.array(local_coordinates)

def load_onnx_model(model_path):
    """Load ONNX model and create inference session"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ONNX model not found at {model_path}")
    
    # Create ONNX Runtime session
    session = ort.InferenceSession(model_path)
    return session

def load_model(model_path: str, use_onnx: bool) -> CurvePredictor:
    """Load model and create appropriate predictor"""
    # Load model
    if use_onnx:
        print("Using ONNX model for inference")
        model = ort.InferenceSession(model_path)
    else:
        print("Using TensorFlow model for inference")
        model = keras.models.load_model(model_path)
    
    # Load scalers
    scaler_dir = os.path.dirname(model_path)
    x_scaler = joblib.load(os.path.join(scaler_dir, 'X_scaler.joblib'))
    y_scaler = joblib.load(os.path.join(scaler_dir, 'y_scaler.joblib'))
    
    # Create predictor
    predictor_class = ONNXPredictor if use_onnx else TFPredictor
    return predictor_class(model, x_scaler, y_scaler)

# Example SVG data used for testing
EXAMPLE_CURVE_A = "M14 57.0001C29.3464 35.0113 51.5 22 85 22C110 22 134 25 160.5 52.5"
EXAMPLE_CURVE_B = "M0.5 65.5C28.9136 9.86327 64 -3.92666 102 1.99999C140 7.92664 170 29.5 181 40"

def load_example_curves() -> Tuple[CurveData, CurveData]:
    """Load example curves for testing"""
    # Parse SVG paths and sample points
    A_segments = parse_svg_path(EXAMPLE_CURVE_A)
    B_segments = parse_svg_path(EXAMPLE_CURVE_B)
    
    # Create CurveData objects
    curve_a = CurveData(bezier_sample(A_segments))
    curve_b = CurveData(bezier_sample(B_segments))
    
    return curve_a, curve_b

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test curve prediction model')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to the model file (TF or ONNX)')
    parser.add_argument('--use_onnx', action='store_true', 
                        help='Use ONNX model instead of TensorFlow')
    parser.add_argument('--show_plots', action='store_true', 
                        help='Show validation plots')
    parser.add_argument('--n_validations', type=int, default=100, 
                        help='Number of validation runs')
    
    args = parser.parse_args()
    
    # Load example curves
    curve_a, curve_b = load_example_curves()
    
    # Load model and create predictor
    predictor = load_model(args.model_path, args.use_onnx)
    
    # Run validation
    mean_error, std_error = validate_model(
        predictor,
        curve_a,
        curve_b,
        n_validations=args.n_validations,
        show_plots=args.show_plots
    )
    
    print(f"\nValidation Results:")
    print(f"Mean Error: {mean_error:.4f} ± {std_error:.4f} pixels")

if __name__ == '__main__':
    main()
