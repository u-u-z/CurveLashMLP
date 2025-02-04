import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import argparse
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from utils import parse_svg_path, bezier_sample

def arc_length_parameterization(points):
    """Arc length parameterization"""
    dx = np.diff(points[:,0])
    dy = np.diff(points[:,1])
    seg_lengths = np.sqrt(dx**2 + dy**2)
    cum_length = np.cumsum(seg_lengths)
    total_length = cum_length[-1]
    normalized = np.insert(cum_length, 0, 0) / total_length
    return normalized, total_length

def compute_tangent_normal(points):
    """Compute curve tangent and normal vectors"""
    dx = np.gradient(points[:,0])
    dy = np.gradient(points[:,1])
    
    lengths = np.sqrt(dx**2 + dy**2)
    tangent = np.column_stack([dx/lengths, dy/lengths])
    normal = np.column_stack([-tangent[:,1], tangent[:,0]])
    
    return tangent, normal

def transform_to_local_coordinates(point, origin, tangent, normal):
    """Transform point to local coordinate system"""
    translated = point - origin
    x_local = np.dot(translated, tangent)
    y_local = np.dot(translated, normal)
    return np.array([x_local, y_local])

def prepare_curve_data(A_points, B_points):
    """Prepare curve training data"""
    # Calculate position parameters of curve A (arc length normalization)
    A_diffs = np.diff(A_points, axis=0)
    A_lengths = np.sqrt(np.sum(A_diffs**2, axis=1))
    A_cumsum = np.cumsum(A_lengths)
    A_cumsum = np.insert(A_cumsum, 0, 0)
    
    # Prevent division by zero
    total_length = A_cumsum[-1]
    if total_length < 1e-10:  # If total length is too small
        print("Warning: Curve total length is close to zero")
        total_length = 1e-10
        
    A_params = A_cumsum / total_length
    
    # Calculate offsets of curve B relative to curve A
    offsets = []
    for i in range(len(A_points)):
        # Calculate tangent vector of curve A at this point
        if i == len(A_points) - 1:
            tangent = A_points[i] - A_points[i-1]
        else:
            tangent = A_points[i+1] - A_points[i]
            
        # Prevent zero vector
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm < 1e-10:
            print(f"Warning: Tangent vector at point {i} is close to zero")
            tangent = np.array([1e-10, 0])
            tangent_norm = np.linalg.norm(tangent)
            
        tangent = tangent / tangent_norm
        
        # Calculate normal vector (perpendicular to tangent vector)
        normal = np.array([-tangent[1], tangent[0]])
        
        # Calculate offset of point B relative to point A
        diff = B_points[i] - A_points[i]
        
        # Decompose into tangential and normal components
        tangential_offset = np.dot(diff, tangent)
        normal_offset = np.dot(diff, normal)
        
        offsets.append([tangential_offset, normal_offset])
    
    offsets = np.array(offsets)
    return A_params.reshape(-1, 1), offsets

def predict_B_curve(model, A_points, X_scaler, y_scaler):
    """Predict curve B from curve A using the trained model"""
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

def validate_model_multiple_times(model_path, A_points, B_points, X_scaler, y_scaler, n_validations=100, show_plots=False):
    """
    Validate model multiple times with random transformations
    
    Args:
        model_path: Path to model file
        A_points: Points of curve A
        B_points: Points of curve B
        X_scaler: Input data scaler
        y_scaler: Output data scaler
        n_validations: Number of validations
        show_plots: Whether to display plots
    """
    # Load model
    model = keras.models.load_model(model_path, compile=False)
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    errors = []
    all_A_curves = []
    all_B_curves = []
    all_pred_B = []
    
    for i in range(n_validations):
        # Translation parameters
        shift_x = np.random.normal(0, 2.0)
        shift_y = np.random.normal(0, 2.0)
        
        # Scaling parameters
        curve_scale_A = np.random.normal(1.0, 0.1)
        curve_scale_B = np.random.normal(1.0, 0.1)
        
        # Transform curve A
        A_curve = A_points.copy()
        if np.linalg.norm(A_curve[-1] - A_curve[-2]) < 1e-6:
            direction = A_curve[-1] - A_curve[-2]
            direction = direction / np.linalg.norm(direction)
            A_curve[-1] = A_curve[-2] + direction * 0.1
        
        A_curve[:, 0] += shift_x
        A_curve[:, 1] = A_points[:, 1] * curve_scale_A + shift_y
        
        # Transform curve B
        B_curve = B_points.copy()
        if np.linalg.norm(B_curve[-1] - B_curve[-2]) < 1e-6:
            direction = B_curve[-1] - B_curve[-2]
            direction = direction / np.linalg.norm(direction)
            B_curve[-1] = B_curve[-2] + direction * 0.1
        
        B_curve[:, 0] += shift_x
        B_curve[:, 1] = B_points[:, 1] * curve_scale_B + shift_y
        
        # Predict curve B
        predicted_B = predict_B_curve(model, A_curve, X_scaler, y_scaler)
        
        # Store curves for overlap plot
        all_A_curves.append(A_curve)
        all_B_curves.append(B_curve)
        all_pred_B.append(predicted_B)
        
        # Calculate error
        error = np.mean(np.sqrt(np.sum((predicted_B - B_curve)**2, axis=1)))
        errors.append(error)
        
        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1}/{n_validations} validations")
    
    if show_plots:
        # Individual validation plots
        plt.figure(figsize=(20, 10))
        for i in range(min(10, n_validations)):
            ax = plt.subplot(2, 5, i + 1)
            
            plt.plot(A_points[:, 0], -A_points[:, 1], 'b-', alpha=0.3, label='Original A')
            plt.plot(B_points[:, 0], -B_points[:, 1], 'r-', alpha=0.3, label='Original B')
            
            plt.plot(all_A_curves[i][:, 0], -all_A_curves[i][:, 1], 'b--', 
                    label=f'A {i+1}')
            plt.plot(all_B_curves[i][:, 0], -all_B_curves[i][:, 1], 'r--', 
                    label=f'B {i+1}')
            plt.plot(all_pred_B[i][:, 0], -all_pred_B[i][:, 1], 'g--', 
                    label=f'Pred B\n(err:{errors[i]:.2f})')
            
            plt.title(f'Test {i+1}')
            plt.grid(True)
            if i == 0:
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.show()

        # Overlap plot
        plt.figure(figsize=(12, 8))
        plt.plot(A_points[:, 0], -A_points[:, 1], 'b-', linewidth=2, label='Original A', alpha=0.8)
        plt.plot(B_points[:, 0], -B_points[:, 1], 'r-', linewidth=2, label='Original B', alpha=0.8)
        
        for i in range(n_validations):
            plt.plot(all_A_curves[i][:, 0], -all_A_curves[i][:, 1], 'b-', alpha=0.1)
            plt.plot(all_B_curves[i][:, 0], -all_B_curves[i][:, 1], 'r-', alpha=0.1)
            plt.plot(all_pred_B[i][:, 0], -all_pred_B[i][:, 1], 'g-', alpha=0.1)
        
        plt.title('Overlap of All Validation Results')
        plt.grid(True)
        plt.legend(['Original A', 'Original B', 'Transformed A', 'Transformed B', 'Predicted B'])
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

        # 3D visualization
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        for i in range(n_validations):
            # Process curves
            s_values, local_coords = process_curves(all_A_curves[i], all_pred_B[i])
            ax.plot(s_values, local_coords[:,0], local_coords[:,1],
                    alpha=0.5, linewidth=1,
                    color=plt.cm.viridis(i/n_validations))
        
        ax.set_xlabel('Arc Length')
        ax.set_ylabel('Tangential Offset')
        ax.set_zlabel('Normal Offset')
        ax.set_title('3D View of Relative Motion')
        
        norm = plt.Normalize(0, n_validations)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Validation Index')
        plt.tight_layout()
        plt.show()

        # Error distribution plot
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=20, edgecolor='black')
        plt.title('Error Distribution')
        plt.xlabel('Prediction Error (pixels)')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

    # Print validation statistics
    print("\nValidation Statistics:")
    print(f"Mean Error: {np.mean(errors):.4f} ± {np.std(errors):.4f} pixels")
    print(f"Min Error: {np.min(errors):.4f} pixels")
    print(f"Max Error: {np.max(errors):.4f} pixels")
    
    return errors

def process_curves(curve_A, curve_B):
    """Process curves for 3D visualization"""
    # Arc length parameterization
    s_A, _ = arc_length_parameterization(curve_A)
    tangent_A, normal_A = compute_tangent_normal(curve_A)
    
    # Create dense sampling of curve B
    num_samples = 1000
    t_dense = np.linspace(0, 1, num_samples)
    B_dense = np.column_stack([
        np.interp(t_dense, np.linspace(0, 1, len(curve_B)), curve_B[:,0]),
        np.interp(t_dense, np.linspace(0, 1, len(curve_B)), curve_B[:,1])
    ])
    
    # Arc length parameterization for dense B curve
    s_B_dense, _ = arc_length_parameterization(B_dense)
    
    # Sample points on curve A
    num_A_samples = 200
    s_values = np.linspace(0, 1, num_A_samples)
    A_sampled = np.column_stack([
        np.interp(s_values, np.linspace(0, 1, len(curve_A)), curve_A[:,0]),
        np.interp(s_values, np.linspace(0, 1, len(curve_A)), curve_A[:,1])
    ])
    
    # Compute tangent and normal vectors for sampled points
    tangent_sampled, normal_sampled = compute_tangent_normal(A_sampled)
    
    # Calculate local coordinates
    local_coordinates = []
    for i in range(len(s_values)):
        idx = np.argmin(np.abs(s_B_dense - s_values[i]))
        B_point = B_dense[idx]
        
        local_coord = transform_to_local_coordinates(
            B_point,
            A_sampled[i],
            tangent_sampled[i],
            normal_sampled[i]
        )
        local_coordinates.append(local_coord)
        
    return s_values, np.array(local_coordinates)

def main():
    parser = argparse.ArgumentParser(description='Test curve prediction model')
    parser.add_argument('--show', action='store_true', help='Show validation plots')
    parser.add_argument('--validations', type=int, default=100, help='Number of validation iterations')
    args = parser.parse_args()

    # Use example SVG data from training
    A_d = "M14 57.0001C29.3464 35.0113 51.5 22 85 22C110 22 134 25 160.5 52.5"
    B_d = "M0.5 65.5C28.9136 9.86327 64 -3.92666 102 1.99999C140 7.92664 170 29.5 181 40"

    # Get curve points
    A_segments = parse_svg_path(A_d)
    B_segments = parse_svg_path(B_d)

    A_points = bezier_sample(A_segments)
    B_points = bezier_sample(B_segments)

    # Prepare training data for scaling
    X, y = prepare_curve_data(A_points, B_points)
    
    # Create and fit scalers
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    X_scaler.fit(X)
    y_scaler.fit(y)

    # Validate model
    errors = validate_model_multiple_times(
        'curve_model.keras',
        A_points,
        B_points,
        X_scaler,
        y_scaler,
        n_validations=args.validations,
        show_plots=args.show
    )

if __name__ == '__main__':
    main()
