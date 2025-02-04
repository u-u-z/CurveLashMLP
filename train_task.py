# Import required packages
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from train_model import train_curve_model, predict_B_curve
from utils import parse_svg_path, bezier_sample

# Use example SVG data
A_d = "M14 57.0001C29.3464 35.0113 51.5 22 85 22C110 22 134 25 160.5 52.5"
B_d = "M0.5 65.5C28.9136 9.86327 64 -3.92666 102 1.99999C140 7.92664 170 29.5 181 40"

# Get curve points
A_segments = parse_svg_path(A_d)
B_segments = parse_svg_path(B_d)

A_points = bezier_sample(A_segments)
B_points = bezier_sample(B_segments)


def plot_curve_variations(A_curves, B_curves, A_points, B_points):
    # Create output directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)

def generate_curve_variations(A_points, B_points, num_variations=100):
    A_curves = []
    B_curves = []
    plt.figure(figsize=(20, 15))
    
    for i in range(num_variations):
        # Overall translation parameters
        shift_x = np.random.normal(0, 2.0)  # x-direction translation, std=2.0
        shift_y = np.random.normal(0, 2.0)  # y-direction translation, std=2.0
        
        # Curvature parameters (changed by scaling y-coordinates)
        curve_scale_A = np.random.normal(1.0, 0.1)  # Random variation around 1.0, std=0.1
        curve_scale_B = np.random.normal(1.0, 0.1)
        
        # Apply transformation to curve A
        A_curve = A_points.copy()
        # Ensure continuity of start and end points before transformation
        if np.linalg.norm(A_curve[-1] - A_curve[-2]) < 1e-6:
            # If the last two points are too close, slightly adjust the last point
            direction = A_curve[-1] - A_curve[-2]
            direction = direction / np.linalg.norm(direction)
            A_curve[-1] = A_curve[-2] + direction * 0.1
        
        A_curve[:, 0] += shift_x  # x translation
        A_curve[:, 1] = A_points[:, 1] * curve_scale_A + shift_y  # y scaling and translation
        
        # Apply transformation to curve B
        B_curve = B_points.copy()
        # Ensure continuity of start and end points before transformation
        if np.linalg.norm(B_curve[-1] - B_curve[-2]) < 1e-6:
            direction = B_curve[-1] - B_curve[-2]
            direction = direction / np.linalg.norm(direction)
            B_curve[-1] = B_curve[-2] + direction * 0.1
            
        B_curve[:, 0] += shift_x  # x translation
        B_curve[:, 1] = B_points[:, 1] * curve_scale_B + shift_y  # y scaling and translation
        
        A_curves.append(A_curve)
        B_curves.append(B_curve)
        
        # Draw a plot every 10 iterations
        if (i + 1) % 10 == 0:
            ax = plt.subplot(2, 5, (i + 1) // 10)
            
            # Draw original curves (semi-transparent)
            plt.plot(A_points[:, 0], -A_points[:, 1], 'b-', alpha=0.3, label='Original A')
            plt.plot(B_points[:, 0], -B_points[:, 1], 'r-', alpha=0.3, label='Original B')
            
            # Draw transformed curves
            plt.plot(A_curve[:, 0], -A_curve[:, 1], 'b--', 
                    label=f'A {i+1}\n(shift:{shift_x:.1f},{shift_y:.1f})')
            plt.plot(B_curve[:, 0], -B_curve[:, 1], 'r--', 
                    label=f'B {i+1}\n(scale:{curve_scale_A:.2f})')
            
            plt.title(f'Iteration {i+1}')
            plt.grid(True)
            if i == 0:  # Only show legend in the first subplot
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                
            # Set axis scale and range
            ax.set_aspect('equal', adjustable='box')
            
            # Get range of all points and add some margin
            all_points = np.vstack([A_points, B_points, A_curve, B_curve])
            x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
            y_min, y_max = -all_points[:, 1].max(), -all_points[:, 1].min()
            
            margin = 10
            ax.set_xlim(x_min - margin, x_max + margin)
            ax.set_ylim(y_min - margin, y_max + margin)
    
    plt.figure(figsize=(20, 10))  # Increase figure size
    plt.tight_layout()
    plt.savefig('plots/curve_variations.png')
    plt.close()
    
    return A_curves, B_curves

def print_noise_statistics(A_curves, B_curves, A_points, B_points):
    print("Noise Statistics:")
    A_noise_avg = np.mean([np.mean(np.sqrt(np.sum((A - A_points)**2, axis=1))) for A in A_curves])
    B_noise_avg = np.mean([np.mean(np.sqrt(np.sum((B - B_points)**2, axis=1))) for B in B_curves])
    print(f"Average noise amplitude for curve A: {A_noise_avg:.4f} pixels")
    print(f"Average noise amplitude for curve B: {B_noise_avg:.4f} pixels")

def plot_training_history(history):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('plots/training_history.png')
    plt.close()

def train_and_evaluate(A_curves, B_curves, A_points, show_plots=False):
    # Train model
    model, history, X_scaler, y_scaler = train_curve_model(
        A_curves, 
        B_curves,
        epochs=50,  # Can be adjusted as needed
        batch_size=32
    )
    
    if show_plots:
        plot_training_history(history)
    
    return model, X_scaler, y_scaler
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train curve prediction model')
    parser.add_argument('--show', action='store_true', help='Show and save plots')
    args = parser.parse_args()
    
    # Use example SVG data
    A_d = "M14 57.0001C29.3464 35.0113 51.5 22 85 22C110 22 134 25 160.5 52.5"
    B_d = "M0.5 65.5C28.9136 9.86327 64 -3.92666 102 1.99999C140 7.92664 170 29.5 181 40"

    # Get curve points
    A_segments = parse_svg_path(A_d)
    B_segments = parse_svg_path(B_d)

    A_points = bezier_sample(A_segments)
    B_points = bezier_sample(B_segments)
    
    # Generate curve variations
    A_curves, B_curves = generate_curve_variations(A_points, B_points)
    
    # Print statistics
    print_noise_statistics(A_curves, B_curves, A_points, B_points)
    
    # Train model and get predictions
    model, X_scaler, y_scaler = train_and_evaluate(A_curves, B_curves, A_points, show_plots=args.show)
    predicted_B_points = predict_B_curve(model, A_points, X_scaler, y_scaler)
    
    if args.show:
        # Plot final results
        plt.figure(figsize=(12, 8))
        plt.plot(A_points[:, 0], -A_points[:, 1], 'b-', label='Input Curve A')
        plt.plot(B_points[:, 0], -B_points[:, 1], 'r-', label='Target Curve B')
        plt.plot(predicted_B_points[:, 0], -predicted_B_points[:, 1], 'g--', label='Predicted Curve B')
        plt.title('Curve Prediction Results')
        plt.grid(True)
        plt.legend()
        plt.axis('equal')
        plt.savefig('plots/prediction_results.png')
        plt.close()

    # Print evaluation metrics
    test_error = np.mean(np.sqrt(np.sum((predicted_B_points - B_points)**2, axis=1)))
    print(f"Test error: {test_error:.4f} pixels")

if __name__ == '__main__':
    main()
