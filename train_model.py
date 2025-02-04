import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def prepare_curve_data(A_points, B_points, n_samples=50):
    """
    Prepare curve training data
    
    Args:
        A_points (np.array): Sampling points of curve A
        B_points (np.array): Sampling points of curve B
        n_samples (int): Number of sampling points
    
    Returns:
        X: Position parameters of curve A
        y: Tangential and normal offsets of curve B
    
    Formula ref: README.md#2.1-Curve-Data-Preparation
    """
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
    
    # Check for invalid values
    if np.any(np.isnan(A_params)) or np.any(np.isnan(offsets)):
        print("Warning: Data contains NaN values")
        print("A_params statistics:", np.nanmin(A_params), np.nanmax(A_params))
        print("offsets statistics:", np.nanmin(offsets), np.nanmax(offsets))
    
    return A_params.reshape(-1, 1), offsets

def create_model():
    """
    Create neural network model
    
    Formula ref: README.md#3.2-Neural-Network-Model
    """
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(1,)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(2)  # Output tangential and normal offsets
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model

def train_curve_model(A_curves, B_curves, epochs=100, batch_size=32):
    """
    Train curve prediction model
    
    Args:
        A_curves (list): List of curve A points
        B_curves (list): List of curve B points
        epochs (int): Number of training epochs
        batch_size (int): Batch size
    
    Returns:
        model: Trained model
        history: Training history
    
    Formula ref: README.md#3.3-Curve-Model-Training
    """
    # Prepare data
    X_all = []
    y_all = []
    
    for A_points, B_points in zip(A_curves, B_curves):
        X, y = prepare_curve_data(A_points, B_points)
        X_all.append(X)
        y_all.append(y)
    
    X_all = np.vstack(X_all)
    y_all = np.vstack(y_all)
    
    # Data standardization
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    X_scaled = X_scaler.fit_transform(X_all)
    y_scaled = y_scaler.fit_transform(y_all)
    
    # Split training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )
    
    # Create and train model
    model = create_model()
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=1
    )
    
    return model, history, X_scaler, y_scaler

def predict_B_curve(model, A_points, X_scaler, y_scaler):
    """
    Predict curve B
    
    Args:
        model: Trained model
        A_points (np.array): Points of curve A
        X_scaler: Input data standardizer
        y_scaler: Output data standardizer
    
    Returns:
        np.array: Predicted points of curve B
    
    Formula ref: README.md#3.4-Curve-Prediction
    """
    # Prepare curve A parameters
    A_diffs = np.diff(A_points, axis=0)
    A_lengths = np.sqrt(np.sum(A_diffs**2, axis=1))
    A_cumsum = np.cumsum(A_lengths)
    A_cumsum = np.insert(A_cumsum, 0, 0)
    
    # 防止除以零
    total_length = A_cumsum[-1]
    if total_length < 1e-10:
        print("Warning: Curve total length is close to zero during prediction")
        total_length = 1e-10
        
    A_params = A_cumsum / total_length
    
    # Check and print debug information
    print("A_params range:", np.min(A_params), np.max(A_params))
    
    # Standardize input
    X_scaled = X_scaler.transform(A_params.reshape(-1, 1))
    print("X_scaled range:", np.min(X_scaled), np.max(X_scaled))
    
    # Predict offsets
    offsets_scaled = model.predict(X_scaled)
    print("Prediction range:", np.min(offsets_scaled), np.max(offsets_scaled))
    
    offsets = y_scaler.inverse_transform(offsets_scaled)
    print("Range after inverse standardization:", np.min(offsets), np.max(offsets))
    
    # Reconstruct curve B points
    B_points = []
    for i in range(len(A_points)):
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
        normal = np.array([-tangent[1], tangent[0]])
        
        # Reconstruct B point
        B_point = (A_points[i] + 
                  offsets[i][0] * tangent + 
                  offsets[i][1] * normal)
        B_points.append(B_point)
    
    B_points = np.array(B_points)
    
    # 检查最终结果
    if np.any(np.isnan(B_points)):
        print("Warning: Prediction contains NaN values")
        print("NaN values at positions: ", np.where(np.isnan(B_points)))
    
    return B_points
