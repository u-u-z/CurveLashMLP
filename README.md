# Curve AB Research Project

This project focuses on analyzing and predicting the relative positions between two curves (Curve A and Curve B) using machine learning techniques.

## Overview

The project implements a neural network model to study and predict the relationship between two curves in terms of their relative positions and movements. This has potential applications in:

- Robot path planning and following behavior
- Relative motion analysis between two objects
- Dance choreography for dual movements
- Industrial automation and coordinated motion control

## Technical Implementation

### Neural Network Architecture
The model uses a Multi-Layer Perceptron (MLP) with the following structure:
- Input layer: 1 neuron
- Hidden layer 1: 64 neurons with ReLU activation
- Hidden layer 2: 128 neurons with ReLU activation
- Hidden layer 3: 64 neurons with ReLU activation
- Output layer: 2 neurons (for tangential and normal offsets)

The model is trained using:
- Adam optimizer
- Mean Squared Error (MSE) loss function
- Mean Absolute Error (MAE) as a metric

### Data Analysis Views

The project provides three main visualization perspectives:

1. **Original Curves View**
   - Shows both curves in their original coordinate system
   - Blue line represents Curve A
   - Red line represents Curve B
   - Provides intuitive understanding of spatial relationships

2. **Relative Position Graph**
   - X-axis: Normalized arc length parameter of Curve A (0 to 1)
   - Blue line: Tangential offset (front-back distance)
   - Red line: Normal offset (left-right distance)
   - Visualizes Curve B's offset relative to each position on Curve A

3. **3D Relative Motion View**
   - X-axis: Normalized arc length parameter of Curve A
   - Y-axis: Tangential offset
     - Positive values: B is in front of A
     - Negative values: B is behind A
   - Z-axis: Normal offset
     - Positive values: B is to the left of A
     - Negative values: B is to the right of A

## Dependencies

- NumPy
- TensorFlow
- Matplotlib
- SciPy
- svg.path

## Usage

The main research and implementation can be found in `research.ipynb` Jupyter notebook. The notebook contains:
- Data preprocessing and curve analysis
- Model training and evaluation
- Visualization of results
- Detailed explanations of the methodology

## Results

The model demonstrates the ability to:
1. Learn the relative positioning patterns between curves
2. Predict offsets based on arc length parameters
3. Visualize complex spatial relationships in multiple perspectives

## License

This project is part of the Kigland Research initiative. All rights reserved.
