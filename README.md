# Predictive Maintenance Machine Learning Model
A Simple MATLAB-based machine learning model for predicting equipment failures using sensor data and maintenance records.

# Overview
This project implements a Random Forest classifier to predict potential equipment failures based on various operational parameters including:
- Air Temperature (K)
- Process Temperature (K)
- Rotational Speed (RPM)
- Torque (Nm)
- Tool Wear (minutes)

# Data
The dataset consists of 10,000 data points with the following features:

- **UID**: Unique identifier (1 to 10000)

- **ProductID**: Product quality variant identifier
  - L: Low (50% of products)
  - M: Medium (30% of products)
  - H: High (20% of products)
 
- **Air Temperature [K]**: Normally distributed around 300K with σ = 2K

- **Process Temperature [K]**: Air temperature + 10K with σ = 1K

- **Rotational Speed [rpm]**: Derived from 2860W power with normal noise

- **Torque [Nm]**: Normally distributed around 40Nm with σ = 10Nm

- **Tool Wear [min]**: Varies by product quality
  - H: +5 minutes
  - M: +3 minutes
  - L: +2 minutes

- **Target Variables**: The model predicts two targets:
  - Binary classification of machine failure (Yes/No)
  - Type of failure (when applicable)

⚠️ **Important**: Both targets should be treated as prediction targets and not used as features to avoid data leakage.

Link: https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification

# Features

- Data preprocessing and feature selection
- 80-20 train-test split
- Random Forest model training with 100 trees
- Feature importance analysis
- Model evaluation using confusion matrix
- Prediction function for new data points
- Model persistence (save/load functionality)


# Usage
## Prerequisites
- MATLAB
  - Statistics and Machine Learning Toolbox
  - Deep Learning Toolbox

## Code Structure:
1. Data Loading and Preprocessing
2. Feature Selection
3. Train-Test Split (80-20)
4. Model Training (Random Forest with 100 trees)
5. Prediction Function
6. Model Evaluation
7. Model Persistence

## Training the Model
- Place your 'maintenance_data.csv' in the working directory
- Run the main script to:
  - Load and preprocess data
  - Train the model
  - Evaluate performance
  - Save the model
- Use the predictMaintenance function for new predictions
  ```
  % Example prediction
  newData = [299, 311, 1000, 2.6, 70];
  [predicted_failure, probability] = predictMaintenance(model, newData);
  ```
- The trained model is automatically saved as 'maintenance_model.mat'. To load the saved model:
  ```
  load('maintenance_model.mat');
  ```

## Author
Zenilus

## Date
Last Updated: November 30, 2024
