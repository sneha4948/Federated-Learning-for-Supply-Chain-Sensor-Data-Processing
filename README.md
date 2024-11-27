# Cold Chain Logistics: Classification and Imputation Models using Federated Learning

## Introduction

This repository contains code and datasets for implementing classification and imputation tasks in the context of cold chain logistics using federated learning strategies. The aim is to monitor and maintain the integrity of temperature-sensitive products during transportation by analyzing sensor data from IoT devices in cold storage containers.

## Project Overview

The project addresses two main tasks:

1. **Classification Task**: Classify the condition of cold storage containers based on sensor data into categories that reflect operational statuses, such as "Normal Operation", "Door Open", and "Human Intervention".

2. **Imputation Task**: Impute missing sensor data to ensure data integrity for reliable online decision-making, addressing issues like communication failures or sensor malfunctions.

Both tasks leverage federated learning strategies to preserve data privacy across distributed clients.

## Repository Structure

```
.
├── Classification Model
│   ├── 3_layer_MLP_Experiment_with_Simulated_data_flower_local.ipynb
│   ├── Hybrid_CNN+RNN_Classification_with_Simulated_data_flower_local.ipynb
│   └── MLP_Classification_with_Simulated_data_flower_local.ipynb
├── Dataset
│   ├── Data.xlsx
│   ├── container_conditions.csv
│   └── sensor3_1000data.csv
└── Imputation Model
    ├── LSTM_Imputation_with_Simulated_data_flower_local.ipynb
    ├── MLP_Imputation_with_Simulated_data_flower_local.ipynb
    └── VAR_KNN_Imputation_with_Simulated_data_flower_local.ipynb
```

### Classification Model

Contains Jupyter notebooks for the classification task:

- **3_layer_MLP_Experiment_with_Simulated_data_flower_local.ipynb**: Experiments with a 3-layer Multilayer Perceptron (MLP) model.
- **Hybrid_CNN+RNN_Classification_with_Simulated_data_flower_local.ipynb**: Implements a hybrid Convolutional Neural Network (CNN) and Recurrent Neural Network (RNN) model.
- **MLP_Classification_with_Simulated_data_flower_local.ipynb**: Basic MLP model for classification.

### Dataset

Includes the datasets used for training and evaluation:

- **Data.xlsx**: Original dataset containing sensor readings.
- **container_conditions.csv**: Processed dataset with labeled container conditions.
- **sensor3_1000data.csv**: Subset of data from Sensor 3, consisting of 1000 data points.

### Imputation Model

Contains Jupyter notebooks for the imputation task:

- **LSTM_Imputation_with_Simulated_data_flower_local.ipynb**: Imputation using Long Short-Term Memory (LSTM) networks.
- **MLP_Imputation_with_Simulated_data_flower_local.ipynb**: Imputation using MLPs.
- **VAR_KNN_Imputation_with_Simulated_data_flower_local.ipynb**: Comparative analysis using Vector Autoregression (VAR) and K-Nearest Neighbors (KNN) imputation methods.

## Requirements

To run the code, you need the following installed:

- Python 3.x
- Jupyter Notebook or JupyterLab
- Required Python packages (can be installed via `requirements.txt` or manually):

  ```bash
  pip install pandas numpy scikit-learn tensorflow flwr statsmodels matplotlib seaborn
  ```

## Getting Started

1. **Clone the Repository**

   ```bash
   git clone https://github.com/sneha4948/Federated-Learning-for-Supply-Chain-Sensor-Data-Processing.git
   cd Federated-Learning-for-Supply-Chain-Sensor-Data-Processing
   ```

2. **Set Up the Environment**

   Create a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use venv\Scripts\activate
   ```

   Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Notebooks**

   Open Jupyter Notebook or JupyterLab:

   ```bash
   jupyter notebook
   ```

   Navigate to the desired notebook under the **Classification Model** or **Imputation Model** folders.

## Project Details

### Classification Task

- **Objective**: Classify the container conditions based on sensor readings to detect anomalies like door openings or human interventions.
- **Models Used**:
  - Multilayer Perceptron (MLP) with varying depths (3, 5, 7 layers) and neuron counts (64, 128, 256 neurons per layer).
  - Hybrid CNN-RNN architecture combining convolutional layers and LSTM units.
- **Federated Learning Strategies**:
  - Synchronous Methods: FedAvg, FedProx.
  - Asynchronous Methods: FedBuff, FedAsync.

### Imputation Task

- **Objective**: Impute missing sensor data to ensure data integrity and support reliable online decision-making.
- **Models Used**:
  - MLP-based imputation.
  - LSTM-based imputation.
- **Comparative Methods**:
  - VAR (Vector Autoregression) and KNN (K-Nearest Neighbors) imputation for benchmark comparison.

## Data Description

- **Data.xlsx**: Original dataset containing sensor readings from multiple sensors (Temperature, Humidity, CO2 levels).
- **container_conditions.csv**: Preprocessed dataset with labeled conditions:
  - **Normal Operation**: Optimal conditions with all systems functioning correctly.
  - **Door Open**: Periods when the container door is open.
  - **Human Intervention**: Times when manual adjustments or inspections occur.
- **sensor3_1000data.csv**: Subset of data from Sensor 3, consisting of 1000 data points, used for experiments requiring smaller datasets.

## How to Use

1. **Classification Models**:

   - Navigate to the **Classification Model** folder.
   - Open the desired notebook (e.g., `3_layer_MLP_Experiment_with_Simulated_data_flower_local.ipynb`).
   - Ensure that the dataset files are accessible in the specified paths.
   - Run the cells sequentially to train and evaluate the model using federated learning strategies.

2. **Imputation Models**:

   - Navigate to the **Imputation Model** folder.
   - Open the desired notebook (e.g., `LSTM_Imputation_with_Simulated_data_flower_local.ipynb`).
   - Run the cells to perform data imputation and evaluate the performance metrics.

## Results

- **Classification**:

  - The models achieve high accuracy in classifying container conditions.
  - Synchronous federated learning strategies (FedAvg, FedProx) generally provide stable performance.
  - The hybrid CNN-RNN model captures both spatial and temporal features effectively.

- **Imputation**:

  - LSTM-based imputation performs well in datasets with temporal dependencies.
  - MLP-based imputation offers a simpler alternative with reasonable accuracy.
  - VAR and KNN methods provide benchmark results but are not suitable for federated learning due to their non-parametric nature.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
