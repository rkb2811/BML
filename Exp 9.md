# Manual PCA on Red Wine Quality Dataset

This project manually implements **Principal Component Analysis (PCA)** from scratch using NumPy and visualizes the results with Matplotlib.

## Overview

The code performs the following tasks:

- Loads the **Red Wine Quality** dataset.
- Standardizes the dataset manually (zero mean and unit variance).
- Computes the **covariance matrix**.
- Performs **eigen decomposition** to find eigenvalues and eigenvectors.
- Sorts eigenvalues and eigenvectors to find principal components in order of explained variance.
- Calculates and plots the **explained variance** (Scree Plot).
- Projects the dataset into a **2D space** using the top two principal components and visualizes the transformed data.

## Dataset

The dataset used is `winequality-red.csv`, available from the UCI Machine Learning Repository.  
The features include various physicochemical properties of red wine samples, while the target variable (`quality`) is **excluded** during PCA.

- Dataset location in the code:  
  `/content/drive/MyDrive/winequality-red.csv`

- Separator used in CSV: `;`

## Libraries Used

- `numpy` - For mathematical operations.
- `pandas` - For data loading and basic manipulation.
- `matplotlib` - For visualization (scree plot and 2D projection).

## How the Code Works

1. **Load the Data**:  
   Read the dataset using pandas and separate features (`X`) by dropping the `quality` column.

2. **Standardize Features**:  
   Subtract mean and divide by standard deviation for each feature.

3. **Compute Covariance Matrix**:  
   Calculate the covariance matrix from the standardized features.

4. **Eigen Decomposition**:  
   Obtain eigenvalues and eigenvectors of the covariance matrix.

5. **Sort Components**:  
   Sort eigenvalues and corresponding eigenvectors in descending order.

6. **Explained Variance**:  
   Compute the proportion of variance explained by each principal component.

7. **Visualization**:
   - **Scree Plot**: Shows the explained variance for each principal component.
   - **2D PCA Scatter Plot**: Projects the data onto the top 2 principal components and plots it.

## Visual Outputs

- **Scree Plot**:  
  Helps determine how many components explain the majority of variance.

- **2D PCA Scatter Plot**:  
  Visualizes the dataset compressed into 2 dimensions based on the most significant components.

## Requirements

Make sure you have the following Python libraries installed:

```bash
pip install numpy pandas matplotlib
```

## Notes

- **Manual PCA**: No external PCA library (like `sklearn.decomposition.PCA`) is used.
- **Standardization** is performed manually instead of using `StandardScaler`.
- PCA is based purely on **linear algebra** concepts here.

## Code:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the Red Wine dataset
df = pd.read_csv("/content/drive/MyDrive/winequality-red.csv", sep=';')

# Extract features (drop 'quality')
X = df.drop(columns=["quality"]).values

# Standardize the dataset manually
X_standardized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Covariance matrix
cov_matrix = np.cov(X_standardized, rowvar=False)

# Eigen decomposition
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Sort eigenvalues and eigenvectors (descending order)
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues_sorted = eigenvalues[sorted_indices]
eigenvectors_sorted = eigenvectors[:, sorted_indices]

# Compute explained variance ratio
explained_variance = eigenvalues_sorted / np.sum(eigenvalues_sorted)

# Select top 2 eigenvectors for 2D transformation
top_2_eigenvectors = eigenvectors_sorted[:, :2]
X_transformed = np.dot(X_standardized, top_2_eigenvectors)

# Scree Plot (All Components)
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance')
plt.title('Scree Plot (Manual PCA) - Red Wine Dataset')
plt.grid(True)
plt.show()

# 2D PCA Projection
plt.figure(figsize=(8, 6))
plt.scatter(X_transformed[:, 0], X_transformed[:, 1], alpha=0.6)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2D PCA Projection (Manual) - Red Wine Dataset')
plt.grid(True)
plt.show()
```
## Output:
![download (16)](https://github.com/user-attachments/assets/9ce131b4-c02e-4c7d-a16f-a7e4add7f266)
![download (17)](https://github.com/user-attachments/assets/55a9ef2b-9675-4d75-be59-bf22001459e2)

