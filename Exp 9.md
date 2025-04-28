# PCA implementation using own dataset

## Overview
This project implements **Principal Component Analysis (PCA) from scratch** without using libraries like `sklearn`. PCA is a dimensionality reduction technique widely used in Machine Learning and Data Science.

## Steps Implemented
1. **Standardize the dataset** – Normalize features to have mean = 0 and standard deviation = 1.
2. **Compute covariance matrix** – Understand relationships between features.
3. **Find eigenvalues and eigenvectors** – Identify important features.
4. **Sort eigenvalues and select top components** – Reduce dimensions.
5. **Transform the data** – Project onto new axes.

# Principal Component Analysis (PCA) - Overview

## What is PCA?
Principal Component Analysis (PCA) is a **dimensionality reduction** technique used in Machine Learning and Data Science. It helps in simplifying complex datasets while retaining the most important information.

## Why Use PCA?
-  **Reduces Complexity** – Converts high-dimensional data into fewer dimensions.
-  **Identifies Important Features** – Finds patterns and relationships in data.
-  **Improves Efficiency** – Speeds up machine learning algorithms.
-  **Removes Noise** – Helps in filtering unnecessary variations in data.

## How PCA Works:
1. **Standardize the data** – Normalize features to have mean = 0 and variance = 1.
2. **Compute the covariance matrix** – Understand relationships between features.
3. **Find eigenvalues & eigenvectors** – Identify key patterns in the data.
4. **Sort & select top components** – Pick the most significant features.
5. **Transform the data** – Project data onto the new feature space.

## Applications of PCA:
 **Image Compression** – Reduces storage while preserving key details.  
 **Data Visualization** – Helps in plotting high-dimensional data in 2D/3D.  
 **Noise Filtering** – Removes irrelevant variations in data.  
 **Feature Selection** – Improves machine learning model performance.  

## Example:
Using PCA on a dataset with 100 features, we might reduce it to just **2 or 3** features while keeping most of the original information.

## code
```
import numpy as np
import pandas as pd

def pca(X, k):
    # Step 1: Standardize dataset
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
    # Step 2: Compute covariance matrix
    C = np.cov(X, rowvar=False)
    
    # Step 3: Compute eigenvalues and eigenvectors
    vals, vecs = np.linalg.eig(C)
    
    # Step 4: Sort eigenvalues in descending order and select top k eigenvectors
    idx = np.argsort(vals)[::-1]
    top_vecs = vecs[:, idx[:k]]
    
    # Step 5: Transform data using selected eigenvectors
    X_reduced = np.dot(X, top_vecs)
    
    return X_reduced, top_vecs

# Generate dataset
X = np.random.rand(10, 3)  

# Convert to DataFrame for better display
df = pd.DataFrame(X, columns=["Feature1", "Feature2", "Feature3"])
print("Original Dataset:")
print(df)

# Apply PCA
_, comps = pca(X, 2)

# Display only principal components
print("\nPrincipal Components:")
print(pd.DataFrame(comps, columns=["PC1", "PC2"]))
```

## Output
```
Original Dataset:
   Feature1  Feature2  Feature3
0  0.388677  0.271349  0.828738
1  0.356753  0.280935  0.542696
2  0.140924  0.802197  0.074551
3  0.986887  0.772245  0.198716
4  0.005522  0.815461  0.706857
5  0.729007  0.771270  0.074045
6  0.358466  0.115869  0.863103
7  0.623298  0.330898  0.063558
8  0.310982  0.325183  0.729606
9  0.637557  0.887213  0.472215

Principal Components:
        PC1       PC2
0 -0.514776 -0.735125
1 -0.540166  0.677677
2  0.665753 -0.018575
```
