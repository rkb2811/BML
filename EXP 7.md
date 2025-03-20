# K-Nearest Neighbors (KNN) Classifier 

## Overview
This project implements a K-Nearest Neighbors (KNN) classifier using Python. KNN is a simple, non-parametric algorithm used for classification and regression tasks. It classifies a data point based on the majority class of its nearest neighbors.

## Features
- Implementation using `scikit-learn`
- Training and testing a KNN classifier
- Performance evaluation using accuracy
- Adjustable `k` value for tuning

## Requirements
Ensure you have the following Python libraries installed:

```sh
pip install numpy pandas scikit-learn matplotlib
```

## Usage
### 1. Clone the Repository
```sh
git clone https://github.com/yourusername/knn-classifier-python.git
cd knn-classifier-python
```

### 2. Run the Script
Execute the Python script to train and test the KNN model:

```sh
python knn_classifier.py
```

## Example Code
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('data.csv')  # Replace with your dataset
X = df.drop(columns=['target'])
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train KNN model
k = 3  # Set the number of neighbors
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Predictions
y_pred = knn.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```
### Output:
```
KNN model Accuracy: 100.000000%

```


## License
This project is licensed under the MIT License.


