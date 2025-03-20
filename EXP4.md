# Naive Bayes Classifier

## Introduction
The Naive Bayes classifier is a probabilistic machine learning model based on Bayes' theorem. It is commonly used for classification tasks such as spam filtering, sentiment analysis, and document classification.

## Prerequisites
Before running the Naive Bayes classifier, ensure you have the following installed:
- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

You can install the required packages using:
```bash
pip install numpy pandas scikit-learn matplotlib
```

## Dataset
The classifier works on a dataset with categorical or numerical features. You can use any dataset, such as the Iris dataset or a custom dataset.

## Implementation Steps
1. **Import Required Libraries**
2. **Load the Dataset**
3. **Preprocess the Data**
4. **Split Data into Training and Testing Sets**
5. **Train the Naïve Bayes Model**
6. **Make Predictions**
7. **Evaluate the Model**
8. **Visualize the Results**

## Code Implementation
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Dataset (Example: Iris Dataset)
from sklearn.datasets import load_iris
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Splitting Data
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Naïve Bayes Model
model = GaussianNB()
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Evaluate Model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Plot Confusion Matrix
import seaborn as sns
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
```

## Output Example
```
Accuracy: 0.96
Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        10
           1       0.92      1.00      0.96        12
           2       1.00      0.88      0.93         8

    accuracy                           0.96        30
   macro avg       0.97      0.96      0.96        30
weighted avg       0.96      0.96      0.96        30

Confusion Matrix:
[[10  0  0]
 [ 0 12  0]
 [ 0  1  7]]
```

## Applications
- Spam email classification
- Sentiment analysis
- Medical diagnosis
- Document classification

## Usage
To use this classifier on a different dataset, modify the `load_iris()` section with your dataset and preprocess it accordingly.

---
