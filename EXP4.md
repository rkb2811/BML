# Naive Bayes Classifier

## Introduction
The Naive Bayes classifier is a probabilistic machine learning model based on Bayes' theorem. It is commonly used for classification tasks such as spam filtering, sentiment analysis, and document classification.

## Requirements
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
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

# Generate sample dataset
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Naive Bayes classifier
model = GaussianNB()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Print accuracy
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
```

## Output Example
```
Accuracy: 0.81
```

## Applications
- Spam email classification
- Sentiment analysis
- Medical diagnosis
- Document classification

## Usage
To use this classifier on a different dataset, modify the `load_iris()` section with your dataset and preprocess it accordingly.

---
