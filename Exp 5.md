# Decision Tree 

## Overview
This project implements a Decision Tree classifier using Python. The Decision Tree algorithm is a supervised learning method used for classification and regression tasks. It works by recursively splitting the data into subsets based on the most significant attribute.

## Features
- Implementation using `scikit-learn`
- Training and testing a Decision Tree classifier
- Visualizing the Decision Tree
- Performance evaluation using accuracy and confusion matrix

## Requirements
Ensure you have the following Python libraries installed:

```sh
pip install numpy pandas scikit-learn matplotlib
```

## Usage

###  Run the Script
Execute the Python script to train and test the Decision Tree model:

```sh
python decision_tree.py
```

## Example Code
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Sample dataset
data = {
    'Feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Feature2': [5, 3, 8, 7, 6, 2, 4, 9, 1, 10],
    'Target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
}
df = pd.DataFrame(data)

# Splitting dataset
X = df.drop(columns=['Target'])
y = df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training Decision Tree model
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Evaluating accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Visualizing the Decision Tree
plt.figure(figsize=(8, 6))
plot_tree(clf, feature_names=X.columns, class_names=['Class 0', 'Class 1'], filled=True)
plt.show()

```
### Output:
```
Accuracy: 1.0
```



## License
This project is licensed under the MIT License.



