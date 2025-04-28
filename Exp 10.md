# Wine Quality Prediction using Random Forest Classifier

This project predicts the quality of red wine based on various physicochemical features using a **Random Forest Classifier**.

## Dataset
The dataset used is the **Wine Quality - Red Wine** dataset, which can be found [here](https://archive.ics.uci.edu/ml/datasets/wine+quality) or was loaded from:
```
/content/drive/MyDrive/winequality-red.csv
```
The dataset is separated by a semicolon (`;`).

**Target variable:** `quality`  
**Feature variables:** Fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol.

## Requirements
- Python 3.x
- pandas
- scikit-learn

Install the dependencies using:
```bash
pip install pandas scikit-learn
```

## Project Workflow

1. **Load the Dataset:**  
   Read the CSV file with the appropriate separator (`;`).

2. **Prepare Features and Target:**
   - `X` → All columns except `quality`
   - `y` → Only the `quality` column

3. **Split the Dataset:**
   - 80% training data
   - 20% testing data
   - Random state = 42 (for reproducibility)

4. **Train the Model:**
   - Using `RandomForestClassifier` with 100 trees (`n_estimators=100`).

5. **Make Predictions:**
   - Predict the wine quality on the test set.

6. **Evaluate the Model:**
   - Accuracy Score
   - Classification Report (Precision, Recall, F1-score)
   - Confusion Matrix

## Code Snippet
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv("/content/drive/MyDrive/winequality-red.csv", sep=';')

# Prepare features and target
X = df.drop("quality", axis=1)
y = df["quality"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
```

## Results
The model outputs:
- Overall **accuracy** on the test set.
- Detailed **classification metrics** for each class (wine quality score).
- **Confusion matrix** to analyze prediction errors.

## Notes
- The dataset is slightly imbalanced; some quality scores have fewer samples than others.
- Random Forests help to handle feature importance and provide robustness against overfitting.

