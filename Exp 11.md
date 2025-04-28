# Wine Quality Prediction using XGBoost Classifier

This project predicts the quality of red wine based on various physicochemical features using an **XGBoost Classifier**.

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
- xgboost

Install the dependencies using:
```bash
pip install pandas scikit-learn xgboost
```

## Project Workflow

1. **Load the Dataset:**  
   Read the CSV file with the appropriate separator (`;`).

2. **Prepare Features and Target:**
   - `X` → All columns except `quality`
   - `y` → Only the `quality` column

3. **Shift Labels:**
   - Subtract 3 from `y` so that the minimum label becomes 0 (because minimum quality score is 3).

4. **Split the Dataset:**
   - 80% training data
   - 20% testing data
   - Random state = 42 (for reproducibility)

5. **Train the Model:**
   - Using `XGBClassifier` with `use_label_encoder=False` and `eval_metric='mlogloss'`.

6. **Make Predictions:**
   - Predict the wine quality on the test set.

7. **Evaluate the Model:**
   - Accuracy Score
   - Classification Report (Precision, Recall, F1-score)
   - Confusion Matrix

## Code Snippet
```python
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv("/content/drive/MyDrive/winequality-red.csv", sep=';')

# Prepare features and target
X = df.drop("quality", axis=1)
y = df["quality"]

# Shift labels to start from 0
y = y - 3  # Since minimum quality is 3

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train XGBoost classifier
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb_model.fit(X_train, y_train)

# Make predictions
y_pred = xgb_model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
```
## Output 
Accuracy: 0.696875

Classification Report:
               precision    recall  f1-score   support

           0       0.00      0.00      0.00         1
           1       0.00      0.00      0.00        10
           2       0.75      0.80      0.78       130
           3       0.68      0.73      0.70       132
           4       0.64      0.55      0.59        42
           5       0.00      0.00      0.00         5

    accuracy                           0.70       320
   macro avg       0.34      0.35      0.34       320
weighted avg       0.67      0.70      0.68       320


Confusion Matrix:
 [[  0   0   1   0   0   0]
 [  0   0   7   3   0   0]
 [  0   1 104  24   1   0]
 [  0   1  25  96   9   1]
 [  0   0   1  17  23   1]
 [  0   0   0   2   3   0]]

## Results
The model outputs:
- Overall **accuracy** on the test set.
- Detailed **classification metrics** for each shifted class label (wine quality score minus 3).
- **Confusion matrix** to analyze prediction errors.

## Notes
- Label shifting is necessary because some machine learning models expect class labels to start from 0.
- XGBoost is powerful and often provides better performance compared to traditional classifiers.
- The dataset is slightly imbalanced; some quality scores have fewer samples than others.
- Hyperparameter tuning can further improve model performance.

