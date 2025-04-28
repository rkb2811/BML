# Wine Quality Prediction using Gradient Boosting Classifier

This project predicts the quality of red wine based on various physicochemical features using a **Gradient Boosting Classifier**.

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
- matplotlib
- seaborn

Install the dependencies using:
```bash
pip install pandas scikit-learn matplotlib seaborn
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
   - Using `GradientBoostingClassifier` with 100 estimators and a learning rate of 0.1.

6. **Make Predictions:**
   - Predict the wine quality on the test set.

7. **Evaluate the Model:**
   - Accuracy Score
   - Classification Report (Precision, Recall, F1-score)
   - Confusion Matrix visualization (heatmap)

8. **Visualize Results:**
   - Plot the distribution of predicted wine quality scores.

## Code Snippet
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("/content/drive/MyDrive/winequality-red.csv", sep=';')

# Prepare features and target
X = df.drop("quality", axis=1)
y = df["quality"]

# Shift labels to start from 0
y = y - 3

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Gradient Boosting Classifier
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gb_model.fit(X_train, y_train)

# Make predictions
y_pred = gb_model.predict(X_test)

# Evaluate the model
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Bar Plot: Predicted Wine Quality Distribution
predicted_quality = pd.Series(y_pred)
quality_counts = predicted_quality.value_counts().sort_index()

plt.figure(figsize=(8,6))
sns.barplot(x=quality_counts.index, y=quality_counts.values, palette="viridis")
plt.xlabel('Predicted Wine Quality (After Shifting)')
plt.ylabel('Number of Wines')
plt.title('Distribution of Predicted Wine Quality')
plt.grid(axis='y')
plt.show()
```
## Output
Accuracy:  0.65

Classification Report:
               precision    recall  f1-score   support

           0       0.00      0.00      0.00         1
           1       0.50      0.10      0.17        10
           2       0.70      0.77      0.74       130
           3       0.63      0.65      0.64       132
           4       0.60      0.50      0.55        42
           5       0.00      0.00      0.00         5

    accuracy                           0.65       320
   macro avg       0.41      0.34      0.35       320
weighted avg       0.64      0.65      0.64       320


![download (18)](https://github.com/user-attachments/assets/0730122e-b0d5-4fea-906e-d268e9506a9b)

![download (19)](https://github.com/user-attachments/assets/9b8ec727-775d-413b-9567-547a58f4f3e9)

## Results
The model outputs:
- Overall **accuracy** on the test set.
- Detailed **classification metrics** for each shifted class label (wine quality score minus 3).
- **Confusion matrix** heatmap to visualize prediction errors.
- **Bar plot** showing the distribution of predicted wine quality.

## Notes
- Label shifting is necessary because some machine learning models work better when class labels start from 0.
- Gradient Boosting combines multiple weak learners (typically decision trees) to form a strong predictive model.
- The dataset is slightly imbalanced; some quality scores have fewer samples than others.
- Proper hyperparameter tuning can further boost performance.

