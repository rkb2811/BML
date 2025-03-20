# APPLY DATA PREPROCESSING TECHNIQUES TO MAKE DATA SUITABLE FOR MACHINE LEARNING.
## Introduction

Data preprocessing is a crucial step in machine learning to ensure that the dataset is clean, structured, and suitable for model training. This process helps improve the accuracy and efficiency of machine learning algorithms by handling missing values, removing noise, and transforming data into a usable format.

## Steps in Data Preprocessing

### 1. Importing Libraries

To begin with, import the necessary libraries for handling data.

### 2. Loading the Dataset

### 3. Handling Missing Values

Remove missing values:

Fill missing values with mean/median/mode:

### 4. Encoding Categorical Data

Label Encoding (for categorical variables with few categories):

One-Hot Encoding (for categorical variables with multiple categories):

### 5. Feature Scaling

Standardization (values have mean 0 and variance 1):

Normalization (scales values between 0 and 1):

### 6. Splitting Data into Training and Testing Sets

## Conclusion

Data preprocessing enhances the quality of the dataset, ensuring better performance of machine learning models. Proper handling of missing values, encoding categorical data, and feature scaling are essential steps for preparing data before model training.

## Requirements

Python 3.x

Pandas

NumPy

Scikit-learn

## Usage

Place your dataset in the project folder.

Modify the script to match your dataset columns.

Run the script to preprocess your data before training machine learning models.

## Code Implementation 
```
import pandas as pd
import matplotlib.pyplot as plt

# Load the Iris dataset
iris_data = pd.read_csv('/content/drive/MyDrive/iris.data')

# Display the first 5 rows of the dataset
print("First 5 rows of the dataset:")
print(iris_data.head())

# Get dataset information (columns, non-null counts, datatypes, memory usage)
print("\nDataset Information:")
print(iris_data.info())

# Get the total number of elements in the DataFrame
print("\nTotal number of elements in the dataset:")
print(iris_data.size)

# Generate descriptive statistics for numeric columns
print("\nDescriptive statistics:")
print(iris_data.describe())

# Get the shape of the dataset (rows, columns)
print("\nShape of the dataset:")
print(iris_data.shape)

# Check for missing values in each column
print("\nMissing values in each column:")
print(iris_data.isnull().sum())

# Check if there are any missing values in the entire dataset
print("\nAre there any missing values in the dataset?")
print(iris_data.isnull().any().any())

# Display rows with missing values, if any
missing_rows = iris_data[iris_data.isnull().any(axis=1)]
if missing_rows.empty:
    print("\nNo rows contain missing values.")
else:
    print("\nRows with missing values:")
    print(missing_rows)

# Optionally, fill missing values with 0
iris_data.fillna(value=0, inplace=True)
print("\nMissing values have been filled with 0.")

# Verify again after filling missing values
print("\nMissing values check after filling:")
print(iris_data.isnull().sum())

# Assign column names if they are missing
iris_data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

# Display the first few rows
print(iris_data.head())

plt.figure(figsize=(8, 6))
iris_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].boxplot()
plt.title('Box Plot of Iris Features')
plt.ylabel('Measurement (cm)')
plt.grid(True)
plt.show()
