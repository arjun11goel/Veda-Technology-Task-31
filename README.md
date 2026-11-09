# Task 31: Features and Target Selection

## Description
This project demonstrates the process of selecting suitable input features and a target variable from a tabular dataset for supervised machine learning tasks using the Iris Dataset.

---

## Objective
Understand the core structure of supervised machine learning problems by distinguishing between feature variables (X) and target variables (y).

---

## Tools Used
- **Python**
- **Pandas**
- **Jupyter Notebook**

---

## Dataset Overview
The **Iris Dataset** (`iris_dataset.csv`) consists of 150 instances of iris flowers spanning 3 species (*setosa*, *versicolor*, *virginica*), with four physical measurements for each sample:
- `sepal length (cm)`
- `sepal width (cm)`
- `petal length (cm)`
- `petal width (cm)`
- `species` (Target class)

---

## Code Implementation

```python
import pandas as pd

# 1. Load Dataset
df = pd.read_csv('iris_dataset.csv')

# Display first 5 rows
print("--- First 5 Rows of the Iris Dataset ---")
print(df.head(), "\n")

# 2. Select Features (X) and Target (y)
feature_columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
X = df[feature_columns]

target_column = 'species'
y = df[target_column]

# 3. Print Results
print("--- Selected Feature Columns (X) ---")
print(X.head(3), "\n")

print("--- Selected Target Column (y) ---")
print(y.head(3), "\n")

print(f"Shape of Features (X): {X.shape}")
print(f"Shape of Target (y): {y.shape}")
```

---

## Execution Output

```text
--- First 5 Rows of the Iris Dataset ---
   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  \
0                5.1               3.5                1.4               0.2   
1                4.9               3.0                1.4               0.2   
2                4.7               3.2                1.3               0.2   
3                4.6               3.1                1.5               0.2   
4                5.0               3.6                1.4               0.2   

  species  
0  setosa  
1  setosa  
2  setosa  
3  setosa  
4  setosa  


--- Selected Feature Columns (X) ---
   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
0                5.1               3.5                1.4               0.2
1                4.9               3.0                1.4               0.2
2                4.7               3.2                1.3               0.2

--- Selected Target Column (y) ---
0    setosa
1    setosa
2    setosa
Name: species, dtype: str

Shape of Features (X): (150, 4)
Shape of Target (y): (150,)
```

---

## Selection Explanation & Rationale

1. **Feature Matrix (X):**
   - **Columns Selected:** `sepal length (cm)`, `sepal width (cm)`, `petal length (cm)`, `petal width (cm)`
   - **Reasoning:** These continuous numerical dimensions represent the physical characteristics of the flowers. They serve as independent predictor variables that the model uses to learn structural patterns.

2. **Target Vector (y):**
   - **Column Selected:** `species`
   - **Reasoning:** The species label represents the outcome category to predict. Because the objective is to classify an unknown flower into its species based on physical measurements, `species` is designated as the target variable.

3. **Preventing Data Leakage:**
   - The target variable `species` is explicitly separated from matrix X to prevent data leakage during model training.

---

## Interview Questions & Answers

### Q1: What is a feature?
**Answer:**  
A feature is an individual, measurable property or characteristic of a phenomenon being observed. In machine learning, features serve as the input variables (X) provided to a model to make predictions. In tabular datasets, features correspond to dataset columns containing input data.

### Q2: What is a target variable?
**Answer:**  
The target variable (also referred to as the label, response variable, or dependent variable) is the specific property or outcome that the machine learning algorithm aims to predict (y). For classification tasks, the target is categorical; for regression tasks, it is continuous.

### Q3: What is the difference between X and y in machine learning?
**Answer:**  
- **X (Capital letter):** Represents a 2D matrix or Dataframe of input features with shape `(n_samples, n_features)`. Uppercase notation is used because it represents a multi-column matrix.
- **y (Lowercase letter):** Represents a 1D vector or Series of target labels with shape `(n_samples,)`. Lowercase notation is used because it represents a single variable array.
