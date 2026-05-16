# Support Vector Machine (SVM) Algorithm

## Introduction

Support Vector Machine (SVM) is a popular Supervised Machine Learning Algorithm primarily used for Classification problems. It can also be used for Regression and Outlier Detection.

SVM creates a Decision Boundary that can best separate different Classes.

It is one of the most powerful and effective algorithms in Machine Learning.

---

## Table of Contents

1. [What is SVM?](#what-is-svm)
2. [How SVM Works](#how-svm-works)
3. [Hyperplane](#hyperplane)
4. [Support Vector](#support-vector)
5. [Margin](#margin)
6. [Types of SVM](#types-of-svm)
7. [Kernel Trick](#kernel-trick)
8. [Mathematical Formula](#mathematical-formula)
9. [Advantages of SVM](#advantages-of-svm)
10. [Disadvantages of SVM](#disadvantages-of-svm)
11. [Applications of SVM](#applications-of-svm)
12. [Python Implementation](#python-implementation)
13. [Workflow](#workflow)
14. [SVM Parameters](#svm-parameters)
15. [Linear vs Non-linear SVM](#linear-vs-non-linear-svm)
16. [SVM vs Logistic Regression](#svm-vs-logistic-regression)
17. [Conclusion](#conclusion)

---

## 1. What is SVM?

Support Vector Machine (SVM) is a Supervised Learning Algorithm that creates an Optimal Hyperplane to separate different classes in a dataset.

The main objective of SVM is:

> To create a Boundary that maximizes the distance (Margin) between two classes.

### Example

Suppose we have two types of data:

* Cat
* Dog

SVM will find a Line or Boundary that separates Cat and Dog.

---

## 2. How SVM Works

SVM creates a Boundary between data points that separates two classes.

It doesn't just create any Boundary — it creates the Boundary with the maximum Margin.

### Core Concepts

* Create a Hyperplane
* Maximize the Margin
* Use Support Vectors

---

## 3. Hyperplane

A Hyperplane is a Decision Boundary that separates different classes.

### Hyperplane in Different Dimensions

| Dimension        | Hyperplane |
| ---------------- | ---------- |
| 2D               | Line       |
| 3D               | Plane      |
| Higher Dimension | Hyperplane |

### Hyperplane Equation

```math
w^Tx + b = 0
```

Where:

* `w` = Weight Vector
* `x` = Feature Vector
* `b` = Bias

---

## 4. Support Vector

Support Vectors are the data points closest to the Hyperplane.

These points are very important because:

* They determine the Hyperplane
* If they change, the Model changes

---

## 5. Margin

Margin is the distance between the Hyperplane and the nearest data point.

SVM always tries to maximize the Margin.

### Margin Formula

```math
Margin = \frac{2}{||w||}
```

---

## 6. Types of SVM

### A. Linear SVM

When data can be easily separated by a Straight Line, Linear SVM is used.

#### Examples

* Spam Detection
* Sentiment Analysis
* Binary Classification

### B. Non-linear SVM

When data cannot be separated by a Straight Line, Non-linear SVM is used.

In this case, Kernels are used.

---

## 7. Kernel Trick

Kernel is a Technique that takes data to a Higher Dimension so that data can be easily separated.

### Popular Kernel Functions

#### 1. Linear Kernel

```math
K(x_i,x_j)=x_i^Tx_j
```

#### 2. Polynomial Kernel

```math
K(x_i,x_j)=(x_i^Tx_j+c)^d
```

#### 3. RBF Kernel

```math
K(x_i,x_j)=exp(-\gamma ||x_i-x_j||^2)
```

#### 4. Sigmoid Kernel

```math
K(x_i,x_j)=tanh(\alpha x_i^Tx_j+c)
```

---

## 8. Mathematical Formula

SVM's core Optimization Objective:

```math
\min \frac{1}{2}||w||^2
```

Subject to:

```math
y_i(w^Tx_i+b)\geq1
```

Where:

* `y_i` = Class Label
* `x_i` = Training Sample

---

## 9. Advantages of SVM

### 1. High Accuracy

Provides high accuracy.

### 2. Works Well with Small Dataset

Effective even with less data.

### 3. Less Overfitting

Especially with high-dimensional data.

### 4. Can Handle Non-linear Data

Using Kernel Trick.

---

## 10. Disadvantages of SVM

### 1. Slow with Large Dataset

Training time increases with large datasets.

### 2. Difficult Parameter Tuning

Such as:

* C
* Gamma
* Kernel

Must be tuned properly.

### 3. Less Interpretable

Harder to understand compared to Decision Tree.

---

## 11. Applications of SVM

### Image Classification

* Face Recognition
* Object Detection

### Text Classification

* Spam Detection
* Sentiment Analysis

### Medical Diagnosis

* Cancer Detection
* Disease Prediction

### Bioinformatics

* Protein Classification
* Gene Analysis

### Handwriting Recognition

Used in OCR Systems.

---

## 12. Python Implementation

### Library Import

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
```

### Dataset Load

```python
iris = datasets.load_iris()
X = iris.data
y = iris.target
```

### Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### Model Training

```python
model = SVC(kernel='linear')
model.fit(X_train, y_train)
```

### Prediction

```python
y_pred = model.predict(X_test)
```

### Accuracy Check

```python
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```

---

## 13. Workflow

```text
Collect Dataset
      ↓
Data Preprocessing
      ↓
Feature Selection
      ↓
Select Kernel
      ↓
Model Training
      ↓
Prediction
      ↓
Evaluation
```

### Real-life Example

Suppose we need to create an Email Spam Detection System.

SVM:

* Analyzes Email features
* Separates Spam and Non-spam
* Creates an Optimal Boundary

Thus it can predict whether a new Email is Spam or not.

---

## 14. SVM Parameters

### C Parameter

* Helps reduce error
* Large C → Less error
* Small C → Larger margin

### Gamma Parameter

Used for RBF Kernel.

* Large Gamma → May cause overfitting
* Small Gamma → May cause underfitting

---

## 15. Linear vs Non-linear SVM

| Feature    | Linear SVM         | Non-linear SVM |
| ---------- | ------------------ | -------------- |
| Data Type  | Linearly Separable | Complex Data   |
| Speed      | Fast               | Slower         |
| Kernel     | Linear             | RBF/Polynomial |
| Complexity | Low                | High           |

---

## 16. SVM vs Logistic Regression

| Topic            | SVM            | Logistic Regression |
| ---------------- | -------------- | ------------------- |
| Boundary         | Maximum Margin | Probability Based   |
| Small Dataset    | Excellent      | Good                |
| Large Dataset    | Slow           | Fast                |
| Non-linear Data  | Excellent      | Limited             |

---

## 17. Conclusion

Support Vector Machine (SVM) is a powerful Machine Learning Algorithm that is highly effective for Classification problems.

It:

* Creates Maximum Margin
* Can handle Complex Data
* Can solve Non-linear Problems using Kernel Trick

### Current Usage

* Image Processing
* NLP
* Medical AI
* Cyber Security
* Spam Detection

SVM is widely used in various fields today.
