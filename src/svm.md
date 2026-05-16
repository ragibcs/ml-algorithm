# সাপোর্ট ভেক্টর মেশিন (SVM) অ্যালগরিদম

## ভূমিকা

Support Vector Machine (SVM) হলো একটি জনপ্রিয় Supervised Machine Learning Algorithm যা মূলত Classification সমস্যার জন্য ব্যবহার করা হয়। তবে এটি Regression এবং Outlier Detection কাজেও ব্যবহার করা যায়।

SVM এমন একটি Decision Boundary তৈরি করে যা বিভিন্ন Class-কে সবচেয়ে ভালোভাবে আলাদা করতে পারে।

এটি Machine Learning-এর সবচেয়ে শক্তিশালী এবং কার্যকর অ্যালগরিদমগুলোর একটি।

---

## সূচিপত্র

1. [SVM কী?](#svm-কী)
2. [SVM কীভাবে কাজ করে](#svm-কীভাবে-কাজ-করে)
3. [Hyperplane](#hyperplane)
4. [Support Vector](#support-vector)
5. [Margin](#margin)
6. [SVM-এর প্রকারভেদ](#svm-এর-প্রকারভেদ)
7. [Kernel Trick](#kernel-trick)
8. [Mathematical Formula](#mathematical-formula)
9. [SVM-এর সুবিধা](#svm-এর-সুবিধা)
10. [SVM-এর অসুবিধা](#svm-এর-অসুবিধা)
11. [SVM-এর ব্যবহার](#svm-এর-ব্যবহার)
12. [Python Implementation](#python-implementation)
13. [Workflow](#workflow)
14. [SVM-এর Parameter](#svm-এর-parameter)
15. [Linear vs Non-linear SVM](#linear-vs-non-linear-svm)
16. [SVM বনাম Logistic Regression](#svm-বনাম-logistic-regression)
17. [উপসংহার](#উপসংহার)

---

## 1. SVM কী?

Support Vector Machine (SVM) হলো একটি Supervised Learning Algorithm যা Dataset-এর বিভিন্ন Class-কে আলাদা করার জন্য একটি Optimal Hyperplane তৈরি করে।

SVM-এর মূল উদ্দেশ্য হলো:

> এমন একটি Boundary তৈরি করা যাতে দুইটি Class-এর মধ্যে দূরত্ব (Margin) সর্বাধিক হয়।

### উদাহরণ

ধরি দুই ধরনের Data আছে:

* Cat
* Dog

SVM এমন একটি Line বা Boundary বের করবে যা Cat এবং Dog-কে আলাদা করবে।

---

## 2. SVM কীভাবে কাজ করে

SVM Data Point-গুলোর মধ্যে এমন একটি Boundary তৈরি করে যেটি দুইটি Class-কে আলাদা করে।

এটি শুধু Boundary তৈরি করে না, বরং এমন Boundary তৈরি করে যার Margin সবচেয়ে বেশি।

### মূল ধারণা

* Hyperplane তৈরি করা
* Margin সর্বাধিক করা
* Support Vector ব্যবহার করা

---

## 3. Hyperplane

Hyperplane হলো একটি Decision Boundary যা বিভিন্ন Class-কে আলাদা করে।

### বিভিন্ন Dimension-এ Hyperplane

| Dimension        | Hyperplane |
| ---------------- | ---------- |
| 2D               | Line       |
| 3D               | Plane      |
| Higher Dimension | Hyperplane |

### Hyperplane Equation

```math
w^Tx + b = 0
```

যেখানে:

* `w` = Weight Vector
* `x` = Feature Vector
* `b` = Bias

---

## 4. Support Vector

Support Vector হলো সেই Data Point যেগুলো Hyperplane-এর সবচেয়ে কাছে থাকে।

এই Point-গুলো খুব গুরুত্বপূর্ণ কারণ:

* এগুলো Hyperplane নির্ধারণ করে
* এগুলো পরিবর্তন হলে Model পরিবর্তিত হয়

---

## 5. Margin

Margin হলো Hyperplane এবং কাছের Data Point-এর মধ্যকার দূরত্ব।

SVM সর্বদা Margin সর্বাধিক করার চেষ্টা করে।

### Margin Formula

```math
Margin = \frac{2}{||w||}
```

---

## 6. SVM-এর প্রকারভেদ

### A. Linear SVM

যখন Data সহজে Straight Line দিয়ে আলাদা করা যায় তখন Linear SVM ব্যবহার করা হয়।

#### উদাহরণ

* Spam Detection
* Sentiment Analysis
* Binary Classification

### B. Non-linear SVM

যখন Data Straight Line দিয়ে আলাদা করা যায় না তখন Non-linear SVM ব্যবহার করা হয়।

এক্ষেত্রে Kernel ব্যবহার করা হয়।

---

## 7. Kernel Trick

Kernel হলো এমন একটি Technique যা Data-কে Higher Dimension-এ নিয়ে যায় যাতে Data সহজে আলাদা করা যায়।

### জনপ্রিয় Kernel Function

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

SVM-এর মূল Optimization Objective:

```math
\min \frac{1}{2}||w||^2
```

Subject to:

```math
y_i(w^Tx_i+b)\geq1
```

যেখানে:

* `y_i` = Class Label
* `x_i` = Training Sample

---

## 9. SVM-এর সুবিধা

### 1. High Accuracy

উচ্চ Accuracy প্রদান করে।

### 2. Small Dataset-এ ভালো কাজ করে

কম Data থাকলেও কার্যকর।

### 3. Overfitting কম হয়

বিশেষ করে High-dimensional Data-এ।

### 4. Non-linear Data Handle করতে পারে

Kernel Trick ব্যবহার করে।

---

## 10. SVM-এর অসুবিধা

### 1. Large Dataset-এ Slow

Dataset বড় হলে Training Time বেড়ে যায়।

### 2. Parameter Tuning কঠিন

যেমন:

* C
* Gamma
* Kernel

ঠিকভাবে Tune করতে হয়।

### 3. কম Interpretable

Decision Tree-এর তুলনায় বোঝা কঠিন।

---

## 11. SVM-এর ব্যবহার

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

OCR System-এ ব্যবহার করা হয়।

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
Dataset সংগ্রহ
        ↓
Data Preprocessing
        ↓
Feature Selection
        ↓
Kernel নির্বাচন
        ↓
Model Training
        ↓
Prediction
        ↓
Evaluation
```

### বাস্তব জীবনের উদাহরণ

ধরি একটি Email Spam Detection System তৈরি করতে হবে।

SVM:

* Email-এর Feature বিশ্লেষণ করবে
* Spam এবং Non-spam আলাদা করবে
* একটি Optimal Boundary তৈরি করবে

ফলে নতুন Email Spam কিনা তা Predict করতে পারবে।

---

## 14. SVM-এর গুরুত্বপূর্ণ Parameter

### C Parameter

* Error কমাতে সাহায্য করে
* Large C → কম Error
* Small C → বড় Margin

### Gamma Parameter

RBF Kernel-এর জন্য ব্যবহৃত হয়।

* Large Gamma → Overfitting হতে পারে
* Small Gamma → Underfitting হতে পারে

---

## 15. Linear vs Non-linear SVM

| Feature    | Linear SVM         | Non-linear SVM |
| ---------- | ------------------ | -------------- |
| Data Type  | Linearly Separable | Complex Data   |
| Speed      | Fast               | Slower         |
| Kernel     | Linear             | RBF/Polynomial |
| Complexity | Low                | High           |

---

## 16. SVM বনাম Logistic Regression

| বিষয়            | SVM            | Logistic Regression |
| --------------- | -------------- | ------------------- |
| Boundary        | Maximum Margin | Probability Based   |
| Small Dataset   | Excellent      | Good                |
| Large Dataset   | Slow           | Fast                |
| Non-linear Data | Excellent      | Limited             |

---

## 17. উপসংহার

Support Vector Machine (SVM) হলো একটি শক্তিশালী Machine Learning Algorithm যা Classification সমস্যায় অত্যন্ত কার্যকর।

এটি:

* Maximum Margin তৈরি করে
* Complex Data Handle করতে পারে
* Kernel Trick ব্যবহার করে Non-linear Problem সমাধান করতে পারে

### বর্তমান ব্যবহার

* Image Processing
* NLP
* Medical AI
* Cyber Security
* Spam Detection

সহ বিভিন্ন ক্ষেত্রে SVM ব্যাপকভাবে ব্যবহৃত হচ্ছে।
