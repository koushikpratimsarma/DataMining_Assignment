# 📊 Data Mining Assignments (Machine Learning)

This repository contains practical machine learning assignments implemented using Python and the scikit-learn library.

---

## 📌 1. Digit Recognition using Logistic Regression

* **Dataset:** MNIST (from OpenML)
* **Algorithm:** Logistic Regression

**Description:**
This program classifies handwritten digits (0–9) using Logistic Regression. The dataset is normalized, split into training and testing sets, and evaluated using accuracy.

---

## 📌 2. Digit Recognition using Naive Bayes

* **Dataset:** MNIST (from OpenML)
* **Algorithm:** Gaussian Naive Bayes

**Description:**
This program uses a probabilistic approach to classify handwritten digits. It assumes that features follow a Gaussian (normal) distribution and predicts digits based on probability.

---

## 📌 3. Iris Classification using KNN

* **Dataset:** Iris Dataset
* **Algorithm:** K-Nearest Neighbors (KNN)

**Description:**
This program classifies iris flowers into three categories: Setosa, Versicolor, and Virginica using KNN based on sepal and petal measurements.

**What has been done:**

* Loaded the Iris dataset from scikit-learn
* Split the dataset into training and testing sets
* Applied KNN algorithm for classification
* Evaluated the model using accuracy score

---

## 📌 4. Cancer Data Classification

* **Dataset:** Breast Cancer Dataset (from scikit-learn)
* **Algorithm:** Logistic Regression / Classification Model

**Description:**
This program classifies tumors as malignant or benign using machine learning techniques based on various medical features.

**What has been done:**

* Loaded the breast cancer dataset
* Preprocessed and normalized the data
* Split the dataset into training and testing sets
* Trained the model using classification algorithm
* Evaluated performance using accuracy score

---

## ⚙️ Technologies Used

* Python
* NumPy
* Matplotlib
* scikit-learn
* seaborn
---

## ▶️ How to Run

1. Install required libraries:

   ```
   pip install numpy matplotlib scikit-learn seaborn
   ```

2. Navigate to any assignment folder and run:

   ```
   python main.py
   ```

---

## 📊 Results

* Logistic Regression (MNIST): ~90–95% accuracy
* Naive Bayes (MNIST): ~80–85% accuracy
* KNN (Iris): ~95–100% accuracy

---

## 📚 Dataset Sources

* MNIST dataset from OpenML
* Iris dataset from scikit-learn

---

## 📁 Project Structure

```
DataMining_Assignment/
│
├── Digit-Recognition-Logistic/
├── Digit-Recognition-NaiveBayes/
├── Iris-KNN/
└── cancer_data/
```

---

## 👨‍💻 Author

*Koushik Pratim Sarma
