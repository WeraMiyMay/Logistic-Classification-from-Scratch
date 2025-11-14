<div align="center">

# 🔵 Logistic Classification from Scratch  
### *Gradient Descent · Log-Loss Optimization · Decision Boundary Visualization*

</div>

## 📌 Overview
This project implements a **binary linear classifier** trained entirely **from scratch**, without using machine learning libraries such as scikit-learn.  
The goal of this work is to demonstrate how a classic machine-learning model can be implemented manually, including:

- formulation of the loss function  
- computation of gradients  
- optimization via gradient descent  
- understanding the decision boundary  
- visualization of classification results  

This makes the project especially useful for students and developers who want deeper insight into **how classification models learn** at the mathematical and algorithmic level.

---

## 🎯 The Essence of the Model Training

### ### 🔍 What the Model Tries to Learn
We have a dataset consisting of two numerical features and binary class labels  
**+1** (blue class) and **–1** (red class).

A linear classifier must learn a function:

\[
f(x) = w_0 + w_1 x_1 + w_2 x_2
\]

The model predicts class sign:

- if **f(x) > 0 → class +1**  
- if **f(x) < 0 → class -1**

The weights \( w_0, w_1, w_2 \) define the **decision boundary**, which is a straight line dividing the classes.

---

### 🧠 Why Log-Loss?
We optimize the **logarithmic loss**:

\[
\log(1 + e^{-y \cdot (w^T x)})
\]

It penalizes the model stronger for confident wrong predictions.

Its key properties:

- smooth and differentiable  
- stable numerically  
- directly tied to probabilistic interpretation  
- used in logistic regression, boosting, and deep learning  

---

### ⚙️ How The Model Learns

Training is performed using **gradient descent**:

1. Compute the margin:  
   \[
   m_i = y_i \cdot (w^T x_i)
   \]

2. Compute log-loss gradient  
3. Update weights in the opposite direction of the gradient  
4. Repeat for 5000 iterations  

As a result, weights gradually shift in a direction that:

- increases the margin for correctly predicted samples  
- decreases error on misclassified points  
- finds the best separating line  

---

### 📈 What the Final Model Represents
After training, the obtained weights define a **linear decision boundary**:

\[
x_2 = -\frac{w_0 + w_1 x_1}{w_2}
\]

This boundary is plotted on the graph along with the dataset:

- Blue points = class +1  
- Red points = class –1  
- KDE contours show density regions  
- The line shows how the classifier separates classes  

This gives an intuitive and visual understanding of how the model behaves.

---

## 🧠 Requirements
- Python 3.x  
- NumPy  
- Matplotlib  
- Seaborn  

Install dependencies:
```bash
pip install numpy matplotlib seaborn


