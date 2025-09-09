# Loan Approval Prediction

This project predicts whether a **loan application will be approved or not** using machine learning. It is built during my internship as part of a practical machine learning task.

## Project Description

Banks and financial institutions receive many loan applications every day. To reduce manual effort and make faster decisions, we can use machine learning models to predict loan approvals based on applicant details like income, education, employment, credit history, etc.

In this project, I:

* Cleaned and preprocessed the dataset.
* Handled missing values.
* Encoded categorical features into numeric format.
* Trained classification models (Logistic Regression & Decision Tree).
* Dealt with **imbalanced data** using **SMOTE**.
* Evaluated performance using **precision, recall, and F1-score**.

---

## Dataset

I used the [Loan Approval Prediction Dataset](https://www.kaggle.com/datasets) (Kaggle).

* Features include: Applicant Income, Coapplicant Income, Loan Amount, Loan Term, Education, Property Area, Credit History, etc.
* Target column: **Loan\_Status** (`1 = Approved, 0 = Not Approved`)

---

## Steps in the Project

### 1. Data Preprocessing

* Filled missing values:

  * **Numerical columns** → median.
  * **Categorical columns** → mode.
* Converted categorical columns into numbers using:

  * **Label Encoding** (Gender, Married, Education, Self\_Employed, Loan\_Status).
  * **One-Hot Encoding** (Property\_Area).

### 2. Train-Test Split

* Split dataset into **training (80%)** and **testing (20%)** sets.

### 3. Feature Scaling

* Applied `StandardScaler` to normalize numerical values.

### 4. Model Training

* Trained models:

  * Logistic Regression
  * Decision Tree Classifier

### 5. Handling Imbalanced Data

* Observed that the dataset was **imbalanced** (more approved loans than not approved).
* Applied **SMOTE (Synthetic Minority Oversampling Technique)** to balance the data.
* Retrained the models on balanced data.

### 6. Model Evaluation

* Compared performance using:

  * Accuracy
  * Precision
  * Recall
  * F1-score
* Focused on **precision, recall, and F1-score**, since accuracy alone is not reliable for imbalanced data.

---

## Results

* Before SMOTE: Model was biased towards the majority class.
* After SMOTE: Better balance between predicting approvals and rejections.
* Decision Tree and Logistic Regression gave different performance — I compared both.

---

## Tools & Libraries

* Python
* Pandas
* NumPy
* Scikit-learn
* imbalanced-learn (for SMOTE)

---

## How to Run

1. Clone this repo:

   ```bash
   git clone https://github.com/MorsCodee/Loan_Approval_Prediction.git
   cd Loan_Approval_Prediction
   ```
2. Install required libraries:

   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook/script to train the model and see results.

---

## Future Improvements

* Try other models (Random Forest, XGBoost).
* Hyperparameter tuning for better accuracy.
* Deploy the model as a web app using Flask/Streamlit.

---

## Acknowledgements

* Dataset: Kaggle
* Libraries: Scikit-learn, Pandas, NumPy, imbalanced-learn
