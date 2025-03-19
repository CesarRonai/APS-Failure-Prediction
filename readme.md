# 🚛 APS Failure Prediction - Machine Learning Project

## 📌 Project Overview

This project focuses on predicting failures in the **Air Pressure System (APS)** of heavy-duty **Scania Trucks** using **Machine Learning**. The dataset contains operational data from trucks, with labels indicating whether a failure is related to the APS (`pos`) or another component (`neg`).

The goal is to **accurately predict APS failures** while minimizing false negatives, as missing a real failure can lead to high operational costs.

## 🛠️ Technologies Used

- **Python**
- **Pandas, NumPy** (Data Manipulation)
- **Matplotlib, Seaborn** (Data Visualization)
- **Scikit-learn** (Machine Learning)
- **Imbalanced-learn (SMOTE)** (Handling Class Imbalance)
- **Random Forest Classifier** (Predictive Model)
- **Git & GitHub** (Version Control)

---

## 📊 Data Overview

### 📁 Dataset Information

- **Training Set**: 60,000 instances
- **Test Set**: 16,000 instances
- **Features**: 171 anonymized operational variables
- **Class Distribution**:
  - `pos` (APS failure) → **1.67%** (Highly Imbalanced)
  - `neg` (Other failures) → **98.33%**

### ⚠️ Challenge:

- **Imbalanced Data**: Severe class imbalance, requiring oversampling techniques like **SMOTE**.
- **High-dimensionality**: Requires feature selection and engineering.
- **Cost-sensitive Prediction**: Missing an APS failure (`false negative`) is 50x more expensive than a false positive.

---

## 🔎 Exploratory Data Analysis (EDA)

Before training the model, we conducted an in-depth **EDA**, including:
✅ **Missing values analysis** → Columns with >70% missing values removed.
✅ **Feature correlation analysis** → Redundant features identified.
✅ **Outlier detection & transformation** → Log transformation applied.
✅ **Class balance correction** → **SMOTE** used to handle imbalance.

---

## 🤖 Model Training & Evaluation

### **📌 Model Used: Random Forest**

A **Random Forest Classifier** was trained due to its robustness against outliers and ability to handle high-dimensional data.

### **🔢 Model Performance**

| Metric        | Class `neg` | Class `pos` |
| ------------- | ----------- | ----------- |
| **Precision** | **1.00**    | **0.97**    |
| **Recall**    | **0.99**    | **0.98**    |
| **F1-score**  | **1.00**    | **0.98**    |
| **Accuracy**  | **99%**     | -           |

### **Confusion Matrix**

```
[[11725    75]
 [   36  2324]]
```

✅ **High Recall (98%) for ************************`pos`************************ class** → Ensures most failures are detected.
✅ **Low False Negative Rate** → Reduces risks of missed APS failures.

---

## 🚀 How to Use the Model

### 🔹 **Installation**

Clone the repository and install dependencies:

```bash
git clone https://github.com/CesarRonai/APS-Failure-Prediction
cd APS-Failure-Prediction
pip install -r requirements.txt
```

### 🔹 **Run Model Training**

```bash
python src/train_model.py
```

### 🔹 **Make Predictions**

```python
import joblib
import pandas as pd

# Load trained model
model = joblib.load("models/modelo_random_forest.pkl")

# Load sample data
X_sample = pd.read_csv("data/sample_input.csv")

# Make prediction
prediction = model.predict(X_sample)
print("Predicted Class:", prediction)
```

---

## 🏗️ Project Structure

```
APS-Failure-Prediction/
│── data/                      # (Optional) Raw datasets
│── notebooks/                 # Jupyter Notebooks
│   ├── eda.ipynb              # Exploratory Data Analysis
│   ├── model_training.ipynb   # Model Training
│── models/                    # Trained models
│   ├── modelo_random_forest.pkl
│── src/                       # Core scripts
│   ├── train_model.py         # Model training script
│   ├── predict.py             # Prediction script
│── README.md                  # Project Documentation
│── requirements.txt           # Dependencies
│── .gitignore                 # Files to ignore
```

---

## 📌 Future Improvements

- ✅ **Hyperparameter tuning** for improved performance.
- ✅ **Deploy API** to serve predictions in real-time.
- ✅ **Compare with XGBoost** to validate the best model.

---

## 📬 Contact & Contributions

Feel free to fork this project, submit issues, or contribute!

🔗 **GitHub:** CesarRonai 📧 **Email: Cesar.ronai@hotmail.com**

🚀 **Let's Predict Failures More Efficiently!**

