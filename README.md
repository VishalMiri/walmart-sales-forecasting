<img width="1907" height="856" alt="Screenshot 2026-07-10 202846" src="https://github.com/user-attachments/assets/1f9a893e-8eeb-47cb-8096-faf5e12ec64e" />
<img width="1890" height="859" alt="Screenshot 2026-07-10 203023" src="https://github.com/user-attachments/assets/4833addf-389b-4696-afa1-b7c17b00541a" />
# 📈 Walmart Weekly Sales Forecasting

## 📌 Overview

This project predicts Walmart weekly sales using Machine Learning.

The objective is to forecast weekly sales based on:

- Store information
- Department
- Calendar features
- Economic indicators
- Historical sales

The final model was deployed as an interactive Streamlit web application.

---

## 📊 Dataset

Source: Walmart Sales Forecasting Dataset

Files used:

- train.csv
- features.csv
- stores.csv
- test.csv

---

## 🛠 Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- XGBoost
- Streamlit
- Joblib

---

## 🤖 Models Compared

| Model | MAE | RMSE | R² |
|-------|------:|------:|------:|
| Linear Regression | 1999.45 | 4400.12 | 0.9605 |
| Random Forest | 1564.51 | 3416.80 | 0.9762 |
| XGBoost | **1498.40** | **3271.33** | **0.9781** |

---

## 🏆 Final Model

XGBoost Regressor

Performance:

- MAE : 1498.40
- RMSE : 3271.33
- R² : 0.9781

---

## 📂 Project Structure

```

app/
data/
models/
notebooks/
outputs/
src/
README.md
requirements.txt

```

---

## 🚀 Features

- Data Cleaning
- Feature Engineering
- Lag Features
- Rolling Mean Features
- Model Comparison
- Streamlit Web Application
- Interactive Sales Prediction

---

## 📸 Application Link

(https://walmart-sales-forecasting-vishal.streamlit.app/)

---

## 👨‍💻 Author

**Vishal Miri**

NIT Tiruchirappalli

Machine Learning | Data Analytics
