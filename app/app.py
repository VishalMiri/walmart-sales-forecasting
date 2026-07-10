import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Walmart Weekly Sales Forecasting",
    page_icon="📈",
    layout="wide"
)

# -----------------------------
# Load Saved Model
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "forecasting_pipeline.pkl"

artifacts = joblib.load(MODEL_PATH)

model = artifacts["model"]
feature_cols = artifacts["features"]

# -----------------------------
# Load Store Information
# -----------------------------

STORE_PATH = BASE_DIR / "data" / "stores.csv"

stores_df = pd.read_csv(STORE_PATH)

# -----------------------------
# Title
# -----------------------------
st.title("📈 Walmart Weekly Sales Forecasting")

st.caption(
    "Predict weekly retail sales using a trained XGBoost model."
)

st.markdown("---")

# -----------------------------
# Sidebar
# -----------------------------

st.sidebar.title("📊 Project Information")

st.sidebar.markdown("""
### Walmart Weekly Sales Forecasting

This application predicts Walmart weekly sales using a machine learning model trained on historical sales data.

---

### Model

✅ XGBoost Regressor

---

### Model Performance

**R² Score:** 0.9781

**MAE:** ₹1,498

**RMSE:** ₹3,271

---

### Tech Stack

- Python
- Pandas
- Scikit-learn
- XGBoost
- Streamlit

---

Developed by Vishal Miri
""")

# -----------------------------
# Forecast Information
# -----------------------------

st.subheader("📅 Forecast Information")

forecast_date = st.date_input(
    "Forecast Date"
)

is_holiday = st.checkbox(
    "Holiday Week"
)

# Automatically create date features

year = forecast_date.year

month = forecast_date.month

week = forecast_date.isocalendar().week

quarter = (month - 1) // 3 + 1

day_of_week = forecast_date.weekday()

# -----------------------------
# Store Information
# -----------------------------

st.markdown("---")

st.subheader("🏪 Store Information")

store = st.selectbox(
    "Store",
    options=list(range(1, 46))
)

dept = st.selectbox(
    "Department",
    options=list(range(1, 100))
)

# -----------------------------
# External Factors
# -----------------------------

st.markdown("---")

st.subheader("🌍 External Factors")

temperature = st.number_input(
    "Temperature (°C)",
    value=25.0,
    step=0.1
)

fuel_price = st.number_input(
    "Fuel Price",
    value=3.50,
    step=0.01
)

cpi = st.number_input(
    "Consumer Price Index (CPI)",
    value=210.0,
    step=0.1
)

unemployment = st.number_input(
    "Unemployment Rate",
    value=7.0,
    step=0.1
)

# -----------------------------
# Historical Sales
# -----------------------------

st.markdown("---")

st.subheader("📊 Historical Sales")

lag_1 = st.number_input(
    "Previous Week Sales",
    min_value=0.0,
    value=40000.0,
    step=100.0
)

lag_2 = st.number_input(
    "Sales 2 Weeks Ago",
    min_value=0.0,
    value=39000.0,
    step=100.0
)

lag_4 = st.number_input(
    "Sales 4 Weeks Ago",
    min_value=0.0,
    value=41000.0,
    step=100.0
)

rolling_mean = (lag_1 + lag_2 + lag_4) / 3

# -----------------------------
# Prediction
# -----------------------------

st.markdown("---")

predict_button = st.button(
    "🚀 Predict Weekly Sales",
    use_container_width=True
)

store_size = stores_df.loc[
    stores_df["Store"] == store,
    "Size"
].iloc[0]

if predict_button:

    input_data = pd.DataFrame({
        "Store": [store],
        "Dept": [dept],
        "IsHoliday": [is_holiday],
        "Temperature": [temperature],
        "Fuel_Price": [fuel_price],
        "CPI": [cpi],
        "Unemployment": [unemployment],
        "Size": [store_size],
        "Year": [year],
        "Month": [month],
        "Week": [week],
        "Quarter": [quarter],
        "DayOfWeek": [day_of_week],
        "Lag_1": [lag_1],
        "Lag_2": [lag_2],
        "Lag_4": [lag_4],
        "Rolling_Mean_4": [rolling_mean]
    })

    prediction = model.predict(input_data)

    st.markdown("---")

    st.subheader("📈 Prediction")

    st.metric(
        label="Expected Weekly Sales",
        value=f"₹ {prediction[0]:,.2f}"
    )

    st.success("Prediction completed successfully!")

st.markdown("### Input Summary")

summary = pd.DataFrame({
    "Feature": [
        "Store",
        "Department",
        "Forecast Date",
        "Holiday",
        "Temperature",
        "Fuel Price",
        "CPI",
        "Unemployment"
    ],
    "Value": [
        store,
        dept,
        forecast_date,
        is_holiday,
        temperature,
        fuel_price,
        cpi,
        unemployment
    ]
})

st.dataframe(
    summary,
    use_container_width=True,
    hide_index=True
)

st.markdown("---")

st.caption(
    "Built by Vishal Miri • Machine Learning Portfolio Project • 2026"
)