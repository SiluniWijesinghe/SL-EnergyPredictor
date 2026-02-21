# ⚡ LankaGrid-Forecaster
**Short-Term Load Forecasting using XGBoost for the Sri Lankan Power Grid**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-green.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)

---

## 📖 Project Overview
LankaGrid-Forecaster is a Machine Learning-based Decision Support System (DSS) designed to predict the electrical load demand of the Sri Lankan national grid. By utilizing **Extreme Gradient Boosting (XGBoost)**, the system provides high-precision forecasts in 15-minute intervals to help optimize power generation and grid stability.



## 🛠️ Key Technical Features
* **Dual-Temporal Logic:** Utilizes both `Day of Week` and `is_weekend` features. While correlated, this allows the model to differentiate between specific daily trends and broad industrial/holiday shifts.
* **Sunday Mapping:** Explicitly maps **Sunday to 0** to align with the Sri Lankan holiday baseline.
* **Robust Preprocessing:** Implements `StandardScaler` for normalization and **Linear Interpolation** for time-series continuity.

## 📊 Model Performance
| Metric | Value |
| :--- | :--- |
| **Algorithm** | XGBoost Regressor |
| **R² Score** | **0.9991** |
| **Scaling** | StandardScaler |
| **Interpretability** | SHAP (Shapley Additive Explanations) |



---

## 🕹️ Interactive Dashboard (Streamlit)
The deployed app allows for **Real-Time Scenario Analysis**. 

> **💡 Technical Insight:** The model identifies a "Domestic-Heavy" pattern where Sunday load (**1700.32 kW**) and Monday load (**1700.31 kW**) are nearly identical. This reveals that residential cooling demand on weekends is currently as significant as industrial starts on weekdays.

### **Simulated Scenarios:**
* **Heatwave Impact:** Slide the Temperature up to see demand spikes.
* **Rainfall Cooling:** Increase Rainfall to see the corresponding drop in grid demand (the "cooling effect").
* **Temporal Stability:** Observe the high baseline on Sundays (Day 0).



---

## 📂 Repository Structure
```bash
├── data/               # Datasets (Preprocessed)
├── models/             # Saved .joblib model and scaler
├── app.py              # Streamlit Dashboard implementation
├── preprocess_logic.py # Python script for data cleaning & engineering
└── requirements.txt    # List of dependencies
