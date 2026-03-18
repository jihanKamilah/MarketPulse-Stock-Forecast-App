# 📊 Stock Forecast Dashboard

An interactive web application for forecasting stock prices using time series models.

## 🚀 Overview

This dashboard enables users to:

* Explore historical stock price data
* Generate future forecasts
* Compare multiple forecasting models

Two models are implemented:

* **Prophet** → captures trend & seasonality
* **ARIMA** → captures statistical patterns in time series

---

## 📌 Key Features

* 📈 Interactive stock price visualization
* 🔮 Forecast up to N days ahead
* ⚖️ Model comparison (Prophet vs ARIMA)
* 📉 Performance metrics (MAE, RMSE & MAPE)
* 🎛️ User controls (stock selection, forecast horizon)

---

## 🛠️ Tech Stack

* Python
* Streamlit
* Prophet
* ARIMA (statsmodels)
* Plotly
* yFinance

---

## 🌐 Live Demo

👉 https://your-app-name.streamlit.app

---

## 🚀 Deployment Options

### 🌐 1. Streamlit Cloud (Recommended)

The app is deployed using Streamlit Community Cloud for a stable and permanent link.

**Advantages:**

* Always online
* No setup required
* Ideal for portfolio

---

### 🔗 2. Local + ngrok (Development Only)

Run locally and expose using ngrok:

```bash
streamlit run app.py
```

```python
from pyngrok import ngrok
ngrok.connect(8501)
```

Or here you can running the Code ```(Stock_Forecasting_Dashboard.ipynb)```

**Notes:**

* Temporary URL
* Requires active runtime
* Used for development/testing

---

## 📂 How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📊 Model Insight Example

* Prophet performs well on trending data
* ARIMA performs better on stable patterns

---

## 👤 Author

Jihan Kamilah
