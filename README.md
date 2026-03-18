# 📊 Stock Forecast App (Dashboard Using Streamlit And Ngrok) 
**Model Comparison: Prophet vs ARIMA**

An interactive web application for analyzing stock price trends and forecasting future movements using time series models.

---

## 🚀 Overview

This dashboard helps users explore historical stock data, generate future price predictions, and compare forecasting performance between two popular models:

- **Prophet** → captures trend and seasonality effectively  
- **ARIMA** → models statistical patterns in time series data  

It is designed as an end-to-end analytical tool, combining data exploration, forecasting, and model evaluation in one interface.

---

## ✨ Key Features

- 📈 Interactive visualization of stock prices  
- 🔮 Forecast future prices up to N days ahead  
- ⚖️ Side-by-side model comparison (Prophet vs ARIMA)  
- 📉 Performance evaluation using:
  - MAE (Mean Absolute Error)  
  - RMSE (Root Mean Squared Error)  
  - MAPE (Mean Absolute Percentage Error)  
- 🎛️ Dynamic user controls:
  - Stock selection  
  - Date range  
  - Forecast horizon  
- 📊 Data preview with customizable row display  

---

## 🛠️ Tech Stack

- **Python**  
- **Streamlit** (web app framework)  
- **Prophet** (time series forecasting)  
- **ARIMA** – `statsmodels`  
- **Plotly** (interactive visualization)  
- **yFinance** (data source)  

---

## 🌐 Live Demo

👉 https://marketpulse-stock-forecast-app.streamlit.app/

---

## 🚀 Deployment Options

### 🌐 1. Streamlit Community Cloud *(Recommended)*

This project is deployed using Streamlit Cloud for a stable and publicly accessible application.

**Advantages:**
- Always online  
- No local setup required  
- Ideal for showcasing portfolio projects  

---

### 🔗 2. Local + ngrok *(Development Only)*

Run the app locally and expose it using ngrok:

```bash
streamlit run app.py
```

```python
from pyngrok import ngrok
ngrok.connect(8501)
```

You can also run the notebook version:
```
Stock_Forecasting_Dashboard.ipynb
```

**Notes:**
- Temporary public URL  
- Requires active runtime session  
- Suitable for testing and development  

---

## 📂 How to Run Locally

1. Clone the repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the app
```bash
streamlit run app.py
```

---

## 📊 Model Insights

- **Prophet** performs better on data with clear trends and seasonality  
- **ARIMA** performs better on stable and stationary time series  
- Model performance is evaluated using error metrics to ensure fair comparison  

---

## ⚠️ Notes

- Forecast accuracy depends on the selected stock and date range  
- Some tickers may return limited or no data from Yahoo Finance  
- Ensure sufficient historical data for reliable predictions  

---

## 👤 Author

**Jihan Kamilah**
