
import streamlit as st
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go

from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(layout="wide")

# =============================
# HEADER
# =============================
st.title("📊 Stock Forecast Dashboard")
st.caption(
    "Model Comparison: Prophet vs ARIMA"
)
st.caption(
    "This dashboard is designed to analyze stock price trends and compare forecasting performance "
    "between Prophet and ARIMA models. It helps users understand historical movements, generate future "
    "price predictions, and evaluate model accuracy using MAE, RMSE, and MAPE."
)

# =============================
# SIDEBAR
# =============================
st.sidebar.header("⚙️ Settings")

start_date = st.sidebar.date_input("Start date", datetime.date(2020,1,1))
end_date = st.sidebar.date_input("End date", datetime.date(2021,1,1))

ticker_list = pd.read_csv('/content/yahoo_tickers.txt', header=None)[0].tolist()
ticker = st.sidebar.selectbox("Stock", ticker_list)

n_day = st.sidebar.slider("Forecast Days", 1, 60, 7)

# =============================
# LOAD DATA
# =============================
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, start=start_date, end=end_date)
    data.reset_index(inplace=True)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.columns.name = None
    return data

data = load_data(ticker)

# =============================
# KEY INSIGHTS
# =============================
st.subheader("📌 Key Insights")

col1, col2, col3 = st.columns(3)

if not data.empty:
    last_price = float(data['Close'].iloc[-1])
    first_price = float(data['Close'].iloc[0])

    change = last_price - first_price
    pct_change = (change / first_price) * 100

    col1.metric("Last Price", f"{last_price:.2f}")
    col2.metric("Change", f"{change:.2f}")
    col3.metric("% Change", f"{pct_change:.2f}%")
else:
    st.warning("No data available")

# =============================
# RAW DATA
# =============================
st.subheader("Raw Data")

col1, col2 = st.columns([2,1])

with col1:
    st.caption("Preview latest data")

with col2:
    show_last = st.number_input(
        "Rows",
        min_value=5,
        max_value=len(data) if not data.empty else 100,
        value=20,
        step=5,
        label_visibility="collapsed"
    )

st.dataframe(data.tail(int(show_last)), use_container_width=True)

# =============================
# PREPARE DATA
# =============================
df = data[['Date','Close']]
df.columns = ['ds','y']
df = df.dropna()

# =============================
# PROPHET
# =============================
prophet_model = Prophet()
prophet_model.fit(df)

future = prophet_model.make_future_dataframe(periods=n_day)
forecast_prophet = prophet_model.predict(future)

# =============================
# ARIMA
# =============================
model = ARIMA(df['y'], order=(5,1,0))
model_fit = model.fit()

forecast_result = model_fit.get_forecast(steps=n_day)
arima_forecast = forecast_result.predicted_mean
conf_int_arima = forecast_result.conf_int()

future_dates = pd.date_range(
    start=df['ds'].iloc[-1],
    periods=n_day+1
)[1:]

# =============================
# MODEL COMPARISON
# =============================
st.subheader("Model Comparison")

model_option = st.selectbox(
    "Select Model",
    ["All", "Prophet Only", "ARIMA Only"]
)

show_ci = st.checkbox("Show Confidence Interval")
highlight_forecast = st.checkbox("Highlight Forecast Area")

fig = go.Figure()

# Actual
fig.add_trace(go.Scatter(
    x=df['ds'],
    y=df['y'],
    mode='lines',
    name='Actual',
    line=dict(color='black')
))

forecast_start = df['ds'].iloc[-1]

if highlight_forecast:
    fig.add_vrect(
        x0=forecast_start,
        x1=future_dates[-1],
        fillcolor="lightgray",
        opacity=0.3,
        line_width=0
    )

fig.add_annotation(
    x=forecast_start,
    y=df['y'].max(),
    text="Forecast Start",
    showarrow=True
)

# Prophet
if model_option in ["All", "Prophet Only"]:
    fig.add_trace(go.Scatter(
        x=forecast_prophet['ds'],
        y=forecast_prophet['yhat'],
        mode='lines',
        name='Prophet',
        line=dict(color='blue')
    ))

    if show_ci:
        fig.add_trace(go.Scatter(
            x=list(forecast_prophet['ds']) + list(forecast_prophet['ds'][::-1]),
            y=list(forecast_prophet['yhat_upper']) + list(forecast_prophet['yhat_lower'][::-1]),
            fill='toself',
            fillcolor='rgba(0,0,255,0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False
        ))

# ARIMA
if model_option in ["All", "ARIMA Only"]:
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=arima_forecast,
        mode='lines',
        name='ARIMA',
        line=dict(color='red')
    ))

    if show_ci:
        fig.add_trace(go.Scatter(
            x=list(future_dates) + list(future_dates[::-1]),
            y=list(conf_int_arima.iloc[:,1]) + list(conf_int_arima.iloc[:,0][::-1]),
            fill='toself',
            fillcolor='rgba(255,0,0,0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False
        ))

fig.update_layout(
    hovermode="x unified",
    xaxis_title="Date",
    yaxis_title="Price"
)

st.plotly_chart(fig, use_container_width=True)

# =============================
# PERFORMANCE
# =============================
st.subheader("Model Performance")

split = int(len(df) * 0.8)
train = df.iloc[:split]
test = df.iloc[split:]

# Prophet eval
prophet_eval = Prophet()
prophet_eval.fit(train)

future_test = prophet_eval.make_future_dataframe(periods=len(test))
forecast_test = prophet_eval.predict(future_test)

prophet_pred = forecast_test['yhat'].iloc[-len(test):].values

# ARIMA eval
model_eval = ARIMA(train['y'], order=(5,1,0))
model_eval_fit = model_eval.fit()

arima_pred = model_eval_fit.forecast(steps=len(test))

# Metrics
mae_prophet = mean_absolute_error(test['y'], prophet_pred)
rmse_prophet = np.sqrt(mean_squared_error(test['y'], prophet_pred))
mape_prophet = np.mean(np.abs((test['y'] - prophet_pred) / test['y'])) * 100

mae_arima = mean_absolute_error(test['y'], arima_pred)
rmse_arima = np.sqrt(mean_squared_error(test['y'], arima_pred))
mape_arima = np.mean(np.abs((test['y'] - arima_pred) / test['y'])) * 100

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Prophet")
    st.metric("MAE", f"{mae_prophet:.4f}")
    st.metric("RMSE", f"{rmse_prophet:.4f}")
    st.metric("MAPE", f"{mape_prophet:.2f}%")

with col2:
    st.markdown("### ARIMA")
    st.metric("MAE", f"{mae_arima:.4f}")
    st.metric("RMSE", f"{rmse_arima:.4f}")
    st.metric("MAPE", f"{mape_arima:.2f}%")

# Insight
st.subheader("🏆 Model Insight")

if rmse_prophet < rmse_arima:
    winner = "Prophet"
    best_mape = mape_prophet
    best_rmse = rmse_prophet
else:
    winner = "ARIMA"
    best_mape = mape_arima
    best_rmse = rmse_arima

def interpret_mape(mape):
    if mape < 10:
        return "✅ Good Model"
    elif mape < 20:
        return "⚠️ Acceptable Model"
    else:
        return "❌ Poor Model"

st.success(f"{winner} performs better based on RMSE")
st.markdown(f"**Model Quality:** {interpret_mape(best_mape)}")
st.markdown(f"**MAPE:** {best_mape:.2f}%")

# Extra insight
st.markdown("---")
st.markdown("### 💡 Additional Insight")

avg_price = test['y'].mean()
relative_rmse = (best_rmse / avg_price) * 100

st.write(f"Average Price (test set): **{avg_price:.2f}**")
st.write(f"Relative RMSE: **{relative_rmse:.2f}%**")

# =============================
# FOOTER
# =============================
st.markdown("---")
st.caption("Created by Jihan Kamilah | Using • Streamlit • Prophet • ARIMA")
