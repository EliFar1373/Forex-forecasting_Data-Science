import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
import yfinance as yf
import time


st.title("Forex Currency Exchange Rate Prediction")
st.header("Forex LSTM Prediction")

# ----------------------------
# 1. User input
# ----------------------------
currency_pairs = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X"]
pair = st.selectbox("Select currency pair", currency_pairs)

forecast_horizon = st.number_input(
    "Forecast horizon (days)", min_value=1, max_value=365, value=30
)

# ----------------------------
# 2. Download data
# ---------------------------

# Cache the function for 1 hour
@st.cache_data(ttl=3600)
def get_data(pair, period="1y"):
    retries = 3
    for i in range(retries):
        data = yf.download(pair, period=period, interval="1d")
        if not data.empty:
            return data
        time.sleep(5)  # wait before retrying
    return pd.DataFrame()  # return empty if all retries fail

data = get_data(pair, period="1y")  # try 1y instead of 5y



if data.empty:
    st.write("⚠️ Could not connect to Yahoo Finance. Please upload a CSV file containing historical OHLC data.")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.success("CSV file loaded successfully!")
    else:
        st.stop()  # stop execution if no data available




# ---- Fix MultiIndex if present
if isinstance(data.columns, pd.MultiIndex):
    data.columns = [col[0] for col in data.columns.values]

# ---- Show preview
st.subheader("Data Preview")
st.dataframe(data.head())

# ----------------------------
# 3. Feature engineering: log-return
# ----------------------------
data["log_return"] = np.log(data["Close"] / data["Close"].shift(1))
data = data.dropna()

df = data["log_return"]

# ----------------------------
# 4. Outlier removal (IQR)
# ----------------------------

# ----------------------------
# 5. Scaling
# ----------------------------
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df.values.reshape(-1, 1))

# ----------------------------
# 6. Create sequences for LSTM
# ----------------------------
X, Y = [], []
input_sequence = 20

for i in range(len(data_scaled) - input_sequence):
    X.append(data_scaled[i:i + input_sequence])
    Y.append(data_scaled[i + input_sequence])

X = np.array(X)
Y = np.array(Y)

# ----------------------------
# 7. Load trained model
# ----------------------------
# Define running_in_docker before using it
running_in_docker = os.path.exists("/.dockerenv")  # True if inside Docker
if running_in_docker:
    # Path inside Docker container
    model_path = "models/model_LSTM_forex.keras"
    
else:
    # Path on your Windows machine / virtual environment
    model_path = r"C:\Users\Public\project\models\model_LSTM_forex.keras"



if not os.path.exists(model_path):
    st.error("Model file not found!")
    st.stop()

model_LSTM = load_model(model_path)
st.success("Model loaded successfully!")






# ----------------------------
# 8. Forecast future values
# ----------------------------
last_sample = X[-1].reshape(1, input_sequence, 1)
preds = []

for _ in range(forecast_horizon):

    next_pred = model_LSTM.predict(last_sample)[0, 0]
    preds.append(next_pred)

    # roll window left
    last_sample = np.roll(last_sample, -1)

    # place new value at last position
    last_sample[0, -1, 0] = next_pred

# Convert back to original scale
preds_inv = scaler.inverse_transform(np.array(preds).reshape(-1,1))

# ----------------------------
# 9. Display results
# ----------------------------
forecast_df = pd.DataFrame({
    "Day": np.arange(1, forecast_horizon + 1),
    "Forecast": preds_inv.flatten()
})

st.subheader("Forecast Table")
st.dataframe(forecast_df)

st.subheader("Forecast Chart")
st.line_chart(forecast_df.set_index("Day"))
