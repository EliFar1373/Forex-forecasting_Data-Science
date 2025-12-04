import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error

st.title("Forex Currency Exchange Rate Prediction")
st.header("Forex LSTM Prediction")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(data.head())
    
    # Calculate log returns
    data['log_return'] = np.log(data['Close_EURUSD'] / data['Close_EURUSD'].shift(1))
    data = data.dropna()
    df = data['log_return']
    
    # Histogram
    fig, ax = plt.subplots()
    sns.histplot(df, kde=True, ax=ax)
    ax.set_title("Histogram of Log Returns")
    st.pyplot(fig)
    
    # Box plot
    fig_box, ax_box = plt.subplots()
    sns.boxplot(df, ax=ax_box)
    ax_box.set_title("Box Plot of Log Returns")
    st.pyplot(fig_box)
    
    # Outlier removal
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_clean = df[(df >= lower_bound) & (df <= upper_bound)].reset_index(drop=True)
    
    st.subheader("Cleaned Data")
    st.line_chart(df_clean)
    
    # Scaling
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df_clean.values.reshape(-1,1))
    
    # Prepare sequences
    X, Y = [], []
    input_sequence = 10
    for i in range(len(data_scaled) - input_sequence):
        X.append(data_scaled[i:i+input_sequence])
        Y.append(data_scaled[i + input_sequence])
    X = np.array(X)
    Y = np.array(Y)
    
    # Split data
    train_size = int(len(X)*0.7)
    val_size = int(len(X)*0.15)
    X_train, Y_train = X[:train_size], Y[:train_size]
    X_val, Y_val = X[train_size:train_size+val_size], Y[train_size:train_size+val_size]
    X_test, Y_test = X[train_size+val_size:], Y[train_size+val_size:]
    
    # Load model
    model_path = os.path.join("models","model_LSTM.h5")
    model_LSTM = load_model(model_path)
    
    # Predict
    y_pred = model_LSTM.predict(X_test)
    
    # Evaluation
    rmse = np.sqrt(mean_squared_error(Y_test, y_pred))
    mae = mean_absolute_error(Y_test, y_pred)
    st.subheader("Evaluation:")
    st.write(f"RMSE: {rmse:.6f}")
    st.write(f"MAE: {mae:.6f}")
    
    # Predicted vs Actual
    Pred = pd.DataFrame({"Predicted": y_pred.flatten(), "Actual": Y_test.flatten()})
    st.subheader("Predicted vs Actual")
    st.line_chart(Pred)
    
else:
    st.write("Please upload a CSV file to continue.")
