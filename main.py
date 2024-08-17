import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# 1. Fetch Financial Data
ticker = 'AAPL'  # Replace with any ticker symbol you want
data = yf.download(ticker, start='2020-01-01', end='2024-01-01')
data = data[['Close']]

# 2. Preprocess Data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

train_data = []
train_labels = []

# Use the past 60 days to predict the next day
for i in range(60, len(scaled_data)):
    train_data.append(scaled_data[i-60:i, 0])
    train_labels.append(scaled_data[i, 0])

train_data, train_labels = np.array(train_data), np.array(train_labels)
train_data = np.reshape(train_data, (train_data.shape[0], train_data.shape[1], 1))

# 3. Build and Train the LSTM Model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(units=50, return_sequences=True, input_shape=(train_data.shape[1], 1)),
    tf.keras.layers.LSTM(units=50, return_sequences=False),
    tf.keras.layers.Dense(units=25),
    tf.keras.layers.Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(train_data, train_labels, batch_size=1, epochs=1)

# 4. Make Predictions for the Future
test_data = scaled_data[-60:]  # Take the last 60 days for future prediction
future_predictions = []

for _ in range(30):  # Predict for the next 30 days
    x_test = test_data[-60:]
    x_test = np.reshape(x_test, (1, x_test.shape[0], 1))
    prediction = model.predict(x_test)
    future_predictions.append(prediction[0, 0])
    test_data = np.append(test_data, prediction)[1:]  # Update test data with the prediction

# Inverse scale the predictions to get actual price values
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# 5. Plot the Data
plt.figure(figsize=(14, 5))
plt.plot(data.index, data['Close'], label='Actual Prices')
future_dates = pd.date_range(start=data.index[-1], periods=31, freq='B')[1:]  # Generate future dates (business days)
plt.plot(future_dates, future_predictions, label='Predicted Prices', color='red')

# Annotate if price is expected to rise or fall
for i in range(1, len(future_predictions)):
    if future_predictions[i] > future_predictions[i - 1]:
        plt.annotate('↑', (future_dates[i], future_predictions[i]), color='green', fontsize=12)
    else:
        plt.annotate('↓', (future_dates[i], future_predictions[i]), color='red', fontsize=12)

plt.title(f'Stock Price Predictions for {ticker}')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
