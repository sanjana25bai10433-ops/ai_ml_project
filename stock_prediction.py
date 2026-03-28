# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from statsmodels.tsa.arima.model import ARIMA

from keras.models import Sequential
from keras.layers import LSTM, Dense


# load data
stock = "AAPL" # change to TSLA, GOOLE etc.
data = yf.download(stock, start = "2015-01-01", end = "2024-01-01")

df = data[['Close']]
df.dropna(inplace=True)

# visualisation
plt.figure(figsize =(10,5))
plt.plot(df['Close'], label= 'closing price')
plt.title(f"{stock} stock price")
plt.legend()
plt.show()

# scaling data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(df)

# create dataset for LSTM
def create_dataset(data, time_step=60):
    x , y = [], []
    for i in range(len(data) - time_step-1):
        x.append(data[i:(i+time_step), 0])
        y.append(data[i+time_step,0])
    return np.array(x), np.array(y) 

time_step = 60
x, y = create_dataset(scaled_data, time_step)   

# reshape for LSTM
x = x.reshape(x.shape[0], x.shape[1], 1)

# Train-Test split
train_size = int(len(x)*0.8)
x_train, x_test = x[:train_size], x[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step,1)))
model.add(LSTM(50))
model.add(Dense(1))

model.compile(loss='mean_squared_error',optimizer='adam')

# train model
model.fit(x_train,y_train, epochs=10, batch_size=32)

# predictions
train_pred = scaler.inverse_transform(x_train)
test_pred = scaler.inverse_transform(x_test)

y_train_actual = scaler.inverse_transform(y_train.reshape(-1,1))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1,1))

# evaluation
rmse = np.sqrt(mean_squared_error(y_test_actual, test_pred))
print("LSTM RMSE", rmse)

# plot predictions
plt.figure(figsize=(10,5))
plt.plot(y_test_actual, label='actual')
plt.plot(test_pred, label='predicted')
plt.legend()
plt.title("LSTM prediction")
plt.show()

# arima model
arima_model = ARIMA(df['close'],order(5,1,0))
arima_result = arima_model.fit()

arima_pred = arima_result.forecast(steps=30)

# plot arima forcast

plt.figure(figsize=(10,5))
plt.plot(df['close'], label='original')
plt.plot(range(len(df), len(df)+30), arima_pred, label='ARIMA forecast')
plt.legend()
plt.title("ARIMA FORECAST")
plt.show()