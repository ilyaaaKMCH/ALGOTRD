import yfinance as yf
from finta import TA
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
# Загрузка данных
stock = 'BTC-USD'
start = '2014-09-17'
end = '2024-02-13'

df = yf.download(stock, start, end)
df.index = df.index.date
df.fillna(0, inplace=True)
df['RSI'] = TA.RSI(df, 12)
df['SMA'] = TA.SMA(df)
df['OBV'] = TA.OBV(df)
df = df.fillna(0)
scaler = MinMaxScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
test_data = df['Close'][int(0.8*len(df))-100:].values
scaler_test = MinMaxScaler()
scaled_data = scaler_test.fit_transform(test_data.reshape(-1, 1))

x_test, y_test = [], []

for i in range(100, len(test_data)):
    x_test.append(scaled_data[i-100:i, 0])
    y_test.append(scaled_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
# Подготовка тренировочных данных
training_data = df['Close'][:int(0.8*len(df))].values
scaler = MinMaxScaler()
training_data = scaler.fit_transform(training_data.reshape(-1, 1))

x_train, y_train = [], []

for i in range(100, len(training_data)):
    x_train.append(training_data[i-100:i, 0])
    y_train.append(training_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))  # Преобразование для LSTM

# Создание и компиляция модели
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(units=60, return_sequences=True),
    Dropout(0.3),
    LSTM(units=80, return_sequences=True),
    Dropout(0.4),
    LSTM(units=120, return_sequences=False),
    Dropout(0.5),
    Dense(units=1, activation='sigmoid')
])



# Обучение модели
model.compile(optimizer='adam', loss='mean_squared_error')

# EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Обучение модели с использованием EarlyStopping
history = model.fit(x_train, y_train, epochs=30, batch_size=32, verbose=1, validation_data=(x_test, y_test), callbacks=[early_stopping])



# Предсказание
predicted_prices = model.predict(x_test)
predicted_prices = scaler_test.inverse_transform(predicted_prices)

true_prices = scaler_test.inverse_transform(y_test.reshape(-1, 1))
# y_predict = model.predict(x_test)
# scaler_close = MinMaxScaler()
# df['Close'] = scaler_close.fit_transform(df[['Close']])
# y_predict_rescaled = scaler_close.inverse_transform(y_predict)
# y_test_rescaled = scaler_close.inverse_transform(y_test.reshape(-1, 1))
# percentage_change = (y_predict_rescaled - y_test_rescaled) / y_test_rescaled
#
# # # Определение сигналов к покупке и продаже
# buy_signals = percentage_change > 0.01  # Сигнал к покупке, если предсказанная цена на 1% выше
# sell_signals = percentage_change < -0.01  # Сигнал к продаже, если предсказанная цена на 1% ниже
# #
# # Визуализация результатов
# plt.figure(figsize=(14, 7))
# plt.plot(y_test_rescaled, color='g', label='Original Prices')
# plt.plot(y_predict_rescaled, color='r', linestyle='--', label='Predicted Price')
# plt.scatter(np.where(buy_signals), y_test_rescaled[buy_signals], marker='^', color='b', label='Buy Signal', alpha=1)
# plt.scatter(np.where(sell_signals), y_test_rescaled[sell_signals], marker='v', color='m', label='Sell Signal', alpha=1)
# plt.title('Stock Price Prediction with Buy & Sell Signals')
# plt.xlabel('Time')
# plt.ylabel('Price')
# plt.legend()
# plt.show()

# Визуализация
plt.plot(true_prices, label='True Prices')
plt.plot(predicted_prices, label='Predicted Prices')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()










