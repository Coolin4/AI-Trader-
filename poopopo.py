import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

msft = yf.download("MSFT", start="2018-01-01", end="2019-12-31")

msft['Weekly_Return'] = msft['Close'].pct_change(periods=5).shift(-5)

msft['Target'] = np.where(msft['Weekly_Return'] > 0, 1, 0)

msft.dropna(inplace=True)

train_data = msft[msft.index < '2019-01-01']
test_data = msft[msft.index >= '2019-01-01']

train_features = train_data[['Open', 'High', 'Low', 'Close', 'Volume']].values
test_features = test_data[['Open', 'High', 'Low', 'Close', 'Volume']].values

train_target = train_data['Target'].values
test_target = test_data['Target'].values

mean = np.mean(train_features, axis=0)
std = np.std(train_features, axis=0)
train_features = (train_features - mean) / std
test_features = (test_features - mean) / std


model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(train_features.shape[1],)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_features, train_target, epochs=7000, batch_size=32, verbose=1)

predictions = model.predict(test_features)

dates = test_data.index

accuracy = np.mean(np.round(predictions) == test_target)

plt.figure(figsize=(12, 6))
plt.plot(dates, test_data['Close'], label='Close Price')

for i in range(len(predictions)):
    if predictions[i] > 0.5:
        plt.scatter(dates[i], test_data['Close'].iloc[i], color='green', marker='o')

plt.text(dates[-1], test_data['Close'].iloc[-1], f'Accuracy: {accuracy:.2%}', ha='right', va='bottom')

plt.title('MSFT Stock Prices with Long Positions')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.xticks(rotation=45)
plt.show()
