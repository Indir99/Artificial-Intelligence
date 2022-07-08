from keras.layers import LSTM
from keras.layers import Dense
from keras.models import Sequential
from keras.preprocessing.sequence import TimeseriesGenerator

import math
import matplotlib.pyplot as plt
import numpy as np

X_train = np.arange(0, 80, 0.1)
y_train = np.cos(X_train)

X_test = np.arange(80, 100, 0.1)
y_test = np.cos(X_test)


n_features = 1

train_series = y_train.reshape((len(y_train), n_features))
test_series = y_test.reshape((len(y_test), n_features))

fig, ax = plt.subplots(1, 1, figsize=(15, 4))
ax.plot(X_train, y_train, lw=3, c='b', label='Training data')
ax.plot(X_test, y_test,  lw=3, c='g', label='Testing data')
ax.legend(loc="lower left")
plt.show()

look_back = 20

train_generator = TimeseriesGenerator(train_series, train_series,
                                      length=look_back,
                                      sampling_rate=1,
                                      stride=1,
                                      batch_size=10)

test_generator = TimeseriesGenerator(test_series, test_series,
                                     length=look_back,
                                     sampling_rate=1,
                                     stride=1,
                                     batch_size=10)


n_neurons = 12
model = Sequential()
model.add(LSTM(n_neurons, input_shape=(look_back, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.fit(train_generator, epochs=20, verbose=1)

test_predictions = model.predict(test_generator)

print(model.summary())
print(len(model.layers))

x = np.arange(82, 100, 0.1)
#x = np.arange(110, 200, 0.5)
fig, ax = plt.subplots(1, 1, figsize=(15, 5))
ax.plot(X_train, y_train, lw=2, label='train data')
ax.plot(X_test, y_test, lw=3, c='y', label='test data')
ax.plot(x, test_predictions, lw=3, c='r', linestyle=':', label='predictions')
ax.legend(loc="lower left")
plt.show()

extrapolation = list()
seed_batch = y_test[:look_back].reshape((1, look_back, n_features))
current_batch = seed_batch

# extrapolate the next 180 values
for i in range(180):
    predicted_value = model.predict(current_batch)[0]
    extrapolation.append(predicted_value)
    current_batch = np.append(current_batch[:, 1:, :], [
                              [predicted_value]], axis=1)


x = np.arange(82, 100, 0.1)
#x = np.arange(110, 200, 0.5)
fig, ax = plt.subplots(1, 1, figsize=(15, 5))
ax.plot(X_train, y_train, lw=2, label='train data')
ax.plot(X_test, y_test, lw=3, c='y', label='test data')
ax.plot(x, extrapolation, lw=3, c='r', linestyle=':', label='extrapolation')
ax.legend(loc="lower left")
plt.show()
