import math
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
print(tf.__version__)


X = np.linspace(0, 20, 1000)
y = np.cos(X)


X.shape, y.shape
len(X)

X_train = X[:800]
y_train = y[:800]
X_test = X[0:1000]
y_test = y[0:1000]
len(X_train), len(X_test)

plt.figure(figsize=(12, 6))
plt.scatter(X_train, y_train, c='b', label='Training data')
plt.scatter(X_test, y_test, c='g', label='Testing data')
plt.show()


tf.random.set_seed(42)
model_1 = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation=tf.keras.activations.tanh),
])
model_1.compile(loss=tf.keras.losses.mse,
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['mse'])
model_1.fit(tf.expand_dims(X_train, axis=-1), y_train, epochs=20, verbose=1)
preds = model_1.predict(X_test)
preds


def plot_preds(traindata=X_train,
               trainlabels=y_train,
               testdata=X_test,
               testlabels=y_test,
               predictions=preds):
    plt.figure(figsize=(12, 6))
    plt.scatter(traindata, trainlabels, c="b", label="Training data")
    plt.scatter(testdata, testlabels, c="g", label="Testing data")
    plt.scatter(testdata, predictions, c="r", label="Predictions")
    plt.legend()


plot_preds(traindata=X_train,

           trainlabels=y_train,

           testdata=X_test,

           testlabels=y_test,

           predictions=preds)


print(model_1.summary())
print(len(model_1.layers))
tf.random.set_seed(42)
model_2 = tf.keras.Sequential([
    tf.keras.layers.Dense(60, activation=tf.keras.activations.tanh),
    # povecan broj neurona daje bolji model
    tf.keras.layers.Dense(80, activation=tf.keras.activations.tanh),
    tf.keras.layers.Dense(100, activation=tf.keras.activations.tanh),
    # zadnji sloj mora biti jednak dimenzijama sistema
    tf.keras.layers.Dense(1)
])
model_2.compile(loss=tf.keras.losses.mse,
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['mse'])
model_2.fit(tf.expand_dims(X_train, axis=-1), y_train, epochs=100, verbose=1)
preds_2 = model_2.predict(X_test)
plot_preds(predictions=preds_2)
print(model_2.summary())
print(len(model_2.layers))

# tf.random.set_seed(42)
# model_3 = tf.keras.Sequential([
#     tf.keras.layers.Dense(1000, activation=tf.keras.activations.tanh),
#     tf.keras.layers.Dense(1000, activation=tf.keras.activations.tanh),
#     tf.keras.layers.Dense(1000, activation=tf.keras.activations.tanh),
#     tf.keras.layers.Dense(800, activation=tf.keras.activations.tanh),
#     tf.keras.layers.Dense(900, activation=tf.keras.activations.tanh),
#     # zadnji sloj mora biti jednak dimenzijama sistema
#     tf.keras.layers.Dense(1, activation='linear')
# ])
# model_3.compile(loss=tf.keras.losses.mse,
#                 optimizer=tf.keras.optimizers.Adam(),
#                 metrics=['mse'])
# model_3.fit(tf.expand_dims(X_train, axis=-1), y_train, epochs=100, verbose=0)

# preds_3 = model_3.predict(X_test)
# plot_preds(predictions=preds_3)

# print(model_3.summary())
# print(len(model_3.layers))
plt.show()
