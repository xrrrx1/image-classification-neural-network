import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print('TF version:', tf.__version__)

print('x train shape:', x_train.shape)
print('y train shape:', y_train.shape)
print('x test shape:', x_test.shape)
print('y test shape:', y_test.shape)

# matplotlib inline

# plt.imshow(x_train[0], cmap='binary')
# plt.show()

print(y_train[0])
print(set(y_train))

y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)

print('y train encoded shape:', y_train_encoded.shape)
print('y test encoded shape:', y_test_encoded.shape)

print(y_train_encoded[0])

x_train_reshaped = np.reshape(x_train, (60000, 784))
x_test_reshaped = np.reshape(x_test, (10000, 784))

print('x train reshaped shape:', x_train_reshaped.shape)
print('x test encoded shape:', x_test_reshaped.shape)

print(set(x_train_reshaped[0]))

x_mean = np.mean(x_train_reshaped)
x_std = np.std(x_train_reshaped)

epsilon = 1e-10

x_train_norm = (x_train_reshaped - x_mean) / (x_std + epsilon)
x_test_norm = (x_test_reshaped - x_mean) / (x_std + epsilon)

print(set(x_train_norm[0]))

model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(
    optimizer='sgd',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# model.summary()

model.fit(x_train_norm, y_train_encoded, epochs=3)

args, accuracy = model.evaluate(x_test_norm, y_test_encoded)
print('Accuracy: ', accuracy * 100)

predictions = model.predict(x_test_norm)
print('Shape of preds: ', predictions.shape)

plt.figure(figsize=(12, 12))

start_index = 0

for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    prediction = np.argmax(predictions[start_index + i])
    gt = y_test[start_index + i]

    col = 'g'
    if prediction != gt:
        col = 'r'

    plt.xlabel(f'i={start_index + i}, prediction={prediction}, gt={gt}', color=col)
    plt.imshow(x_test[start_index + i], cmap='binary')

plt.show()

plt.plot(predictions[8])
plt.show()
