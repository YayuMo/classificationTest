from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# print(X_train, y_train)
y_test = to_categorical(y_test)
print(y_test)
print(y_test.shape[1])