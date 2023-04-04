## What is ANN:
Artificial Neural Networks (ANNs) are a type of machine learning algorithm that is modeled after the structure and function of the human brain. ANNs are designed to recognize patterns and relationships in data, and they can be used for a wide range of applications, including image recognition, natural language processing, and predictive analytics. ANNs consist of interconnected nodes or neurons that process information and transmit signals to other neurons in the network. The strength of these connections can be adjusted through a process called training, which allows the network to learn from examples and improve its performance over time. ANNs have become increasingly popular in recent years due to their ability to handle complex data sets and their potential for solving challenging problems in various fields.


Here's a simple example of ANN that classifies handwritten digits using the MNIST dataset:

```python
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Flatten the images into a 1D array
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

# Normalize pixel values to [0, 1]
X_train /= 255
X_test /= 255

# Convert class labels to one-hot encoded vectors
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# Define the model architecture
model = Sequential()
model.add(Dense(512, input_dim=num_pixels, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model and specify loss function and optimizer
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model on the training data
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)

# Evaluate the model on test data and print accuracy score
scores = model.evaluate(X_test, y_test)
print("Accuracy: %.2f%%" % (scores[1]*100))
```

This code defines a simple ANN with one hidden layer of 512 neurons and a dropout layer to prevent overfitting. The model is trained on the MNIST dataset for 10 epochs with a batch size of 200. Finally, the accuracy score is printed.

To draw graphs, you can use libraries like Matplotlib or Seaborn to visualize the training and validation loss and accuracy over time. Here's an example using Matplotlib:

```
import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.plot(model.history.history['accuracy'])
plt.plot(model.history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
```

This code plots two graphs showing the training and validation accuracy and loss over time.

and finally:


