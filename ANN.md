## What is ANN:
Artificial Neural Networks (ANNs) are a type of machine learning algorithm that is modeled after the structure and function of the human brain. ANNs are designed to recognize patterns and relationships in data, and they can be used for a wide range of applications, including image recognition, natural language processing, and predictive analytics. ANNs consist of interconnected nodes or neurons that process information and transmit signals to other neurons in the network. The strength of these connections can be adjusted through a process called training, which allows the network to learn from examples and improve its performance over time. ANNs have become increasingly popular in recent years due to their ability to handle complex data sets and their potential for solving challenging problems in various fields.


Here's a simple example of ANN:
```python
import numpy as np
import matplotlib.pyplot as plt

# Define the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the derivative of sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Define the input dataset
X = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1]])

# Define the output dataset
y = np.array([[0],
              [1],
              [1],
              [0]])

# Set random seed for reproducibility
np.random.seed(42)

# Initialize weights randomly with mean 0
weights_0 = 2 * np.random.random((3,4)) - 1
weights_1 = 2 * np.random.random((4,1)) - 1

# Set learning rate and number of iterations
learning_rate = 0.5
num_iterations = 10000

# Train the neural network using backpropagation algorithm
for i in range(num_iterations):

    # Forward propagation
    layer_0 = X
    layer_1 = sigmoid(np.dot(layer_0, weights_0))
    layer_2 = sigmoid(np.dot(layer_1, weights_1))

    # Calculate error and delta for output layer
    layer_2_error = y - layer_2
    layer_2_delta = layer_2_error * sigmoid_derivative(layer_2)

    # Calculate error and delta for hidden layer
    layer_1_error = layer_2_delta.dot(weights_1.T)
    layer_1_delta = layer_1_error * sigmoid_derivative(layer_1)

    # Update weights
    weights_1 += learning_rate * layer_1.T.dot(layer_2_delta)
    weights_0 += learning_rate * layer_0.T.dot(layer_1_delta)

# Print the final output
print("Output after training:")
print(layer_2)

```

This code defines a simple neural network with one hidden layer and trains it using the backpropagation algorithm. The input dataset is a 4x3 matrix, and the output dataset is a 4x1 matrix. The neural network uses sigmoid activation function for both layers. The weights are initialized randomly with mean 0, and the learning rate is set to 0.5. The number of iterations is set to 10,000.

To plot the result, you can use matplotlib library to create a graph of the output values. Here's an example code:

```
import matplotlib.pyplot as plt

# Plot the output values
plt.plot(layer_2)
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('Neural Network Output')
plt.show()

```

and finally:

```python
Output after training:
[[0.00972592]

 [0.98200874]
 
 [0.98440765]
 
 [0.01969368]]
```

 <img width="584" alt="Screenshot 2023-04-04 at 16 05 38" src="https://user-images.githubusercontent.com/109058050/229818397-a640920a-8f55-4da5-b85c-4bc780f38c6b.png">


