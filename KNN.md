## Introduction
KNN is a machine learning algorithm that can be used for classification and regression tasks. In classification tasks, the algorithm works by finding the K nearest data points to a new, unseen data point and then assigns a label to that data point based on the majority class of its K nearest neighbors. In regression tasks, the algorithm works in a similar way by finding the K nearest data points and then predicting the target value for the new data point based on the average of the target values of its K nearest neighbors.

### Python code:

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate random data
X = np.random.rand(100, 2)
y = np.random.choice([0, 1], size=100)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)

# Train the classifier on the training data
knn.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = knn.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# Plot the decision boundary
x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('KNN decision boundary')
plt.show()






In this example, we use the numpy library to generate random data for the KNN algorithm. We generate 100 data points with 2 features (i.e., height and weight) and assign each data point a random gender label ('Male' or 'Female').

We then split the data into training and testing sets using the train_test_split function from scikit-learn, create a KNN classifier with n_neighbors set to 3, train the classifier on the training data using the fit method, make predictions on the testing data using the predict method, and calculate the accuracy of the classifier using the accuracy_score function from scikit-learn. The output will be the accuracy of the classifier on the testing data.
