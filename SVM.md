## What is SVM:
Support Vector Machine (SVM) is a popular machine learning algorithm used for classification and regression analysis. SVM is a powerful algorithm that can be used for a wide range of applications including text classification, image classification, and bioinformatics.
SVM is a discriminative algorithm that classifies data by finding the best boundary (called the hyperplane) between two classes. The hyperplane is chosen so that it maximizes the margin, which is the distance between the hyperplane and the closest data points from each class. SVM can handle high dimensional data, non-linear decision boundaries, and outliers. It is also less prone to overfitting compared to other classification algorithms. These properties make SVM a popular choice in many applications.
To implement SVM in Python, we can use the scikit-learn library which provides a simple and easy-to-use interface for SVM. Here is a step-by-step guide to implement SVM in Python:

Step 1: Load the dataset

First, we need to load the dataset. Scikit-learn provides many built-in datasets that we can use for testing, such as the iris dataset or the breast cancer dataset. We can also load our own dataset using pandas or numpy.

Step 2: Split the dataset into training and testing sets

Next, we need to split the dataset into training and testing sets. The training set is used to train the model, and the testing set is used to evaluate the model's performance. Scikit-learn provides a function called train_test_split that can split the dataset for us.

Step 3: Create an SVM model

We can create an SVM model by importing the SVC (Support Vector Classification) class from the scikit-learn library. We can set the parameters of the SVC class, such as the kernel function, regularization parameter, and gamma parameter.

Step 4: Train the SVM model

We can train the SVM model by calling the fit function of the SVC class and passing in the training data and labels.

Step 5: Evaluate the SVM model

We can evaluate the SVM model by calling the score function of the SVC class and passing in the testing data and labels. The score function returns the accuracy of the model on the testing set.

Step 6: Use the SVM model for prediction

We can use the SVM model for prediction by calling the predict function of the SVC class and passing in new data.
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# load the dataset
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features for simplicity
y = iris.target

# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# create an SVM model
model = SVC(kernel='linear', C=1, gamma='auto')

# train the SVM model
model.fit(X_train, y_train)

# plot the decision boundary
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)

# plot the training points
colors = ['red', 'blue', 'green']
for i, color in zip(model.classes_, colors):
    idx = np.where(y_train == i)
    plt.scatter(X_train[idx, 0], X_train[idx, 1], c=color, label=iris.target_names[i], cmap=plt.cm.Spectral, edgecolors='black')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('SVM Decision Boundary')
plt.legend()
plt.show()
```
And finally:

<img width="604" alt="Screenshot 2023-04-03 at 18 35 14" src="https://user-images.githubusercontent.com/109058050/229572368-a7a50f19-2902-46c3-ad3c-fc012264f32f.png">

