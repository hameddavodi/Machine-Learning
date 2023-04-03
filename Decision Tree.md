## Decision Tree:
Decision tree is a popular machine learning algorithm that can be used for both classification and regression tasks. The algorithm builds a tree-like model of decisions and their possible consequences, with each internal node representing a test on an attribute and each leaf node representing a class label or a numerical value. Decision trees are easy to interpret and visualize, and they can handle both categorical and numerical data.

The main advantages of decision trees are:

  - Easy to understand and interpret: Decision trees can be easily visualized and understood, making them a good choice for exploratory data analysis.
  - Able to handle both categorical and numerical data: Decision trees can handle both types of data, without the need for data preprocessing.
  - Can handle nonlinear relationships: Decision trees can handle nonlinear relationships between the input and output variables, which makes them useful for a wide range of applications.

To use decision trees in Python, we can use the scikit-learn library, which provides a DecisionTreeClassifier class for classification and a DecisionTreeRegressor class for regression. Here's an example of how to use the DecisionTreeClassifier class to build and train a decision tree model on the iris dataset:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split

# load the iris dataset
iris = load_iris()
X = iris.data[:, :2]  # we only take the first two features for simplicity
y = iris.target

# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# create a decision tree model
model = DecisionTreeClassifier(max_depth=3)

# train the decision tree model
model.fit(X_train, y_train)

# plot the decision tree
fig, ax = plt.subplots(figsize=(12, 8))
plot_tree(model, filled=True, feature_names=iris.feature_names[:2], class_names=iris.target_names)
plt.show()
```
And finally: 

<img width="936" alt="Screenshot 2023-04-03 at 18 39 07" src="https://user-images.githubusercontent.com/109058050/229573167-6c3c4e4f-d29e-43f7-b4a1-25a66be5db7a.png">
