## PCA and its use in Dimension reduction
PCA (Principal Component Analysis) is a popular technique in machine learning and data analysis for reducing the dimensionality of a dataset while preserving the most important information. It is a linear transformation method that finds the directions (or principal components) in the feature space that account for the maximum variance in the data, and projects the data onto these directions to obtain a lower-dimensional representation.

The reason for doing PCA is to reduce the complexity of a high-dimensional dataset, while retaining the most important information. This can be useful in various applications, such as data visualization, feature extraction, and data compression. For example, in data visualization, it can be difficult to visualize a high-dimensional dataset directly, but by applying PCA to reduce the dimensionality, we can visualize the data in a lower-dimensional space while still retaining most of the relevant information.

PCA works by identifying the directions in the feature space that account for the most variance in the data. These directions are called principal components, and they form a new orthogonal basis for the data. The first principal component corresponds to the direction of maximum variance, the second principal component corresponds to the direction of maximum variance orthogonal to the first component, and so on. By projecting the data onto these principal components, we can obtain a lower-dimensional representation of the data that captures the most important information.

The PCA transformation can be computed using a variety of methods, including the eigenvalue decomposition of the covariance matrix, singular value decomposition (SVD), and iterative methods. The choice of method may depend on the specific application and the size of the dataset.

```python
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# Generate some random data with 3 features
X = np.random.rand(100, 3)

# Create a PCA object with 2 components
pca = PCA(n_components=2)

# Fit the PCA model to the data and transform it
X_pca = pca.fit_transform(X)

# Plot the original data
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].scatter(X[:, 0], X[:, 1])
ax[0].set_xlabel('Feature 1')
ax[0].set_ylabel('Feature 2')
ax[0].set_title('Original Data')

# Plot the transformed data with two principal components
ax[1].scatter(X_pca[:, 0], X_pca[:, 1])
ax[1].set_xlabel('Principal Component 1')
ax[1].set_ylabel('Principal Component 2')
ax[1].set_title('PCA Transformed Data')

# Show the plots
plt.show()
```
And the output:

<img width="880" alt="Screenshot 2023-04-03 at 18 27 56" src="https://user-images.githubusercontent.com/109058050/229570747-5c63b440-ad70-4857-8325-d302767e4996.png">


