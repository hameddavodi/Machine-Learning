## What is K-means:

K-means clustering is a type of unsupervised learning algorithm used in machine learning and data mining to partition a set of data points into a fixed number (k) of clusters. The goal of k-means clustering is to group the data points in such a way that the points within a group (cluster) are similar to each other and different from the points in other groups.

The k-means algorithm starts by randomly selecting k initial cluster centers, and then iteratively assigns each data point to the nearest cluster center, and re-computes the center of each cluster as the mean of the points assigned to it. The process is repeated until the cluster centers converge, that is, until the assignment of data points to clusters no longer changes significantly.

The k-means algorithm has several advantages, including its simplicity, computational efficiency, and effectiveness in handling large datasets. However, it also has some limitations, such as the sensitivity to the initial choice of cluster centers and the assumption that the clusters are spherical and have equal variance.

Note: K-means clustering has a wide range of applications, including image segmentation, customer segmentation, anomaly detection, and document clustering.

Let's start by importing necessary libraries:

```python 
from sklearn.cluster import KMeans
```
Next, let’s define the inputs we will use for our K-means clustering algorithm. For example age and spending score:

```python 
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Generate some random data
X = np.random.rand(100, 2)

# Create a KMeans object with 3 clusters
kmeans = KMeans(n_clusters=3)

# Fit the KMeans model to the data
kmeans.fit(X)

# Get the cluster labels for each data point
labels = kmeans.labels_

# Get the centroids of each cluster
centroids = kmeans.cluster_centers_

# Create a colormap for the clusters
cmap = plt.cm.get_cmap('viridis', len(np.unique(labels)))

# Plot the data points with color-coded clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap=cmap)

# Plot the centroids as black crosses with larger size
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, linewidths=3, color='k')

# Set the axis limits and add axis labels
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('X')
plt.ylabel('Y')

# Add a colorbar legend to show the cluster assignments
cb = plt.colorbar()
cb.set_label('Cluster')

# Add a title to the plot
plt.title('KMeans Clustering')

# Show the plot
plt.show()


```
And the out put would be like this:

<img width="571" alt="Screenshot 2023-04-03 at 18 18 02" src="https://user-images.githubusercontent.com/109058050/229568533-4fe067ef-46e9-4a83-8f6f-f6ce58fb868e.png">

