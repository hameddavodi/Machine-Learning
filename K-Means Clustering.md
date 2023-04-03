## What is K-means:

K-means clustering is a type of unsupervised learning algorithm used in machine learning and data mining to partition a set of data points into a fixed number (k) of clusters. The goal of k-means clustering is to group the data points in such a way that the points within a group (cluster) are similar to each other and different from the points in other groups.

The k-means algorithm starts by randomly selecting k initial cluster centers, and then iteratively assigns each data point to the nearest cluster center, and re-computes the center of each cluster as the mean of the points assigned to it. The process is repeated until the cluster centers converge, that is, until the assignment of data points to clusters no longer changes significantly.

The k-means algorithm has several advantages, including its simplicity, computational efficiency, and effectiveness in handling large datasets. However, it also has some limitations, such as the sensitivity to the initial choice of cluster centers and the assumption that the clusters are spherical and have equal variance.

Note: K-means clustering has a wide range of applications, including image segmentation, customer segmentation, anomaly detection, and document clustering.

Let's start by importing necessary libraries:

```python 
from sklearn.clusters import KMeans
```
Next, let’s define the inputs we will use for our K-means clustering algorithm. For example age and spending score:

```python 
X = df[['Age', 'Spending Score (1-100)']].copy()

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(X)
    
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.intertia_)
 
import matplotlib.pyplot as plt
import seaborn as sns 


sns.set()

plt.plot(range(1, 11), wcss)

plt.title('Selecting the Numbeer of Clusters using the Elbow Method')

plt.xlabel('Clusters')
plt.ylabel('WCSS')
plt.show()

```
The first for loop iterates over values from 1 to 10, and for each value i, it creates a new KMeans object with i clusters and fits it to the data X. However, this loop does not save any information about the results of each KMeans object.

The second for loop is very similar to the first one, but it adds an extra step of appending the within-cluster sum of squares (WCSS) value to a list called wcss. The WCSS is a measure of the variance within each cluster, and it is used to evaluate the quality of the clustering. By iterating over different numbers of clusters and calculating the WCSS for each one, we can use the elbow method to determine the optimal number of clusters to use.
The `sns.set()` function sets the default style for the plot to use.

The `plt.plot()` function creates a line plot of the WCSS values against the number of clusters used. The range(1, 11) specifies the values for the x-axis (i.e., the number of clusters), and wcss specifies the values for the y-axis (i.e., the WCSS values).

The `plt.title()`, `plt.xlabel()`, and `plt.ylabel()` functions add a title and labels to the plot.

The `plt.show()` function displays the plot.

