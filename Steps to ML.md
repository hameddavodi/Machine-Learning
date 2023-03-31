## Reference books:
  - G. Strang, Linear Algebra and Its Applications, Academic Press 1980
  - I. Goodfellow, Y. Bengio and A. Courville, Deep Learning, MIR Press 2016
  - S. Boyd, Convex Optimization, Cambridge University Press 2004
## Application: Anomaly Detection:
### Multi-variate Normal (MVN):

This is a Python code that performs anomaly detection on a dataset using multivariate Gaussian distribution.

The first part of the code reads the dataset from a CSV file, prints the number of datapoints and dimensions, and plots the dataset on a scatter plot.

The second part of the code defines two functions: "estimateGaussian" and "multivariateGaussian". The "estimateGaussian" function calculates the mean and covariance of the dataset, while the "multivariateGaussian" function calculates the probability density function of a multivariate Gaussian distribution.

The third part of the code uses the "estimateGaussian" function to calculate the mean and covariance of the dataset, and the "multivariateGaussian" function to calculate the probability density of each datapoint in the dataset. It then sets a threshold for anomaly detection and determines the outliers/anomalies in the dataset.

The last part of the code plots the dataset on a scatter plot, with the outliers/anomalies marked in red.
```python
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
from scipy.stats import multivariate_normal
def read_dataset(filePath,delimiter=','):
return genfromtxt(filePath, delimiter=delimiter)
tr_data = read_dataset('Data/anomaly_detect_data.csv')
n_training_samples = tr_data.shape[0]
n_dim = tr_data.shape[1]
print('Number of datapoints in training set: %d' % n_training_samples)
print('Number of dimensions/features: %d' % n_dim)
print(tr_data[1:5,:])
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)')
plt.plot(tr_data[:,0],tr_data[:,1],'bx')
plt.show()
def estimateGaussian(dataset):
mu = np.mean(dataset, axis=0) # mean along each dimension / column
sigma = np.cov(dataset.T)
return mu, sigma
def multivariateGaussian(dataset,mu,sigma):
p = multivariate_normal(mean=mu, cov=sigma)
return p.pdf(dataset)
mu, sigma = estimateGaussian(tr_data)
p = multivariateGaussian(tr_data,mu,sigma)
thresh = 9e-05
# determining outliers/anomalies
outliers = np.asarray(np.where(p < thresh))
outliers
plt.figure()
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)')
plt.plot(tr_data[:,0],tr_data[:,1],'bx')
plt.plot(tr_data[outliers,0],tr_data[outliers,1],'ro')
plt.show()
```
and the results are:
```text
Number of datapoints in training set: 307
Number of dimensions/features: 2
[[13.409 13.763]
[14.196 15.853]
[14.915 16.174]
[13.577 14.043]]
```

<img width="577" alt="Screenshot 2023-03-31 at 15 26 45" src="https://user-images.githubusercontent.com/109058050/229132732-c40c3fff-b6d9-42f4-b195-d077ce78d709.png">

```text
array([[300, 301, 303, 304, 305, 306]], dtype=int64)
```
<img width="581" alt="Screenshot 2023-03-31 at 15 27 18" src="https://user-images.githubusercontent.com/109058050/229132851-b5197036-a50b-4c8a-9090-c5c86e5cdb57.png">


