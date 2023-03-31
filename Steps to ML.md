## Reference books:
  - G. Strang, Linear Algebra and Its Applications, Academic Press 1980
  - I. Goodfellow, Y. Bengio and A. Courville, Deep Learning, MIR Press 2016
  - S. Boyd, Convex Optimization, Cambridge University Press 2004
### Linear Algebra:
We know that:
<img width="802" alt="Screenshot 2023-03-31 at 12 37 05" src="https://user-images.githubusercontent.com/109058050/229098029-a0f88992-331f-462d-9e56-686520642550.png">

### Standard Norms and P-Norms:
The most well-known and widely used norm is Euclidean norm:

\inmath ||x||_2 = \sqrt \Sigma | x_i | ^2

which corresponds to the distance in our real life (the vectors might have complex elements, thus is the modulus here).
Euclidean norm, or P-norm, is a subclass of an important class of P-norms:


\inmath ||x||_p = {\sqrt {\Sigma | x_i | ^p}}^{1/p}

  - There are two very important special cases:
    - Infinity norm, or Chebyshev norm which is defined as the maximal element
    - \inmath L_1 norm (or Manhattan distance) which is defined as the sum of modules of the elements of \inmath x_i
With numpy we can compute the norms `np.linalg.norm function`:

```python
import numpy as np
n = 100
a = np.ones(n)
print(a)
print(np.linalg.norm(a, 1)) # L1 norm
print(np.linalg.norm(a, 2)) # L2 norm
print(np.linalg.norm(a, np.inf))
b = a + 1e-3 * np.random.randn(n)
print(b)
print()
print('Relative error:',
np.linalg.norm(a - b, np.inf) / np.linalg.norm(b, np.inf))
```
and the output would be like:

```lua
[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
1. 1. 1. 1.]
100.0
10.0
1.0
[1.00037435 0.99953124 0.99992333 1.0014888 1.00032329 1.00050046
1.00078602 0.99924418 0.99969855 1.00210667 1.00094794 1.00080644
0.9995451 0.9973272 0.99983952 1.00145786 1.00006065 1.00228847
1.00024384 0.99927576 1.00074415 1.00052594 1.00042917 1.00072768
0.99929429 1.00141766 1.00173564 0.99915889 1.00009986 0.99906971
0.99996623 0.99950146 0.99925539 1.00130413 1.00083764 0.99949217
0.99928815 0.99995427 1.00000749 0.9989179 0.99995726 0.99777181
1.00036878 1.00070422 0.99823925 1.00020147 0.99977514 0.99915426
0.99980286 1.00108473 0.99983239 1.00131403 1.00066266 1.00175787
0.99936652 1.00029688 1.0008256 1.00069293 0.99890104 1.00069938
1.00032954 0.9997104 1.00069634 1.0006385 0.9987517 0.99944188
0.99896297 1.00068079 0.99903867 1.00045222 1.00141346 1.00191724
0.99962728 0.99963954 0.99829915 1.00199422 1.00041936 1.0002962
0.9986163 1.00145062 1.00210623 0.99923621 1.0001467 1.00030778
1.00013833 0.9994478 0.99868239 1.00042074 0.99946207 0.99804034
0.99985559 1.00228056 1.00085015 1.00133834 0.99957209 1.00028839
1.0024329 1.00053665 1.00051172 1.00138696]
Relative error: 0.002666316967426874
```
