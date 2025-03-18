# Simple LinearRegressor
Pure rust implemantation of simple LinearRegressor with standarization for overfitting reduction. 



## How it works

### Data standarization
Main fit method takes set of samples $X$ and target values $y$. Each sample is an array of so called features and has assigned its target in $y$ under the same index. We firstly check if data is correct eg. has same number of rows as target and each sample has same number of features. Then we standarize the data.
$$
    x' = \frac{x - \mu}{\sigma}
$$
Where $\mu$ is mean of given feature and $\sigma$ represents standard deviation of the feature
$$
    \sigma = \sqrt{\frac{1}{N} \cdot \sum_{i=1}^N (x_i - \mu)^2}
$$

It is very important step, because unstandarized data with high variance can lead to incorrect steps in gradient descent algorithm.

### Gradient descent
For computing $w_1, w_2, ..., w_N$ i use gradient descent algorithm. I loop over and compute gradient over all samples until one of below happens
- i loop over MAX_EPOCH times
- loss is below the acceptable treshold
- gradient norm is below the treshold

Gradient is a vector of partial derivatiwes of loss to each weight so first step in each loop is to compute loss function partial derivatiwe.

$$
    \frac{\partial L}{\partial f} 
    = \frac{2}{N} \cdot \sum_{i = 1}^N (w_1 * x_1 + ... + bias - t_i)
$$
And than from this using the chain rule i compute other partial derivatiwes for gradient
$$ 
    \frac{\partial L}{\partial w_i} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial w_i}
$$
because the function is linear combination derivatiwe of each parameter is just the parameter
$$ 
    \frac{\partial z}{\partial w_i} = w_i 
$$
It leads us to the gradient
$$
\nabla L = \left(
    \frac{\partial L}{\partial w_1}, 
    \frac{\partial L}{\partial w_2}, 
    \frac{\partial L}{\partial w_3}, 
    ...,
    \frac{\partial L}{\partial w_N}
    \right)
$$
And finaly the weight updates
$$w_1' = w_1 - \eta \cdot \frac{\partial L}{\partial w_1}$$
$$w_2' = w_2 - \eta \cdot \frac{\partial L}{\partial w_2}$$
$$ ... $$
$$w_N' = w_N - \eta \cdot \frac{\partial L}{\partial w_N}$$