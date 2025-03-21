# Simple Linear Regression model from scratch
A pure Rust implementation of a simple Linear Regressor with standardization.

This is my implementation of a Linear Regression model without using any machine learning libraries. The only dependencies used are `csv` and `rand`.

## Topics Covered
- [How It Works](#how-it-works)
  - [Data Standardization](#data-standardization)
  - [Gradient Descent](#gradient-descent)
- [Performance](#performance)
- [Error Rate](#error-rate)
- [Comparison](#comparison)

## How It Works

### Data Standardization
The main `fit` method takes a set of samples $X$ and target values $y$. Each sample is an array of features, and each target value in $y$ corresponds to a sample in $X$ at the same index. First, we validate the data to ensure it is consistent, i.e., the number of rows in $X$ matches the number of targets in $y$, and each sample has the same number of features. Then, we standardize the data as follows: 

$$x' = \frac{x - \mu}{\sigma}$$
Where $\mu$ is the mean of a given feature, and $\sigma$ is the standard deviation of the feature:
$$
    \sigma = \sqrt{\frac{1}{N} \cdot \sum_{i=1}^N (x_i - \mu)^2}
$$

This step is crucial because unstandardized data with high variance can lead to incorrect updates in the gradient descent algorithm.

### Gradient Descent
To compute $w_1, w_2, ..., w_N$, I use the gradient descent algorithm. The algorithm iterates over the data and computes the gradient for all samples until one of the following conditions is met:
- The maximum number of epochs (`MAX_EPOCH`) is reached.
- The loss falls below an acceptable threshold.
- The gradient norm falls below a threshold.

The gradient is a vector of partial derivatives of the loss function with respect to each weight. The first step in each iteration is to compute the partial derivative of the loss function:

$$
    \frac{\partial L}{\partial f} 
    = \frac{2}{N} \cdot \sum_{i = 1}^N (w_1 * x_1 + ... + \text{bias} - t_i)
$$

Using the chain rule, I compute the other partial derivatives for the gradient:

$$ 
    \frac{\partial L}{\partial w_i} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial w_i}
$$

Since the function is a linear combination, the derivative of each parameter is simply the parameter itself:

$$ 
    \frac{\partial z}{\partial w_i} = w_i 
$$

This results in the gradient:

$$
\nabla L = \left(
    \frac{\partial L}{\partial w_1}, 
    \frac{\partial L}{\partial w_2}, 
    \frac{\partial L}{\partial w_3}, 
    ...,
    \frac{\partial L}{\partial w_N}
    \right)
$$
Finally, the weights are updated as follows:
$$w_1' = w_1 - \eta \cdot \frac{\partial L}{\partial w_1}$$
$$w_2' = w_2 - \eta \cdot \frac{\partial L}{\partial w_2}$$
$$ ... $$
$$w_N' = w_N - \eta \cdot \frac{\partial L}{\partial w_N}$$

## Performance
As a demonstration and a large-scale test, I used a dataset of insurance costs from Kaggle. The dataset contains 1,338 rows of data. From visual analysis, the most significant features are:

### Age
![age-impact](/images/age-impact.png)

### BMI
![bmi-impact](/images/bmi-impact.png)

### Smoking Status
![smoking-impact](/images/smoking-impact.png)

The model correctly identified these as major factors, and their coefficients are significantly higher than those of other features.

## Error Rate
The Mean Squared Error (MSE) on the validation data is approximately 37,000,000, resulting in a relative error of around 40%:

$$ \text{Relative Error (\\%)} = \frac{\sqrt{MSE}}{\text{Mean of the Target Variable}} \cdot 100$$

This error might be due to the dataset being relatively small, containing many outliers, and having unevenly distributed data. However, the performance is comparable to other linear regression models.

### Comparison
The performance is identical to existing Linear Regression models, such as the one provided by the `scikit-learn` library in Python.
