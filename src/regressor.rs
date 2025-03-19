use std::vec;

use crate::utils::mean_squared_error;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    InvalidInput,
    NotFitted,
    AlreadyFitted,
}

/// A simple linear regression model using gradient descent.
/// This model can be used to fit a linear function to a set of input data and target values.
/// The model is trained using gradient descent, and the weights and bias are updated iteratively.
/// The model can be used to make predictions on new input data after it has been fitted.
pub struct LinearRegressor {
    bias: f64,
    
    weights: Vec<f64>,
    features_mean: Vec<f64>,
    features_std: Vec<f64>,

    max_epochs: usize,
    loss_threshold: f64,
    gradient_threshold: f64,
    learning_rate: f64,

    fitted: bool,
}
impl LinearRegressor {
    /// Creates a new instance of the LinearRegressor.
    pub fn new() -> Self {
        Self {
            weights: vec![],
            features_mean: vec![],
            features_std: vec![],
            bias: 0.0,
            max_epochs: 1_000_000,
            loss_threshold: 0.0001,
            gradient_threshold: 0.0001,
            learning_rate: 0.01,
            fitted: false,
        }
    }

    /// Fits the model to the given input data and target values.
    /// # Arguments
    /// * `x` - A 2D vector of input data.
    /// * `y` - A vector of target values.
    /// # Returns
    /// The final loss value.
    /// # Errors
    /// Returns an error if the model is already fitted or if the input data is invalid.
    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) -> Result<f64, Error> { 
        if self.fitted {
            return Err(Error::AlreadyFitted);
        }
        if x.is_empty() || y.is_empty() {
            return Err(Error::InvalidInput);
        }
        let n_samples = x.len();
        let n_features = x[0].len();
        if (y.len() != n_samples) || (x.iter().any(|row| row.len() != n_features)) {
            return Err(Error::InvalidInput);
        }
        let standardized_x = self.standarize(x);
        self.weights = vec![1.0; n_features];

        // Gradient descent loop
        for _ in 0..self.max_epochs {

            // Derivatives of prediction function (+ 1 for bias)
            let mut gradient: Vec<f64> = vec![0.0; n_features + 1];
            
            // Loss function
            let mut loss = 0.0; 
            for (j, sample) in standardized_x.iter().enumerate() {
                let target: f64 = y[j];

                // Prediction
                let mut pred: f64 = self.bias;
                for k in 0..n_features {
                    pred += self.weights[k] * sample[k];
                }
                
                loss += (pred - target) * (pred - target);
                
                // Partial derivative of the loss function
                let ldw: f64 = 2. * (pred - target);

                // Update gradient
                for k in 0..n_features {
                    gradient[k] += ldw * sample[k];
                }
                gradient[n_features] += ldw;
            }
            if loss < self.loss_threshold {
                break;
            }
            
            // Update gradient & compute its norm
            let mut gradient_norm: f64 = 0.0;
            for k in 0..=n_features {
                gradient[k] /= n_samples as f64;
                gradient_norm += gradient[k] * gradient[k];
            }
            gradient_norm = gradient_norm.sqrt();            
            if gradient_norm < self.gradient_threshold {
                break;
            }

            // Update weights and bias
            for k in 0..n_features {
                self.weights[k] -= self.learning_rate * gradient[k];
            }
            self.bias -= self.learning_rate * gradient[n_features];
        }

        
        // Compute final loss
        self.fitted = true; 
        let predictions = self.predict(x).unwrap();
        let loss = mean_squared_error(&predictions, &y);
        
        return Ok(loss);        
    }

    /// Predicts the output for the given input data.
    /// # Arguments
    /// * `x` - A 2D vector of input data.
    /// # Returns
    /// A vector of predicted values.
    pub fn predict(&self, x: &[Vec<f64>]) -> Result<Vec<f64>, Error> {
        if !self.fitted {
            return Err(Error::NotFitted);
        }
        let mut predictions: Vec<f64> = vec![];
        let standardized_x = x.iter().map(|sample| {
            let mut standardized = vec![0.0; sample.len()];
            for (i, &value) in sample.iter().enumerate() {
                standardized[i] = (value - self.features_mean[i]) / self.features_std[i];
            }
            standardized
        }).collect::<Vec<_>>();
        for inp in standardized_x.iter() {
            let mut pred: f64 = self.bias;
            for (i, &weight) in self.weights.iter().enumerate() {
                pred += weight * inp[i];
            }
            predictions.push(pred);      
        }
        Ok(predictions)
    }


    /// TODO change how it works one function does too much
    fn standarize(&mut self, x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        self.features_mean = vec![0.0; x[0].len()];
        self.features_std = vec![0.0; x[0].len()];
        for sample in x.iter() {
            for (j, &value) in sample.iter().enumerate() {
                self.features_mean[j] += value;
                self.features_std[j] += value * value;
            }
        }
        for j in 0..self.features_mean.len() {
            self.features_mean[j] /= x.len() as f64;
            self.features_std[j] = (self.features_std[j] / x.len() as f64 - self.features_mean[j] * self.features_mean[j]).sqrt();
        }

        let mut standardized = vec![vec![0.0; x[0].len()]; x.len()];
        for (i, sample) in x.iter().enumerate() {
            for (j, &value) in sample.iter().enumerate() {
                standardized[i][j] = (value - self.features_mean[j]) / self.features_std[j];
            }
        }
        standardized
    }

    /// Returns the weights of the model.
    /// # Returns
    /// A vector of weights.
    pub fn get_weights(&self) -> Result<Vec<f64>, Error> {
        if !self.fitted {
            return Err(Error::NotFitted);
        }
        Ok(self.weights.clone())
    }

    /// Returns the bias of the model.
    /// # Returns
    /// The bias value.
    pub fn get_bias(&self) -> Result<f64, Error> {
        if !self.fitted {
            return Err(Error::NotFitted);
        }
        Ok(self.bias)
    }

    /// Sets the maximum number of epochs for training.
    pub fn set_max_epochs(mut self, max_epochs: usize) -> Self {
        self.max_epochs = max_epochs;
        return self;
    }

    /// Sets the loss threshold for training.
    pub fn set_loss_threshold(mut self, loss_threshold: f64) -> Self {
        self.loss_threshold = loss_threshold;
        return self;
    }

    /// Sets the gradient threshold for training.
    pub fn set_gradient_threshold(mut self, gradient_threshold: f64) -> Self {
        self.gradient_threshold = gradient_threshold;
        return self;
    }

    /// Sets the learning rate for training.
    pub fn set_learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        return self;
    }
}


#[cfg(test)]
pub mod tests {
    use crate::utils::mean_squared_error;

    use super::*;


    #[test]
    fn test_high_variance_multi_params_func() {
        // linear function: y = 3x1 - 1x2 -6x3 + 2
        let mut model: LinearRegressor = 
            LinearRegressor::new()
                .set_loss_threshold(1e-5)
                .set_gradient_threshold(1e-5)
                .set_max_epochs(1_000_000)
                .set_learning_rate(0.01);
        let x = vec![
            vec![100.0, -3.2, -2.0], 
            vec![-300.0, 2000.5, 13.0], 
            vec![5647.0, 1.0, -30.0], 
            vec![-20072.0, 523.0, 142.5], 
            vec![13.0, -123.0, 1321.0]
        ];
        let y: Vec<f64> = x.iter()
            .map(|params| 3. * params[0] - 1. * params[1] - 6. * params[2] + 2.)
            .collect();
        let loss = model.fit(&x, &y).unwrap();
        let weights = model.get_weights().unwrap();
        let bias = model.get_bias().unwrap();
        println!("test params: [3, 1, -6, 2]");
        println!("weights: {:?}", weights);
        println!("bias: {}", bias);
        println!("loss: {}", loss);
        assert!(loss < 1e-4);
        
        // Test prediction
        let predictions = model.predict(&vec![vec![7.0, -3.1, 2.3], vec![-4.0, 1.3, -2.1]]).unwrap();
        let expected_predictions = vec![
            3. * 7.0 + -1. * -3.1 -6. * 2.3 + 2., 
            3. * -4.0 + -1. * 1.3 -6. * -2.1 + 2.
        ];
        let prediction_loss = mean_squared_error(&predictions, &expected_predictions);
        println!("predictions: {:?}", predictions);
        println!("expected predictions: {:?}", expected_predictions);
        println!("prediction loss: {}", prediction_loss);
        assert!(prediction_loss < 1e-3);

    }

    #[test]
    fn test_linear_func() {
        // linear function: y = 2x + 7
        let mut model: LinearRegressor = 
            LinearRegressor::new()
                .set_loss_threshold(1e-5)
                .set_learning_rate(0.01);
        let x = vec![vec![1.0], vec![3.0], vec![4.0], vec![5.0], vec![13.0]];
        let y: Vec<f64> = x.iter().map(|params| 2. * params[0] + 7.).collect();
        let loss = model.fit(&x, &y).unwrap();
        let weights = model.get_weights().unwrap();
        let bias = model.get_bias().unwrap();
        println!("test params: [2, 7]");
        println!("weights: {:?}", weights);
        println!("bias: {}", bias);
        println!("loss: {}", loss);
        assert!(loss < 1e-4);
        
        // Test prediction
        let predictions = model.predict(&vec![vec![7.0], vec![-4.0]]).unwrap();
        let expected_predictions = vec![2. * 7.0 + 7., 2. * -4.0 + 7.];
        let prediction_loss = mean_squared_error(&predictions, &expected_predictions);
        println!("predictions: {:?}", predictions);
        println!("expected predictions: {:?}", expected_predictions);
        println!("prediction loss: {}", prediction_loss);
        assert!(prediction_loss < 1e-4);
    }

    #[test]
    fn test_multi_params_func() {
        // linear function: y = 3x1 - 1x2 -6x3 + 2
        let mut model: LinearRegressor = 
            LinearRegressor::new()
                .set_loss_threshold(1e-5)
                .set_gradient_threshold(1e-5)
                .set_max_epochs(1_000_000)
                .set_learning_rate(0.01);
        let x = vec![
            vec![1.0, 3.2, -2.0], 
            vec![3.0, 2.5, 0.0], 
            vec![4.0, 1.0, -3.0], 
            vec![-2.0, 5.0, 1.5], 
            vec![1.0, -13.0, 10.0]
        ];
        let y: Vec<f64> = x.iter()
            .map(|params| 3. * params[0] - 1. * params[1] - 6. * params[2] + 2.)
            .collect();
        let loss = model.fit(&x, &y).unwrap();
        let weights = model.get_weights().unwrap();
        let bias = model.get_bias().unwrap();
        println!("test params: [3, 1, -6, 2]");
        println!("weights: {:?}", weights);
        println!("bias: {}", bias);
        println!("loss: {}", loss);
        assert!(loss < 1e-4);
        
        // Test prediction
        let predictions = model.predict(&vec![vec![7.0, -3.1, 2.3], vec![-4.0, 1.3, -2.1]]).unwrap();
        let expected_predictions = vec![
            3. * 7.0 + -1. * -3.1 -6. * 2.3 + 2., 
            3. * -4.0 + -1. * 1.3 -6. * -2.1 + 2.
        ];
        let prediction_loss = mean_squared_error(&predictions, &expected_predictions);
        println!("predictions: {:?}", predictions);
        println!("expected predictions: {:?}", expected_predictions);
        println!("prediction loss: {}", prediction_loss);
        assert!(prediction_loss < 1e-3);

    }
    
    #[test]
    fn test_invalid_input() {
        let mut model: LinearRegressor = LinearRegressor::new();
        let x = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let y = vec![1.0, 2.0, 3.0];
        assert_eq!(model.fit(&x, &y), Err(Error::InvalidInput));
    }

    #[test]
    fn test_already_fitted() {
        let mut model: LinearRegressor = LinearRegressor::new();
        let x = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let y = vec![1.0, 2.0];
        model.fit(&x, &y).unwrap();
        assert_eq!(model.fit(&x, &y), Err(Error::AlreadyFitted));
    }

    #[test]
    fn test_not_fitted() {
        let model: LinearRegressor = LinearRegressor::new();
        let x = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        assert_eq!(model.predict(&x), Err(Error::NotFitted));
    }

    #[test]
    fn test_get_weights_not_fitted() {
        let model: LinearRegressor = LinearRegressor::new();
        assert_eq!(model.get_weights(), Err(Error::NotFitted));
    }

    #[test]
    fn test_get_bias_not_fitted() {
        let model: LinearRegressor = LinearRegressor::new();
        assert_eq!(model.get_bias(), Err(Error::NotFitted));
    }
}