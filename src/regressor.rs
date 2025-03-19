use crate::{dataset::DataSet, model::LinearModelParams};

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
    params: Option<LinearModelParams>,

    max_epochs: usize,
    loss_threshold: f64,
    gradient_threshold: f64,
    learning_rate: f64,
}
impl LinearRegressor {
    /// Creates a new instance of the LinearRegressor.
    pub fn new() -> Self {
        Self {
            max_epochs: 1_000_000,
            loss_threshold: 0.0001,
            gradient_threshold: 0.0001,
            learning_rate: 0.01,
            params: None,
        }
    }

    /// # Switches the model parameters.
    /// This function allows the user to switch the model parameters.
    /// It takes a LinearModelParams instance as input and updates the model parameters.
    pub fn switch(&mut self, params: LinearModelParams) -> Result<(), Error> {
        self.params = Some(params);
        Ok(())
    }

    /// # Fits the model to the given dataset.
    /// This function trains the model using the provided dataset.
    /// It standardizes the input data, computes the gradient, and updates the weights and bias.
    /// The training process continues until the maximum number of epochs is reached,
    /// or the loss and gradient thresholds are met.
    pub fn fit(&mut self, mut data: DataSet) -> Result<LinearModelParams, Error> {
        let mut params = LinearModelParams::new(self.learning_rate, &data);
        params.standarize(&mut data);
        for _ in 0..self.max_epochs {
            let grad = params.gradient(&data);   
            params.update_weights(&grad);

            let norm = grad.norm();
            if norm < self.gradient_threshold {
                break;
            }
            let loss = params.loss(&data);
            if loss < self.loss_threshold {
                break;
            }
        }
        self.params = Some(params.clone());
        Ok(params)
    }

    /// # Predicts the output for the given input data.
    /// This function uses the fitted model to make predictions on new input data.
    /// It checks if the model has been fitted before making predictions.
    /// If the model is not fitted, it returns an error.
    pub fn predict(&self, x: &[Vec<f64>]) -> Result<Vec<f64>, Error> {
        if self.params.is_none() {
            return Err(Error::NotFitted);
        }
        let predictions = x.iter().map(
            |f| self.params.as_ref().unwrap().predict_std(f)
        ).collect::<Vec<_>>();
        Ok(predictions)
    }

    /// # Returns the parameters of the model.
    /// This function returns the parameters of the model, including weights and bias.
    /// If the model is not fitted, it returns an error.
    pub fn params(&self) -> Result<LinearModelParams, Error> {
        if self.params.is_none() {
            return Err(Error::NotFitted);
        }
        Ok(self.params.clone().unwrap())
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
        let train_dt = DataSet::new(x.clone(), y.clone()).unwrap();

        // Test prediction
        let val_x = &vec![
            vec![7.0, -3.1, 2.3], 
            vec![-4.0, 1.3, -2.1]
        ];
        let val_y = vec![
            3. * 7.0 + -1. * -3.1 -6. * 2.3 + 2., 
            3. * -4.0 + -1. * 1.3 -6. * -2.1 + 2.
        ];
        let validation_dt = DataSet::new(val_x.clone(), val_y.clone()).unwrap();
        test_model(train_dt, validation_dt);
    }

    #[test]
    fn test_linear_func() {
        // linear function: y = 2x + 7
        let x = vec![
            vec![1.0], 
            vec![3.0], 
            vec![4.0], 
            vec![5.0], 
            vec![13.0]
        ];
        let y: Vec<f64> = x.iter().map(|params| 2. * params[0] + 7.).collect();
        let train_dt = DataSet::new(x.clone(), y.clone()).unwrap();

        // Test prediction
        let val_x = &vec![vec![7.0], vec![-4.0]];
        let val_y= vec![2. * 7.0 + 7., 2. * -4.0 + 7.];
        let validation_dt = DataSet::new(val_x.clone(), val_y.clone()).unwrap();
        test_model(train_dt, validation_dt);
    }

    #[test]
    fn test_multi_params_func() {
        // linear function: y = 3x1 - 1x2 -6x3 + 2
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
        let train_dt = DataSet::new(x.clone(), y.clone()).unwrap();
        let val_x = &vec![
            vec![7.0, -3.1, 2.3], 
            vec![-4.0, 1.3, -2.1]
        ];
        let val_y = vec![
            3. * 7.0 + -1. * -3.1 -6. * 2.3 + 2., 
            3. * -4.0 + -1. * 1.3 -6. * -2.1 + 2.
        ];
        let validation_dt = DataSet::new(val_x.clone(), val_y.clone()).unwrap();
        test_model(train_dt, validation_dt);
    }

    #[test]
    fn test_not_fitted() {
        let model: LinearRegressor = LinearRegressor::new();
        let x = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        assert_eq!(model.predict(&x), Err(Error::NotFitted));
    }


    fn test_model(train_dt: DataSet, validation_dt: DataSet) {
        let mut model: LinearRegressor = 
            LinearRegressor::new()
                .set_loss_threshold(1e-5)
                .set_gradient_threshold(1e-5)
                .set_max_epochs(1_000_000)
                .set_learning_rate(0.01);
        let _ = model.fit(train_dt.clone()).unwrap();
        println!("model params: {:?}", model.params);

        let preds = model.predict(train_dt.data()).unwrap();
        let loss = mean_squared_error(&preds, &train_dt.target());
        println!("predictions: {:?}", preds);
        println!("expected predictions: {:?}", train_dt.target());
        println!("prediction loss: {}", loss);
        assert!(loss < 1e-4);

        // Test prediction
        let predictions = model.predict(&validation_dt.data()).unwrap();
        let prediction_loss = mean_squared_error(&predictions, &validation_dt.target());
        println!("predictions: {:?}", predictions);
        println!("expected predictions: {:?}", validation_dt.target());
        println!("prediction loss: {}", prediction_loss);
        assert!(prediction_loss < 1e-3);
    }
}