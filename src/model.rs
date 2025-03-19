use crate::{dataset::DataSet, gradient::Gradient};


#[derive(Clone, Debug)]
pub struct LinearModelParams { 
    pub weights: Vec<f64>,
    pub bias: f64,

    pub learning_rate: f64,
    pub mean: Vec<f64>,
    pub std: Vec<f64>,
}

impl LinearModelParams {
    
    /// # Creates a new LinearModelParams instance.
    /// This function initializes the weights and bias to zero.
    /// It also computes the mean and standard deviation of the dataset.
    pub fn new(learning_rate: f64, data: &DataSet) -> Self {
        let num_features = data.data()[0].len();
        let mean = Self::compute_mean(data.data());
        let std = Self::compute_std(data.data(), &mean);
        LinearModelParams {
            learning_rate,
            weights: vec![0.; num_features],
            bias: 0.0,
            mean, std,
        }
    }

    /// # Predicts the target value for a given set of features.
    /// This function calculates the dot product of the features and weights,
    /// and adds the bias to get the predicted value.
    pub fn predict(&self, features: &Vec<f64>) -> f64 {
        let mut prediction = self.bias;
        for (i, feature) in features.iter().enumerate() {
            prediction += self.weights[i] * feature; 
        }
        prediction
    }

    /// # Predicts the target value for a given set of features with standarization.
    /// This function calculates the dot product of the features and weights,
    /// and adds the bias to get the predicted value.
    /// It also standardizes the features using the mean and standard deviation.
    /// This is useful for making predictions on new data that may not be in the same scale as the training data.
    pub fn predict_std(&self, features: &Vec<f64>) -> f64 {
        let mut prediction = self.bias;
        for (i, feature) in features.iter().enumerate() {
            let standarized_feature = (feature - self.mean[i]) / self.std[i];
            prediction += self.weights[i] * standarized_feature; 
        }
        prediction
    }


    /// # Computes gradient of the loss function.
    /// This function calculates the gradient of the loss function with respect to the weights and bias.
    /// It iterates over the dataset, computes the prediction for each sample,
    /// and updates the gradient using chain rule.
    pub fn gradient(&self, data: &DataSet) -> Gradient {
        let mut grad = Gradient::new(0., vec![0.; self.weights.len()]);
        for s in data {
            let pred = self.predict(&s.features());
            let loss_derivative = 2. * (pred - s.target());
            for (i, feature) in s.features().iter().enumerate() {
                grad.grad[i] += loss_derivative * feature;
            }
            grad.bias_grad += loss_derivative;
        }
        for feature in grad.grad.iter_mut() {
            *feature /= data.data().len() as f64;
        }
        grad.bias_grad /= data.data().len() as f64;
        grad
    }

    /// # Standardizes the dataset.
    /// This function standardizes the dataset by subtracting the mean and dividing by the standard deviation.
    /// It modifies the dataset in place.
    pub fn standarize(&self, data: &mut DataSet) {
        for value in data.data_mut().iter_mut() {
            for (i, feature) in value.iter_mut().enumerate() {
                *feature = (*feature - self.mean[i]) / self.std[i];
            }
        }
    }

    /// # Computes the loss of the model.
    /// This function calculates the mean squared error between the predicted and actual target values.
    /// It iterates over the dataset, computes the prediction for each sample,
    /// and accumulates the squared difference.
    /// Finally, it divides the total loss by the number of samples.
    /// The loss is used to evaluate the performance of the model.
    /// A lower loss indicates a better fit to the data.
    pub fn loss(&self, data: &DataSet) -> f64 {
        let mut loss = 0.;
        for s in data {
            let pred = self.predict(&s.features());
            loss += (pred - s.target()).powi(2);
        }
        loss / data.data().len() as f64
    }

    /// # Updates the weights of the model.
    /// This function updates the weights and bias of the model using the gradient.
    /// It multiplies the gradient by the learning rate and subtracts it from the weights.
    /// The bias is updated separately.
    pub fn update_weights(&mut self, gradient: &Gradient) {
        for (w, g) in self.weights.iter_mut().zip(gradient.grad.iter()) {
            *w -= self.learning_rate * g;
        }
        self.bias -= self.learning_rate * gradient.bias_grad;
    }

    /// # Computes the standard deviation of the dataset.
    /// This function calculates the standard deviation for each feature in the dataset.
    fn compute_std(data: &Vec<Vec<f64>>, mean: &Vec<f64>) -> Vec<f64> {
        let mut std = vec![0.; data[0].len()];
        for row in data {
            for (i, &value) in row.iter().enumerate() {
                std[i] += (value - mean[i]) * (value - mean[i]);
            }
        }
        for i in std.iter_mut() {
            *i = (*i / (data.len() as f64 - 1.)).sqrt();
        }
        std
    }

    /// # Computes the mean of the dataset.
    /// This function calculates the mean for each feature in the dataset.
    fn compute_mean(data: &Vec<Vec<f64>>) -> Vec<f64> {
        let mut mean = vec![0.; data[0].len()];
        for row in data {
            for (i, &value) in row.iter().enumerate() {
                mean[i] += value;
            }
        }
        for i in mean.iter_mut() {
            *i /= data.len() as f64;
        }
        mean
    }


}