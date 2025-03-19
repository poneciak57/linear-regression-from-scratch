
pub struct Gradient {
    pub grad: Vec<f64>,
    pub bias_grad: f64,
}

impl Gradient {
    /// # Creates a new Gradient instance.
    pub fn new(bias_grad: f64, grad: Vec<f64>) -> Self {
        Gradient {
            bias_grad,
            grad
        }
    }

    /// # Computes the norm of the gradient.
    /// This function calculates the Euclidean norm of the gradient vector.
    /// It is used to measure the magnitude of the gradient.
    pub fn norm(&self) -> f64 {
        (self.grad.iter().map(|&x| x * x).sum::<f64>() + self.bias_grad * self.bias_grad).sqrt()
    }

}
