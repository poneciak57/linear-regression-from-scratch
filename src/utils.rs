

/// Mean Squared Error (MSE) calculation
/// This function calculates the Mean Squared Error (MSE) between two slices of f64.
/// It is commonly used as a loss function in regression problems.
pub fn mean_squared_error(y_true: &[f64], y_pred: &[f64]) -> f64 {
    if y_true.len() != y_pred.len() {
        panic!("Length of y_true and y_pred must be the same");
    }
    let sum_squared_error: f64 = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(y_t, y_p)| (y_t - y_p).powi(2))
        .sum();
    sum_squared_error / y_true.len() as f64
}




#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean_squared_error() {
        let y_true = vec![1.0, 2.0, 3.0];
        let y_pred = vec![1.5, 2.5, 3.5];
        let mse = mean_squared_error(&y_true, &y_pred);
        assert_eq!(mse, 0.25);
    }

    #[test]
    #[should_panic(expected = "Length of y_true and y_pred must be the same")]
    fn test_mean_squared_error_length_mismatch() {
        let y_true = vec![1.0, 2.0];
        let y_pred = vec![1.5, 2.5, 3.5];
        mean_squared_error(&y_true, &y_pred);
    }
}