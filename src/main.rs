use simple_linear_regressor::{dataset::DataSet, regressor, utils::mean_squared_error};

fn main() {

    let features = [
        "female",
        "male",
        "age",
        "bmi",
        "children",
        "smoker",
        "northeast",
        "northwest",
        "southeast",
        "southwest",
    ];
    let dataset = DataSet::from_csv("./insurance_cleaned.csv", &features, "charges").unwrap();
    let (train_data, validation_data) = dataset.split(Some(157), 0.8).unwrap();
    let mut regressor = regressor::LinearRegressor::new()
        .set_gradient_threshold(0.000001)
        .set_learning_rate(0.01)
        .set_loss_threshold(0.000001)
        .set_max_epochs(1_000_000);
    let params = regressor.fit(train_data.clone()).unwrap();
    println!("Model parameters: {:?}", params);

    // Predict on the training data
    let predictions = regressor.predict(train_data.data()).unwrap();
    let target = train_data.target();
    let loss = mean_squared_error(&target, &predictions);
    println!("Training MSE: {:?}", loss);

    // Predict on the validation data
    let predictions = regressor.predict(validation_data.data()).unwrap();
    let target = validation_data.target();
    let loss = mean_squared_error(&target, &predictions);
    println!("Validation MSE: {:?}", loss);

    // let predictions = regressor.predict(&[
    //     vec![0.0, 1.0, 19.0, 27.9, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
    //     vec![1.0, 0.0, 64.0, 38.8, 3.0, 1.0, 1.0, 1.0, 0.0, 1.0],
    // ]).unwrap();
}
