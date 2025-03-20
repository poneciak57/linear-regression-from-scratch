use rand::{seq::SliceRandom, SeedableRng};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataSetError {
    EmptyDataSet,
    InvalidTestSize,
    IncompatibleDimensions,
    IncorrectFeatureLengths,
    CsvError,
    IncorrectCSVHeader,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DataSet {
    data: Vec<Vec<f64>>,
    target: Vec<f64>,
}

impl DataSet {

    /// # Creates a new DataSet instance.
    /// This function initializes a DataSet with the provided data and target values.
    /// # Arguments
    /// * `data` - A 2D vector representing the features of the dataset.
    /// * `target` - A 1D vector representing the target values.
    /// # Returns
    /// A Result containing the DataSet instance or an error.
    /// # Errors
    /// This function will return an error if the dataset is empty or if the dimensions of data and target are incompatible.
    pub fn new(data: Vec<Vec<f64>>, target: Vec<f64>) -> Result<Self, DataSetError> {
        if data.is_empty() || target.is_empty() {
            return Err(DataSetError::EmptyDataSet);
        }
        if data.len() != target.len() {
            return Err(DataSetError::IncompatibleDimensions);
        }
        for row in &data {
            if row.len() != data[0].len() {
                return Err(DataSetError::IncorrectFeatureLengths);
            }
        }
        Ok(DataSet { data, target })
    }

    /// # Splits the dataset into training and testing sets.
    /// The `random_seed` parameter allows for reproducibility of the split.
    /// The `test_size` parameter determines the proportion of the dataset to include in the test split.
    /// For example, if `test_size` is 0.2, then 20% of the data will be used for testing and 80% for training.
    /// Returns a tuple of two DataSet objects: (train_set, test_set).
    /// # Example
    /// ```
    /// use your_crate::dataset::DataSet;
    /// let data = vec![
    ///     vec![1.0, 2.0],
    ///     vec![3.0, 4.0],
    ///     vec![5.0, 6.0],
    ///     vec![7.0, 8.0],
    /// ];
    /// let target = vec![1.0, 2.0, 3.0, 4.0];
    /// let dataset = DataSet::new(data, target);
    /// let (train_set, test_set) = dataset.split(Some(42), 0.5);
    /// assert_eq!(train_set.data.len(), 2);
    /// assert_eq!(test_set.data.len(), 2);
    /// ```
    /// # Errors
    /// This function will return an error if the dataset is empty or if the `test_size` is not between 0.0 and 1.0.
    pub fn split(&self, random_seed: Option<u64>, test_size: f64) -> Result<(DataSet, DataSet), DataSetError> {
        if test_size <= 0.0 || test_size >= 1.0 {
            return Err(DataSetError::InvalidTestSize);
        }
        let mut rng = match random_seed {
            Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
            None => rand::rngs::StdRng::from_rng(rand::thread_rng()).unwrap(),
        };

        let n_samples = self.data.len();
        let n_test = (n_samples as f64 * test_size).round() as usize;
        let n_train = n_samples - n_test;

        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.shuffle(&mut rng);

        let train_indices = &indices[..n_train];
        let test_indices = &indices[n_train..];

        let train_data: Vec<Vec<f64>> = train_indices.iter().map(|&i| self.data[i].clone()).collect();
        let train_target: Vec<f64> = train_indices.iter().map(|&i| self.target[i]).collect();

        let test_data: Vec<Vec<f64>> = test_indices.iter().map(|&i| self.data[i].clone()).collect();
        let test_target: Vec<f64> = test_indices.iter().map(|&i| self.target[i]).collect();

        Ok((
            DataSet::new(train_data, train_target).unwrap(), 
            DataSet::new(test_data, test_target).unwrap()
        ))
    }

    /// # Creates a new DataSet instance from a CSV file.
    /// This function reads a CSV file and initializes a DataSet with the provided features and target values.
    pub fn from_csv(path: &str, features: &[&str], target: &str) -> Result<Self, DataSetError> {
        let mut data = Vec::new();
        let mut target_values = Vec::new();

        let mut reader = csv::Reader::from_path(path).map_err(|_| DataSetError::CsvError)?;
        let headers = reader.headers().map_err(|_| DataSetError::IncorrectCSVHeader)?;
        let mut feature_indexes: Vec<usize> = Vec::new();
        for feature in features {
            if let Some(index) = headers.iter().position(|s| s == *feature) {
                feature_indexes.push(index);
            } else {
                return Err(DataSetError::IncorrectCSVHeader);
            }
        }
        let target_index = headers.iter().position(|s| s == target).ok_or(DataSetError::IncorrectCSVHeader)?;

        
        for result in reader.records() {
            let record = result.map_err(|_| DataSetError::EmptyDataSet)?;
            
            let row: Vec<f64> = feature_indexes.iter()
                .map(|i| record.get(*i).unwrap_or("0").parse().unwrap_or(0.0))
                .collect();
            data.push(row);
            target_values.push(record.get(target_index).unwrap_or("0").parse().unwrap_or(0.0));
        }

        Self::new(data, target_values)
    }

    /// # Returns the data of the dataset.
    pub fn data(&self) -> &Vec<Vec<f64>> {
        &self.data
    }

    /// # Returns a mutable reference to the data of the dataset.
    pub fn data_mut(&mut self) -> &mut Vec<Vec<f64>> {
        &mut self.data
    }

    /// # Returns the target values of the dataset.
    pub fn target(&self) -> &Vec<f64> {
        &self.target
    }

    /// # Returns a mutable reference to the target values of the dataset.
    pub fn target_mut(&mut self) -> &mut Vec<f64> {
        &mut self.target
    }

}

#[derive(Clone, Debug)]
pub struct DataSetIter<'a> {
    dataset: &'a DataSet,
    index: usize,
}
impl<'a> Iterator for DataSetIter<'a> {
    type Item = Sample;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.dataset.data.len() {
            let sample = Sample {
                features: self.dataset.data[self.index].clone(),
                target: self.dataset.target[self.index],
            };
            self.index += 1;
            Some(sample)
        } else {
            None
        }
    }
}

impl<'a> IntoIterator for &'a DataSet {
    type Item = Sample;
    type IntoIter = DataSetIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        DataSetIter {
            dataset: self,
            index: 0,
        }
    }
}


#[derive(Clone, Debug)]
pub struct Sample {
    features: Vec<f64>,
    target: f64,
}

impl Sample {
    pub fn features(&self) -> &Vec<f64> {
        &self.features
    }

    pub fn target(&self) -> f64 {
        self.target
    }
}




#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_csv() {
        let path = "./insurance_cleaned.csv";
        let features = ["female", "male", "age", "bmi", "children", "smoker"];
        let target = "charges";

        // Test the from_csv function
        let dataset = DataSet::from_csv(path, &features, target);
        assert!(dataset.is_ok());
        let dataset = dataset.unwrap();
        assert_eq!(dataset.data.len(), 1338);
        // println!("Data: {:?}", dataset.data);
    }

    #[test]
    fn test_new_error_handling() {
        let ds1 = DataSet::new(vec![], vec![]);
        assert_eq!(ds1, Err(DataSetError::EmptyDataSet));

        let ds2 = DataSet::new(vec![vec![1.0, 2.0]], vec![]);
        assert_eq!(ds2, Err(DataSetError::EmptyDataSet));

        let ds3 = DataSet::new(vec![vec![1.0, 2.0]], vec![1.0]);
        assert_eq!(ds3, Ok(DataSet { data: vec![vec![1.0, 2.0]], target: vec![1.0] }));

        let ds4 = DataSet::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]], vec![1.0]);
        assert_eq!(ds4, Err(DataSetError::IncompatibleDimensions));

        let ds5 = DataSet::new(vec![vec![1.0, 2.0], vec![3.0]], vec![1.0, 2.0]);
        assert_eq!(ds5, Err(DataSetError::IncorrectFeatureLengths));
    }

    #[test]
    fn test_split() {
        let data = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
            vec![7.0, 8.0],
        ];
        let target = vec![1.0, 2.0, 3.0, 4.0];
        let dataset = DataSet::new(data.clone(), target.clone()).unwrap();
        println!("Original Data: {:?}", dataset.data);
        println!("Original Target: {:?}", dataset.target);

        let (train_set, test_set) = dataset.split(Some(57), 0.5).unwrap();

        assert_eq!(train_set.data.len(), 2);
        assert_eq!(test_set.data.len(), 2);

        println!("Train Data: {:?}", train_set.data);
        println!("Train Target: {:?}", train_set.target);
        println!("Test Data: {:?}", test_set.data);
        println!("Test Target: {:?}", test_set.target);
    }

    #[test]
    fn test_split_invalid_test_size() {
        let data = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
            vec![7.0, 8.0],
        ];
        let target = vec![1.0, 2.0, 3.0, 4.0];
        let dataset = DataSet::new(data.clone(), target.clone()).unwrap();

        let result = dataset.split(Some(57), 1.5);
        assert_eq!(result, Err(DataSetError::InvalidTestSize));
    }
    
}