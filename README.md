# Heart Disease Prediction Project

## Overview
This project implements a machine learning solution for heart disease prediction using various classification algorithms. The notebook demonstrates how to build and compare different models to predict the presence of heart disease based on patient characteristics.

## Dataset
The project uses a heart disease dataset (`heart.csv`) containing medical information about patients. The dataset includes the following features:

- **age**: Age of the patient
- **sex**: Gender (1 = male, 0 = female)
- **cp**: Chest pain type (0-3)
- **trestbps**: Resting blood pressure
- **chol**: Serum cholesterol level
- **fbs**: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
- **restecg**: Resting electrocardiographic results
- **thalach**: Maximum heart rate achieved
- **exang**: Exercise induced angina (1 = yes, 0 = no)
- **oldpeak**: ST depression induced by exercise relative to rest
- **slope**: Slope of the peak exercise ST segment
- **ca**: Number of major vessels colored by flourosopy
- **thal**: Thalassemia type
- **target**: Heart disease presence (1 = disease, 0 = no disease)

## Libraries Used
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation and analysis
- **Matplotlib**: Data visualization
- **Seaborn**: Statistical data visualization
- **Scikit-learn**: Machine learning library
  - `train_test_split`: Data splitting
  - `RandomForestClassifier`: Random Forest algorithm
  - `GradientBoostingClassifier`: Gradient Boosting algorithm
  - `SVC`: Support Vector Machine
  - `LogisticRegression`: Logistic Regression
  - `accuracy_score`: Model evaluation metric

## Project Structure
```
Projects/
├── test.ipynb                    # Main notebook file
└── README_test.ipynb.md         # This documentation file
```

## Code Overview

### Data Loading
```python
df = pd.read_csv("/Users/animeshbaggan/Downloads/heart.csv")
```
Loads the heart disease dataset from the specified CSV file.

### Data Preparation
```python
X = df.iloc[:,0:-1]  # Features (all columns except the last)
y = df.iloc[:,-1]    # Target variable (last column)
```
Separates the dataset into features (X) and target variable (y).

## Usage Instructions

1. **Prerequisites**: Ensure you have the required libraries installed:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```

2. **Dataset**: Place the `heart.csv` file in the specified directory or update the file path in the notebook.

3. **Running the Notebook**: 
   - Open the notebook in Jupyter Lab/Notebook
   - Execute the cells sequentially
   - The notebook will load the data and prepare it for machine learning

## Next Steps (To Complete the Project)

The current notebook sets up the foundation for the heart disease prediction project. To complete the implementation, you would typically add:

1. **Data Preprocessing**:
   - Handle missing values
   - Feature scaling/normalization
   - Feature engineering

2. **Model Training**:
   - Train each classifier on the training data
   - Implement cross-validation

3. **Model Evaluation**:
   - Test model performance on the test set
   - Compare accuracy across different algorithms
   - Generate confusion matrices and classification reports

4. **Visualization**:
   - Plot model performance comparisons
   - Visualize feature importance
   - Create ROC curves

5. **Model Selection**:
   - Choose the best performing model
   - Hyperparameter tuning

## Example Expected Output
The notebook loads a dataset with 1025 rows and 14 columns, showing patient information with features ranging from age and gender to medical test results.

## Contributing
This is a learning project demonstrating basic machine learning concepts for heart disease prediction. Feel free to extend the implementation with additional features and improvements.

## License
This project is for educational purposes.
