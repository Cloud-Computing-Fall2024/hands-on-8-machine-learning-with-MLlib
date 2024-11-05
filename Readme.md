# Hands-on 8: Machine Learning with MLlib in PySpark

## Activity Overview
In this hands-on activity, you’ll use PySpark’s MLlib library to build a machine learning pipeline to predict customer churn. The activity covers data preprocessing, feature engineering, building and evaluating models, performing feature selection, and hyperparameter tuning using cross-validation. You’ll also compare different classification models to understand model selection.

## Objectives
1. Preprocess and prepare the dataset for machine learning.
2. Build and evaluate a Logistic Regression model to predict churn.
3. Use feature selection to understand the most important features.
4. Tune hyperparameters and compare models, including Logistic Regression, Decision Tree, Random Forest, and Gradient-Boosted Trees.

## Dataset
The dataset (`customer_churn.csv`) contains information on customer demographics, subscription details, and churn status. Ensure the dataset file is available in the project directory before beginning.

## Instructions

### Task 1: Data Preprocessing and Feature Engineering
1. **Fill Missing Values**: Handle missing values in key columns like `TotalCharges`.
2. **Encode Categorical Variables**: Use `StringIndexer` to convert categorical features (e.g., `gender`, `PhoneService`, `InternetService`, `Churn`) to numeric indices.
3. **One-Hot Encoding**: Apply `OneHotEncoder` to indexed features to create vectors for categorical variables.
4. **Feature Assembly**: Use `VectorAssembler` to combine features into a single feature vector.

### Task 2: Build and Evaluate a Logistic Regression Model
1. **Data Splitting**: Split the data into training and testing sets.
2. **Train Logistic Regression**: Fit a logistic regression model using the training set.
3. **Evaluate Model Performance**: Make predictions on the test set and calculate the model’s accuracy using `BinaryClassificationEvaluator`.

### Task 3: Feature Selection Using Chi-Square Test
1. **Select Important Features**: Use `ChiSqSelector` to select the top 5 most predictive features.
2. **Analyze Selected Features**: Review and interpret the selected features.

### Task 4: Hyperparameter Tuning and Model Comparison
1. **Train Multiple Models**: Experiment with multiple classifiers (Logistic Regression, Decision Tree, Random Forest, Gradient-Boosted Trees).
2. **Parameter Grids**: Define parameter grids for each model to optimize hyperparameters.
3. **Cross-Validation**: Use `CrossValidator` to tune each model, identifying the best parameters and model performance.
4. **Model Comparison**: Evaluate all models and select the one with the highest accuracy or AUC.

## Submission Instructions
1. **GitHub Repository**: Create a repository on GitHub Classroom and add code for each task, with comments and explanations.
2. **Activity Report**: In your repository, include screenshots of your output results and model performance metrics.
3. **Single Submission per Group**: Ensure that only one member from each group submits the repository link on Canvas. The report should include contributions from each member.

## Running the Activity
1. Ensure you have the dataset (`customer_churn.csv`) in the project directory.
2. Follow the hints provided in the template code and complete each task.
3. Run and test each task, ensuring your code works as expected and captures screenshots of outputs.

## Deadline
This activity is due on **Nov 5th**. Late submissions will not be accepted without prior approval.

## Notes
- Make sure to commit regularly to GitHub and include detailed commit messages.
- Review the PySpark MLlib documentation if needed for additional methods.
- Experiment with parameter grids and model options to optimize performance.
- Document your findings and observations in the activity report.

---