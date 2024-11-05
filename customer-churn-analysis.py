from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, ChiSqSelector
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Initialize Spark session
spark = SparkSession.builder.appName("CustomerChurnMLlib").getOrCreate()

# Load dataset
data_path = "customer_churn.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)

# Task 1: Data Preprocessing and Feature Engineering
def preprocess_data(df):
    # TODO: Fill missing values in the "TotalCharges" column
    # Hint: Use the `na.fill()` function with a value of 0 for "TotalCharges"
    
    # TODO: Encode categorical variables into numeric indexes
    # Hint: Use StringIndexer to transform columns like "gender", "PhoneService", "InternetService", and "Churn"
    
    # TODO: One-hot encode the indexed categorical features
    # Hint: Use OneHotEncoder to transform the indexed columns into one-hot encoded vectors
    
    # TODO: Assemble features into a single feature vector
    # Hint: Use VectorAssembler to combine the relevant columns into a single "features" column
    
    # Return the final DataFrame with "features" and "ChurnIndex" columns
    pass

# Task 2: Splitting Data and Building a Logistic Regression Model
def train_logistic_regression_model(df):
    # TODO: Split data into training and testing sets
    # Hint: Use randomSplit with a split ratio of 80% training and 20% testing
    
    # TODO: Train a logistic regression model
    # Hint: Initialize LogisticRegression with "ChurnIndex" as the label column and "features" as the features column
    
    # TODO: Predict on the test set and evaluate the model
    # Hint: Use BinaryClassificationEvaluator with metricName="areaUnderROC" for evaluation
    
    # Print the accuracy of the model
    pass

# Task 3: Feature Selection Using Chi-Square Test
def feature_selection(df):
    # TODO: Select top 5 features using Chi-Square test
    # Hint: Use ChiSqSelector with numTopFeatures set to 5
    
    # Display the selected features for review
    pass

# Task 4: Hyperparameter Tuning with Cross-Validation for Multiple Models
def tune_and_compare_models(df):
    # TODO: Split data into training and testing sets
    # Hint: Use randomSplit with 80% training and 20% testing
    
    # Define models
    # TODO: Define the models for LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, and GBTClassifier
    # Hint: Initialize each model with "ChurnIndex" as the label column and "features" as the features column
    
    # Define hyperparameter grids
    # TODO: Define parameter grids for each model using ParamGridBuilder
    # Example: For LogisticRegression, add `regParam` and `maxIter` as parameters to tune
    
    # TODO: Perform cross-validation for each model and find the best model for each
    # Hint: Use CrossValidator with each model and parameter grid

    # TODO: Get the best model and evaluate it on the test set
    # Print accuracy or AUC for each model
    pass

# Execute tasks
# TODO: Call preprocess_data to process the data
# TODO: Call train_logistic_regression_model to train and evaluate the logistic regression model
# TODO: Call feature_selection to select top features
# TODO: Call tune_and_compare_models to tune and evaluate different models

# Stop Spark session
spark.stop()
