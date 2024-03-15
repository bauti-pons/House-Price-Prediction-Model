# House Price Prediction Project

    # Loading Data Set

# Library import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data set
data = pd.read_csv('https://raw.githubusercontent.com/bauti-pons/data/main/california_housing_prices.csv')
data

    # Data Exploration

# Check if the data set has missing values
data.info()

# Remove missing values from the data set inplace
data.dropna(inplace=True)
data.info()

from sklearn.model_selection import train_test_split

# Get the independent variables by removing the dependent variable column from the dataset
x = data.drop(["median_house_value"], axis = 1)
# Get the dependent variables
y = data["median_house_value"]

# Split the data set into training and testing sets, with 80% of the data used for training and 20% used for testing.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Create a new DataFrame by joining x_train and y_train
train_data = x_train.join(y_train)
train_data

# Generate and display a separate histogram for each numeric column in train_data. Each one shows the distribution of values in its respective column
train_data.hist(figsize=(14, 7))

# Generate a visual representation of how strongly the features in train_data are correlated with each other.
plt.figure(figsize=(12, 6))
sns.heatmap(train_data.corr(), annot=True, cmap="YlOrRd")

    # Data Preprocessing

# Apply a logarithmic transformation to several columns of the train_data datafame to reduce skewness
train_data["total_rooms"] = np.log(train_data["total_rooms"] + 1)
train_data["total_bedrooms"] = np.log(train_data["total_bedrooms"] + 1)
train_data["population"] = np.log(train_data["population"] + 1)
train_data["households"] = np.log(train_data["households"] + 1)

train_data.hist(figsize=(14,7))

# Count the number of occurrences of each unique value in the ocean_proximity column of the train_data dataframe
train_data.ocean_proximity.value_counts()

# Convert the categorical variable ocean_proximity into a dataframe of binary variables (0 = no, 1 = yes)
# Add the new one-hot encoded columns to the original train_data dataframe, while keeping all the original columns
# Drop the original ocean proximity column from the resulting dataframe
train_data = train_data.join(pd.get_dummies(train_data.ocean_proximity)).drop(["ocean_proximity"], axis=1)

train_data

plt.figure(figsize=(12, 6))
sns.heatmap(train_data.corr(), annot=True, cmap="YlOrRd")

# Create a scatter plot that visualizes geographical data points (with latitude and longitude as coordinates) from the train_data dataframe
plt.figure(figsize=(14,7))
sns.scatterplot(x="latitude", y="longitude", data=train_data, hue="median_house_value", palette="coolwarm")

    # Feature Engineering

# Add two new features to the train_data dataframe, calculated from existing columns
# The purpose is to reveal patterns or relationships that were not previously evident, improving the performance of the predictive model
train_data["bedroom_ratio"] = train_data["total_bedrooms"] / train_data["total_rooms"]
train_data["household_rooms"] = train_data["total_rooms"] / train_data["households"]

plt.figure(figsize=(12, 6))
sns.heatmap(train_data.corr(), annot=True, cmap="YlOrRd")

    # Linear Regression Model

# Create and training a linear regression model using the scikit-learn library, specifically its LinearRegression class
# Apply feature scaling to your training data (x_train) using the StandardScaler class from the scikit-learn library
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x_train, y_train = train_data.drop(["median_house_value"], axis=1), train_data["median_house_value"]
x_train_s = scaler.fit_transform(x_train)

reg = LinearRegression()

reg.fit(x_train, y_train)

# Ensure that the test data undergoes the same transformations as the training data before it can be used for model evaluation
test_data = x_test.join(y_test)

test_data["total_rooms"] = np.log(test_data["total_rooms"] + 1)
test_data["total_bedrooms"] = np.log(test_data["total_bedrooms"] + 1)
test_data["population"] = np.log(test_data["population"] + 1)
test_data["households"] = np.log(test_data["households"] + 1)

test_data = test_data.join(pd.get_dummies(test_data.ocean_proximity)).drop(["ocean_proximity"], axis=1)

test_data["bedroom_ratio"] = test_data["total_bedrooms"] / test_data["total_rooms"]
test_data["household_rooms"] = test_data["total_rooms"] / test_data["households"]

# Split the preprocessed test_data dataframe back into features (x_test) and the target variable (y_test) after all the preprocessing steps have been applied
x_test, y_test = test_data.drop(["median_house_value"], axis=1), test_data["median_house_value"]

# Transform the test dataset (x_test) using the previously fitted scaler instance
x_test_s = scaler.transform(x_test)

# Evaluate the performance of the trained linear regression model reg on the test dataset
reg.score(x_test, y_test)

    # Random Forest Model

# Create and train a RandomForestRegressor model using the scikit-learn library
from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor()

forest.fit(x_train_s, y_train)

# Evaluate the performance of your trained RandomForestRegressor model (forest) on the test dataset
forest.score(x_test_s, y_test)

# Import GridSearchCV for hyperparameter tuning
from sklearn.model_selection import GridSearchCV

# Create an instance of the RandomForestRegressor
forest = RandomForestRegressor()

# Define the parameter grid to search through for the RandomForestRegressor
param_grid = {
    "n_estimators" : [100, 200, 300], # The number of trees in the forest
    "min_samples_split" : [2, 4],     # The minimum number of samples required to split an internal node
    "max_depth" : [None, 4, 8]        # The maximum depth of the trees. 'None' means nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples
}

# Initialize GridSearchCV with the RandomForestRegressor instance (forest), the parameter grid, and configuration for cross-validation
grid_search = GridSearchCV(forest, param_grid, cv=5,          # cv=5 specifies 5-fold cross-validation
                           scoring="neg_mean_squared_error",  # Use negative mean squared error as the evaluation metric
                           return_train_score=True)           # Return training scores for analyzing overfitting

# Fit the grid search to the training data. This process will find the best combination of parameters specified in param_grid based on the scoring metric, using 5-fold cross-validation
grid_search.fit(x_train, y_train)

# Access the best estimator found by the grid search
grid_search.best_estimator_

# Evaluate the best estimator found by the grid search on the test dataset
grid_search.best_estimator_.score(x_test_s, y_test)