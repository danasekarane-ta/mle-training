import os
import tarfile
import numpy as np
import pandas as pd
from scipy.stats import randint
from six.moves import urllib
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedShuffleSplit,
)
from sklearn.tree import DecisionTreeRegressor

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("data", "raw")
HOUSING_URL = DOWNLOAD_ROOT + "data/raw/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    """
    Fetches housing data from a URL and saves it to a local path.

    Parameters:
    - housing_url (str): URL of the housing data.
    - housing_path (str): Local directory path to save the data.
    """
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    """
    Loads housing data from a CSV file.

    Parameters:
    - housing_path (str): Local directory path where the data is stored.

    Returns:
    - pd.DataFrame: DataFrame containing the housing data.
    """
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def add_extra_features(data):
    """
    Adds extra features to the housing data.

    Parameters:
    - data (pd.DataFrame): DataFrame containing the housing data.

    Returns:
    - pd.DataFrame: DataFrame with additional features.
    """
    data["rooms_per_household"] = data["total_rooms"] / data["households"]
    data["bedrooms_per_room"] = data["total_bedrooms"] / data["total_rooms"]
    data["population_per_household"] = data["population"] / data["households"]
    return data


def prepare_data(data):
    """
    Prepares the housing data by separating features and labels.

    Parameters:
    - data (pd.DataFrame): DataFrame containing the housing data.

    Returns:
    - tuple: Tuple containing features and labels.
    """
    features = data.drop("median_house_value", axis=1)
    labels = data["median_house_value"].copy()
    return features, labels


def split_data(data):
    """
    Splits the housing data into training and testing sets.

    Parameters:
    - data (pd.DataFrame): DataFrame containing the housing data.

    Returns:
    - tuple: Tuple containing the training and testing sets.
    """
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(data, data["income_cat"]):
        strat_train_set = data.loc[train_index]
        strat_test_set = data.loc[test_index]
    return strat_train_set, strat_test_set


def income_cat_proportions(data):
    """
    Calculates the proportions of income categories in the housing data.

    Parameters:
    - data (pd.DataFrame): DataFrame containing the housing data.

    Returns:
    - pd.Series: Series containing the proportions of income categories.
    """
    return data["income_cat"].value_counts() / len(data)


def train_decision_tree(housing_prepared, housing_labels):
    """
    Trains a decision tree regressor model.

    Parameters:
    - housing_prepared (pd.DataFrame): DataFrame containing prepared housing data.
    - housing_labels (pd.Series): Series containing housing labels.

    Returns:
    - DecisionTreeRegressor: Trained decision tree regressor model.
    """
    tree_reg = DecisionTreeRegressor(random_state=42)
    tree_reg.fit(housing_prepared, housing_labels)
    return tree_reg


def train_linear_regression(housing_prepared, housing_labels):
    """
    Trains a linear regression model.

    Parameters:
    - housing_prepared (pd.DataFrame): DataFrame containing prepared housing data.
    - housing_labels (pd.Series): Series containing housing labels.

    Returns:
    - LinearRegression: Trained linear regression model.
    """
    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)
    return lin_reg


def train_random_forest(param_grid, housing_prepared, housing_labels):
    """
    Trains a random forest regressor model using grid search for hyperparameter tuning.

    Parameters:
    - param_grid (list of dicts): List of dictionaries specifying hyperparameter grids.
    - housing_prepared (pd.DataFrame): DataFrame containing prepared housing data.
    - housing_labels (pd.Series): Series containing housing labels.

    Returns:
    - RandomForestRegressor: Best trained random forest regressor model.
    """
    forest_reg = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(
        forest_reg,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        return_train_score=True,
    )
    grid_search.fit(housing_prepared, housing_labels)
    print("Best parameters:", grid_search.best_params_)
    cvres = grid_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)
    return grid_search.best_estimator_


def preprocess_data(housing):
    """
    Preprocesses the housing data by adding extra features, splitting, and encoding categorical variables.

    Parameters:
    - housing (pd.DataFrame): DataFrame containing the raw housing data.

    Returns:
    - tuple: Tuple containing prepared housing data, labels, and test set.
    """
    housing = add_extra_features(housing)
    strat_train_set, strat_test_set = split_data(housing)
    housing.drop("income_cat", axis=1, inplace=True)
    housing_labels = strat_train_set["median_house_value"].copy()
    features, labels = prepare_data(strat_train_set)

    imputer = SimpleImputer(strategy="median")
    housing_num = features.drop("ocean_proximity", axis=1)
    imputer.fit(housing_num)
    X = imputer.transform(housing_num)
    housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=features.index)
    housing_tr = add_extra_features(housing_tr)
    housing_cat = features[["ocean_proximity"]]
    housing_prepared = housing_tr.join(pd.get_dummies(housing_cat, drop_first=True))

    return housing_prepared, labels, strat_test_set


def evaluate_final_model(final_model, strat_test_set):
    """
    Evaluates the final model on the test set and prints the RMSE.

    Parameters:
    - final_model: Trained final model.
    - strat_test_set: Test set.

    Returns:
    - None
    """
    X_test, y_test = prepare_data(strat_test_set)
    X_test_num = X_test.drop("ocean_proximity", axis=1)
    X_test_prepared = imputer.transform(X_test_num)
    X_test_prepared = pd.DataFrame(
        X_test_prepared, columns=X_test_num.columns, index=X_test.index
    )
    X_test_prepared = add_extra_features(X_test_prepared)
    X_test_cat = X_test[["ocean_proximity"]]
    X_test_prepared = X_test_prepared.join(pd.get_dummies(X_test_cat, drop_first=True))
    final_predictions = final_model.predict(X_test_prepared)
    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)
    print("Final RMSE:", final_rmse)


def fetch_and_prepare_housing_data():
    """
    Fetches and prepares the housing data by adding income category feature.

    Returns:
    - pd.DataFrame: DataFrame containing the prepared housing data.
    """
    fetch_housing_data()
    housing = load_housing_data()
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )
    return housing


housing = fetch_and_prepare_housing_data()

param_grid = [
    {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
    {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
]

housing_prepared, housing_labels, strat_test_set = preprocess_data(housing)
tree_reg = train_decision_tree(housing_prepared, housing_labels)
lin_reg = train_linear_regression(housing_prepared, housing_labels)
forest_reg = train_random_forest(param_grid, housing_prepared, housing_labels)
evaluate_final_model(forest_reg, strat_test_set)