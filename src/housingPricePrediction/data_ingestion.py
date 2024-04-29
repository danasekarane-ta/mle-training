import os
import tarfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from six.moves import urllib
from sklearn.impute import SimpleImputer

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


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
    housing = pd.read_csv(csv_path)
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )
    return housing


def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)


def preprocess_data(housing, X_strat, y_strat, y):
    compare_props = pd.DataFrame(
        {
            "Overall": income_cat_proportions(housing),
            "Stratified": income_cat_proportions(y_strat),
            "Random": income_cat_proportions(y),
        }
    ).sort_index()
    compare_props["Rand. %error"] = (
        100 * compare_props["Random"] / compare_props["Overall"] - 100
    )

    compare_props["Strat. %error"] = (
        100 * compare_props["Stratified"] / compare_props["Overall"] - 100
    )
    for set_ in (X_strat, y_strat):
        set_.drop("income_cat", axis=1, inplace=True)

    return compare_props


def data_visualization(housing):
    """ Visualise the data
     Parameters:
    - housing: Housing Dataset
    """
    housing.plot(kind="scatter", x="longitude", y="latitude")
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
    plt.show()
    corr_matrix = housing.corr(numeric_only=True)
    corr_matrix["median_house_value"].sort_values(ascending=False)
    print(corr_matrix)


def extract_features(housing):
    """ Extract the features from the housing data set
     Parameters:
    - housing: Housing Dataset
    """
    housing["rooms_per_household"] = housing["total_rooms"] / \
        housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"] / \
        housing["total_rooms"]
    housing["population_per_household"] = housing["population"] / \
        housing["households"]
    return housing


def impute_data(X_train):
    """ Impute data using x_train dataset
    Parameters:
    - X_train: X_train Dataset
    """
    housing = X_train.drop("median_house_value", axis=1)
    y = X_train["median_house_value"].copy()

    imputer = SimpleImputer(strategy="median")
    X_num = housing.drop("ocean_proximity", axis=1)
    imputer.fit(X_num)
    X = imputer.transform(X_num)

    X_prepared = pd.DataFrame(X, columns=X_num.columns, index=housing.index)
    return housing, y, X_prepared


def create_dummy_data(housing, X_prepared):
    X_cat = housing[["ocean_proximity"]]

    X_prepared = X_prepared.join(pd.get_dummies(X_cat, drop_first=True))
    return X_prepared