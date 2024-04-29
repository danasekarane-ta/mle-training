from .data_ingestion import (
    load_housing_data,
    fetch_housing_data,
    preprocess_data,
    data_visualization,
    impute_data,
    extract_features,
    create_dummy_data,

)
from .data_training import (
    stratifiedShuffleSplit,
    train_data_regression,
    cross_validation
)
from .logic_score import calculate_score


def data_prediction():
    """ Fetch the data and Predict the modelling
    """
    # Fetch the data
    fetch_housing_data()
    # Load the fetched data
    housing = load_housing_data()

    # train the data
    train_set, test_set, strat_train_set, strat_test_set = (
        stratifiedShuffleSplit(housing)
    )

    # preprocess_data
    preprocess_data(
        housing, strat_train_set, strat_test_set, test_set
    )

    # Data Visualiztion for train set
    housing_train = strat_train_set.copy()
    print("Data Visualization for train set")
    data_visualization(housing_train)

    # Data Visualiztion for test set
    housing_test = strat_train_set.copy()
    print("Data Visualization for test set")
    data_visualization(housing_test)

    # Feature Extraction for train set
    housing_train, housing_y_train, housing_X_train = impute_data(
        strat_train_set
    )

    housing_X_train = extract_features(housing_X_train)
    housing_X_train = create_dummy_data(
        housing_train, housing_X_train
    )
    # Feature Extraction for test set
    housing_test, housing_y_test, housing_X_test = impute_data(
        strat_test_set
    )
    housing_X_test = extract_features(housing_X_test)
    housing_X_test = create_dummy_data(
        housing_test,
        housing_X_test
    )

    # train model for training set
    housing_predictions_lin = train_data_regression(
        "linear", housing_X_train, housing_y_train
    )

    lin_rmse_train, lin_mae_train = calculate_score(
        housing_y_train, housing_predictions_lin
    )

    housing_predictions_reg = train_data_regression(
        "DecessionTree", housing_X_train, housing_y_train
    )
    tree_rmse_train, tree_mae_train = calculate_score(
        housing_y_train, housing_predictions_reg
    )
    final_model_train_random = cross_validation(
        'RandomizedSearchCV',
        housing_X_train,
        housing_y_train
    )
    print("Best Estimator", final_model_train_random)
    final_model_train_grid = cross_validation('GridSearchCV',
                                                            housing_X_train,
                                                            housing_y_train)
    print("Best Estimator", final_model_train_grid)

    final_predictions_train = final_model_train_grid.predict(housing_X_train)
    final_rmse_train, final_mae_train = calculate_score(
        housing_y_train, final_predictions_train
    )
    # scoring for train set
    print("Scoring for train-data: \n",
          final_rmse_train, "   ", final_mae_train)

    # test using trained models
    final_predictions_test = final_model_train_grid.predict(housing_X_test)
    final_rmse_test, final_mae_test = calculate_score(
        housing_y_test, final_predictions_test
    )
    # scoring for test set
    print("Scoring for test-data: \n", final_rmse_test, "   ", final_mae_test)


if __name__ == '__main__':
    data_prediction()
