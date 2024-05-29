import logging
import mlflow
from ingest_data import (
    fetch_housing_data,
    load_housing_data,
    preprocess_data,
    data_visualization,
    impute_data,
    create_dummy_data,
    extract_features
)
from train_data import stratified_shuffle_split, train_regression_data, cross_validation
from logic_score import calculate_score

logger = logging.getLogger(__name__)


mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
# Create nested runs
experiment = mlflow.set_experiment("MlLearningModel")
experiment_id = experiment.experiment_id
with mlflow.start_run(
    run_name="Parent_run",
    experiment_id=experiment_id,
    tags={"version": "v1", "priority": "P1"},
    description="parent",
) as parent_run:
    mlflow.log_param("parent", "yes")
    with mlflow.start_run(
        run_name="DATA_TRAINING_RUN",
        experiment_id=experiment_id,
        description="Data Ingestion Run",
        nested=True,
    ) as data_ingestion_run:
        mlflow.log_param("child", "yes")

        logger.info("Fetching the housing data for data prediction")
        # Fetch the data
        fetch_housing_data()
        # Load the fetched data
        housing_data = load_housing_data()
        # Log the housing data
        mlflow.log_params(housing_data)
        # train the data
        train_set, test_set, strat_train_set, strat_test_set = (
            stratified_shuffle_split(housing_data)
        )
    with mlflow.start_run(
        run_name="DATA_PROCESSING_RUN",
        experiment_id=experiment_id,
        description="Data Processing Run",
        nested=True,
    ) as data_processing_run:
        logger.debug("Preprossing the fetched data")
        # preprocess_data
        preprocess_data(
            housing_data, strat_train_set, strat_test_set, test_set
        )

        # Data Visualiztion for train set
        housing_train = strat_train_set.copy()
        logger.info("Data Visualization for train set")
        data_visualization(housing_train)

        # Data Visualiztion for test set
        housing_test = strat_train_set.copy()
        logger.info("Data Visualization for test set")
        data_visualization(housing_test)

        # Feature Extraction for train set
        housing_train, housing_y_train, housing_X_train = \
            impute_data(strat_train_set)

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

    with mlflow.start_run(
        run_name="SCORE_LOGIC_RUN",
        experiment_id=experiment_id,
        description="Score Logic",
        nested=True,
    ) as score_logic_run:

        # train model for training set
        housing_predictions_lin = train_regression_data(
            "linear", housing_X_train, housing_y_train
        )

        lin_rmse_train, lin_mae_train = calculate_score(
        housing_y_train, housing_predictions_lin
        )

        housing_predictions_reg = train_regression_data(
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
        logger.debug("Best Estimator", final_model_train_random)

        final_model_train_grid = cross_validation('GridSearchCV',
                                                                housing_X_train,
                                                                housing_y_train)
        # Save/Log the model in mlflow
        mlflow.sklearn.log_model(final_model_train_grid, "best_trained_model")

        # Load the model back from run
        run_id = score_logic_run.info.run_id
        trained_model = mlflow.sklearn.load_model(f"runs:/{run_id}/best_trained_model")
        final_predictions_train = trained_model.predict(housing_X_train)
        final_rmse_train, final_mae_train = calculate_score(
            housing_y_train, final_predictions_train
        )
        # scoring for train set
        logger.debug("Scoring for train-data: \n",
                    final_rmse_train, "   ", final_mae_train)

        # test using trained models
        final_predictions_test = trained_model.predict(housing_X_test)
        final_rmse_test, final_mae_test = calculate_score(
            housing_y_test, final_predictions_test
        )
         # log the metrics

        # scoring for test set
        logger.info("Scoring for test-data: \n", final_rmse_test, "   ",
                    final_mae_test)