import argparse
import os
from housingPricePrediction.ingest_pkg import data_ingestion
from housingPricePrediction.train_pkg import data_training


def ingest_input_data(output_folder):
    """Ingest the input data"""
    raw_data_path = output_folder+'/raw'
    os.makedirs(raw_data_path, exist_ok=True)
    DOWNLOAD_ROOT = (
        "https://raw.githubusercontent.com/ageron/handson-ml/master/")
    HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
    data_ingestion.fetch_housing_data(HOUSING_URL, raw_data_path)
    # Load the data
    housing = data_ingestion.load_housing_data(raw_data_path)
    print("Data Loaded Successfully")
    # Train and Test split the data
    train_set, test_set, train, test = (
        data_training.stratifiedShuffleSplit(housing)
    )
    # Pre process the data
    housing, y_train, X_train = data_ingestion.impute_data(train)
    X_train = data_ingestion.feature_extraction(X_train)
    X_train = data_ingestion.creating_dummies(train, X_train)

    housing, y_test, X_test = data_ingestion.impute_data(test)
    X_test = data_ingestion.feature_extraction(X_test)
    X_test = data_ingestion.creating_dummies(housing, X_test)
    print("Preprocessing done Successfully")

    # Save the output to the folder
    processed_data_path = os.path.join(output_folder, 'processed')
    x_train_csv_path = os.path.join(processed_data_path, 'X_train.csv')
    y_train_csv_path = os.path.join(processed_data_path, 'y_train.csv')
    x_test_csv_path = os.path.join(processed_data_path, 'X_test.csv')
    y_test_csv_path = os.path.join(processed_data_path, 'y_test.csv')

    os.makedirs(processed_data_path, exist_ok=True)
    X_train = X_train.to_csv(x_train_csv_path, index=False)
    y_train = y_train.to_csv(y_train_csv_path, index=False)
    X_test = X_test.to_csv(x_test_csv_path, index=False)
    y_test = y_test.to_csv(y_test_csv_path, index=False)
    print("Train Test dataset split Successfully and saved")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_folder", help="Output Folder path")
    args = parser.parse_args()
    ingest_input_data(args.output_folder)


if __name__ == '__main__':
    main()
