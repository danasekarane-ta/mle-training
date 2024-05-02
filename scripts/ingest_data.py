#! /home/dhikshita/miniconda3/envs/mle-dev/bin/python
import argparse
import os

from housePricePrediction import data_ingestion, data_training


def ingestion(output_folder):
    # fetch
    raw_data_path = output_folder+'/raw'
    os.makedirs(raw_data_path, exist_ok=True)
    DOWNLOAD_ROOT = (
        "https://raw.githubusercontent.com/ageron/handson-ml/master/")
    HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
    data_ingestion.fetch_housing_data(HOUSING_URL, raw_data_path)
    print("Data Downloaded n Extracted Successfully")
    # load
    housing = data_ingestion.load_housing_data(raw_data_path)
    print("Data Loaded Successfully")
    # train-test-split
    train_set, test_set, train, test = (
        data_training.stratified_Shuffle_Split(housing)
    )
    # preprocessing
    housing, y_train, X_train = data_ingestion.imputing_data(train)
    X_train = data_ingestion.feature_extraction(X_train)
    X_train = data_ingestion.creating_dummies(train, X_train)

    housing, y_test, X_test = data_ingestion.imputing_data(test)
    X_test = data_ingestion.feature_extraction(X_test)
    X_test = data_ingestion.creating_dummies(housing, X_test)
    print("Preprocessing done Successfully")

    # saving op
    processed_data_path = output_folder + '/processed'
    os.makedirs(processed_data_path, exist_ok=True)
    X_train = X_train.to_csv(processed_data_path + '/X_train.csv', index=False)
    y_train = y_train.to_csv(processed_data_path + '/y_train.csv', index=False)
    X_test = X_test.to_csv(processed_data_path+'/X_test.csv', index=False)
    y_test = y_test.to_csv(processed_data_path+'/y_test.csv', index=False)
    print("Train Test dataset split Successfully")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_folder", help="Add path to output folder")
    args = parser.parse_args()
    ingestion(args.output_folder)


if __name__ == '__main__':
    main()

