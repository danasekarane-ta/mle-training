import argparse
import os
import pickle
import pandas as pd
from housingPricePrediction.train_pkg import data_training


def model_training(input_path, output_path):
    """ Method to train the model
    """
    housing_X = pd.read_csv(os.path.join(input_path, 'X_train.csv'))
    housing_y = pd.read_csv(
        os.path.join(input_path, 'y_train.csv')
        ).values.ravel()

    os.makedirs(output_path, exist_ok=True)
    _, linear_model = data_training.train_data_regression(
        "lin",
        housing_X,
        housing_y
    )
    # Dump the model
    with open(output_path + "/linreg_model.pk", 'wb') as f:
        pickle.dump(linear_model, f)
    _, dtree_model = data_training.train_data_regression("tree", housing_X,
                                                         housing_y)
    with open(output_path+"/decission_tree_model.pk", 'wb') as f:
        pickle.dump(dtree_model, f)
    final_model_rand = data_training.cross_validation('RandomizedSearchCV',
                                                      housing_X,
                                                      housing_y)
    with open(output_path + "/rand_cv_model.pkl", 'wb') as f:
        pickle.dump(final_model_rand, f)
    final_model_grid = data_training.cross_validation('GridSearchCV',
                                                      housing_X,
                                                      housing_y)
    with open(output_path+"/gs_cv_model.pkl", 'wb') as f:
        pickle.dump(final_model_grid, f)
    print("model_training done Successflly")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder", help="Input dataset folder path")
    parser.add_argument("out_folder", help="Output dataset folder path)")
    args = parser.parse_args()
    model_training(args.input_folder, args.out_folder)
