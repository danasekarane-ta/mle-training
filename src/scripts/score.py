import argparse
import os
import pickle
import pandas as pd
from housingPricePrediction.score_pkg import logic_score


def calculate_model_score(data_folder, prediction_folder_path, out_folder):
    """ Calculate the model score based the prediction"""
    X_data = pd.read_csv(os.path.join(data_folder, 'X_test.csv'))
    Y_data = pd.read_csv(os.path.join(data_folder, 'y_test.csv'))
    os.makedirs(out_folder, exist_ok=True)
    files = os.listdir(prediction_folder_path)
    for file in files:
        if os.path.isfile(prediction_folder_path, file):
            with open(os.path.join(prediction_folder_path, file), 'rb') as f:
                pred_model = pickle.load(f)
                final_predictions_test = pred_model.predict(X_data)
                final_rmse_test, final_mae_test = logic_score.scoring_logic(
                    Y_data, final_predictions_test
                )
                print(final_rmse_test)
                with open(
                            os.path.join(
                                out_folder,
                                file,
                                "_score.txt"
                            ),
                        'w') as f:
                    f.write("RMSE : {}\n".format(final_rmse_test))
                    f.write("MAE : {}".format(final_mae_test))
    print("Model Scores saved Successfully")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="Input Dataset Folder Path")
    parser.add_argument("prediction_path",
                        help="Prediction Model FolderPath")
    parser.add_argument("out_file", help="Model Output file")
    args = parser.parse_args()
    calculate_model_score(args.data_path, args.prediction_path, args.out_file)
