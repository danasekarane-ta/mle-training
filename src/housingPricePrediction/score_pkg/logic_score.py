import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calculate_score(y, pred):
    mse = mean_squared_error(y, pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, pred)
    return rmse, mae
