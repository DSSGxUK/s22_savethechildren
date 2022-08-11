import numpy as np
from mapie.regression import MapieRegressor
import joblib
import pandas as pd
from pathlib import Path

random_state=0
np.random.seed(random_state)

def calibrate_prediction_intervals(pipeline_dir, pipeline_name, input_data, target_dim, mapie_dir):

    """
    Train MAPIE Regressor using train data

    Inputs:
        pipeline_dir: path to saved pipeline (from train_model.py)
        pipeline_name: name of saved pkl file
        input_data: Dataframe containing training data, including target variable
        target_dim: one of 6 dimensions from education, health, sanitation, housing, nutrition, water
        mapie_dir: path to save fitted MapieRegressor instance

    Outputs: None
    """
    
    pipeline = joblib.load(Path(pipeline_dir)/(pipeline_name+".pkl"))
    
    fitted = [v for v in vars(pipeline[-1]) if v.endswith("_") and not v.startswith("__")]
    if not fitted:
        print("Warning, pipeline has not been fitted yet, cannot be used to generate intervals!")
        return
    
    y_col = "dep_" + target_dim + "_sev"
    x, y = input_data.loc[:, input_data.columns != y_col], input_data.loc[:, y_col]

    mapie = MapieRegressor(estimator=pipeline, cv="prefit").fit(x, y)
    joblib.dump(mapie, Path(mapie_dir)/("mapie_"+target_dim+".pkl"))


def predict_intervals(input_data, target_dim, mapie_dir, alpha=0.05, save_dir=None):

    """
    Get prediction intervals for all data using fitted MapieRegressor

    Inputs:
        input_data: Dataframe containing data for which intervals are needed; may/may not include target

    Outputs: If save_dir is None, Dataframe of shape (input_dir.shape[0], 3), else None
    """

    mapie = joblib.load(Path(mapie_dir)/("mapie_"+target_dim+".pkl"))

    y_col = "dep_"+target_dim+"_sev"
    if y_col in input_data.columns:
        x = input_data.loc[:, input_data.columns != y_col]
    else:
        x = input_data

    pred_intervals = pd.DataFrame(mapie.predict(x, alpha=alpha)[1].reshape(-1,2), columns=["lower", "upper"])
    pred_intervals["upper"] = pred_intervals["upper"].apply(lambda x: 1 if x > 1 else x)
    pred_intervals["lower"] = pred_intervals["lower"].apply(lambda x: 0 if x < 0 else x)

    pred_intervals["prediction"] = mapie.predict(x)
    pred_intervals = pred_intervals[["lower", "prediction", "upper"]]

    if save_dir:
        pred_intervals.to_csv(Path(save_dir)/("intervals_"+target_dim+".csv"), index=False)
    else:
        return pred_intervals
    


        