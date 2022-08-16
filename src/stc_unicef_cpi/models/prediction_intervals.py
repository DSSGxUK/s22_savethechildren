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
        input_data: Dataframe containing all data; output of make_dataset
        target_dim: dimension to predict
        mapie_dir: path to save fitted MapieRegressor instance

    Outputs: None
    """
    
    pipeline = joblib.load(Path(pipeline_dir)/(pipeline_name+".pkl"))
    
    fitted = [v for v in vars(pipeline[-1]) if v.endswith("_") and not v.startswith("__")]
    if not fitted:
        print("Warning, pipeline has not been fitted yet, cannot be used to generate intervals!")
        return

    # subset by pipeline input
    input_data = input_data[list(np.append(pipeline.feature_names_in_, target_dim))]
    
    # subset training rows with ground truth
    input_data = input_data[~input_data[target_dim].isnull()]
    x, y = input_data.loc[:, input_data.columns != target_dim], input_data.loc[:, target_dim]

    mapie = MapieRegressor(estimator=pipeline, cv="prefit").fit(x, y)
    joblib.dump(mapie, Path(mapie_dir)/("mapie_"+target_dim+".pkl"))


def predict_intervals(input_data, target_dim, mapie_dir, alpha=0.05, save_dir=None):

    """
    Get prediction intervals for all data using fitted MapieRegressor

    Inputs:
        input_data: Dataframe containing all data; output of make_dataset
        target_dim: dimension to predict
        mapie_dir: path to saved MapieRegressor instance
        alpha: percent of out of intervals predictions tolerated
        save_dir: path to save predictions csv

    Outputs: If save_dir is None, Dataframe of shape (input_dir.shape[0], 3), else None
    """

    mapie = joblib.load(Path(mapie_dir)/("mapie_"+target_dim+".pkl"))

    # subset by pipeline input
    input_data = input_data[list(mapie.estimator.feature_names_in_)]

    pred_intervals = pd.DataFrame(
        mapie.predict(input_data, alpha=alpha)[1].reshape(-1,2),
        columns=["lower_"+target_dim, "upper_"+target_dim]
    )
    
    # restrict upper & lower bound for predictions to 1 & 0 resp
    if target_dim != "sumpoor_sev":
        pred_intervals["upper_"+target_dim] = pred_intervals["upper_"+target_dim].apply(lambda x: 1 if x > 1 else x)
        pred_intervals["lower_"+target_dim] = pred_intervals["lower_"+target_dim].apply(lambda x: 0 if x < 0 else x)

    pred_intervals["prediction_"+target_dim] = mapie.predict(input_data)
    pred_intervals = pred_intervals[["lower_"+target_dim, "prediction_"+target_dim, "upper_"+target_dim]]

    pred_intervals["hex_code"] = input_data["hex_code"]

    if save_dir:
        pred_intervals.to_csv(Path(save_dir)/("intervals_"+target_dim+".csv"), index=False)
    else:
        return pred_intervals
    


        