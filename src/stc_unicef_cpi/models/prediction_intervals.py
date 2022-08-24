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
    :param pipeline_dir: path to trained Pipeline instance
    :type pipeline_dir: str
    :param pipeline_name: name of trained Pipeline instance
    :type pipeline_name: str
    :param input_data: Dataframe containing all data
    :type input_data: _type_
    :param target_dim: dimension to predict
    :type target_dim: str
    :param mapie_dir: path for saving MapieRegressor instance
    :type mapie_dir: str
    :return: None
    :rtype: _type_
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


def predict_intervals(input_data, target_dim, mapie_dir, alpha=0.05, batch_size=10000, save_dir=None):

    """
    Get prediction intervals for all data using fitted MapieRegressor
    :param input_data: Dataframe containing all data
    :type input_data: _type_
    :param target_dim: dimension to predict
    :type target_dim: str
    :param mapie_dir: path to saved MapieRegressor instance
    :type mapie_dir: str
    :param alpha: percent of out of intervals predictions tolerated, defualt is 0.05
    :type alpha: int, optional
    :param batch_size: batch size for processing, default is 10000
    :type batch_size: int, optional
    :param save_dir: path to save predictions csv
    :type save_dir: str, optional
    :return: If save_dir is None, pandas Dataframe else None
    :rtype: _type_
    """

    mapie = joblib.load(Path(mapie_dir)/("mapie_"+target_dim+".pkl"))

    # subset by pipeline input
    input_data = input_data[list(mapie.estimator.feature_names_in_)]

    all_preds = pd.DataFrame(columns=["lower_"+target_dim, "prediction_"+target_dim, "upper_"+target_dim])

    for i in range(0, len(input_data), batch_size):

        if i+batch_size > len(input_data):
            end = len(input_data)
        else:
            end = i+batch_size
        
        subset = input_data.iloc[i:end]

        pred_intervals = pd.DataFrame(
        mapie.predict(subset, alpha=alpha)[1].reshape(-1,2),
        columns=["lower_"+target_dim, "upper_"+target_dim]
        )

        # restrict upper & lower bound for predictions to 1 & 0 resp
        if target_dim != "sumpoor_sev":
            pred_intervals["upper_"+target_dim] = pred_intervals["upper_"+target_dim].apply(lambda x: 1 if x > 1 else x)
            pred_intervals["lower_"+target_dim] = pred_intervals["lower_"+target_dim].apply(lambda x: 0 if x < 0 else x)
        
        pred_intervals["prediction_"+target_dim] = mapie.predict(subset)
        pred_intervals = pred_intervals[["lower_"+target_dim, "prediction_"+target_dim, "upper_"+target_dim]]

        all_preds = pd.concat([all_preds, pred_intervals], axis=0)

    all_preds.reset_index(drop=True, inplace=True)
    all_preds["hex_code"] = input_data["hex_code"]

    if save_dir:
        all_preds.to_csv(Path(save_dir)/("intervals_"+target_dim+".csv"), index=False)
    else:
        return all_preds
    


        