import argparse
import sys
import warnings
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import swifter
from flaml import AutoML
from flaml.ml import sklearn_metric_loss_score
from sklearn import set_config
from sklearn.compose import (
    ColumnTransformer,
    make_column_selector,
    make_column_transformer,
)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score  # , log_loss
from sklearn.model_selection import KFold, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)
from tqdm.auto import tqdm

from stc_unicef_cpi.utils.mlflow_utils import fetch_logged_data
from stc_unicef_cpi.utils.scoring import mae

# TODO: write proper warning handler to only suppress unhelpful msgs
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    ### Argument and global variables
    parser = argparse.ArgumentParser("High-res multi-dim CPI model training")
    DATA_DIRECTORY = Path("../../../data")
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        help="Pathway to data directory",
        default=DATA_DIRECTORY,
    )
    parser.add_argument(
        "--clean-name",
        type=str,
        help="Name of clean dataset inside data directory",
        default="clean_nga_w_autov1.csv",
    )
    parser.add_argument(
        "--country",
        type=str,
        help="Choice of which country to use for training - options are 'all', 'nigeria' or 'senegal'",
        default="all",
        choices=["all", "nigeria", "senegal"],
    )
    parser.add_argument(
        "-ip",
        "--interpretable",
        action="store_true",
        help="Make model (more) interpretable - no matter other flags, use only base (non auto-encoder) features so can explain",
    )
    parser.add_argument(
        "--universal-data-only",
        "-univ",
        action="store_true",
        help="Use only universal data (i.e. no country-specific data) - only applicable if --country!=all",
    )
    parser.add_argument(
        "--copy-to-nbrs",
        "-cp2nbr",
        action="store_true",
        help="Use expanded dataset, where 'ground-truth' values are copied to neighbouring cells",
    )
    parser.add_argument(
        "--aug_data", action="store_true", help="Augment data with group features"
    )
    parser.add_argument(
        "--subsel_data", action="store_true", help="Use feature subset selection"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Prefix to name the saved models / checkpoints",
    )
    parser.add_argument("--n_runs", type=int, default=1, help="Number of runs")

    parser.add_argument(
        "--model",
        type=str,
        default="automl",
        choices=[
            "lgbm",
            "automl",
            # "xgb",
            # "huber",
            # "krr",
        ],
        help="Choice of model to train (and tune)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data to exclude for test evaluation, default is 0.2",
    )
    parser.add_argument(
        "--nfolds",
        type=int,
        default=5,
        help="Number of folds of training set for cross validation, default is 5",
    )
    parser.add_argument(
        "--cv-type",
        type=str,
        default="normal",
        choices=["normal", "stratified", "spatial"],
        help="Type of CV to use, default is normal, choices are normal (fully random), stratified and spatial",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="all",
        choices=[
            "all",
            "education",
            "sanitation",
            "housing",
            "water",
            "av-severity",
            "av-prevalence",
        ],
        help="Target variable to use for training, default is all, choices are 'all' (train separate model for each of the following), 'av-severity' (average number of deprivations / child), 'av-prevalence' (average proportion of children with at least one deprivation), or proportion of children deprived in 'education', 'sanitation', 'housing', 'water'",
    )
    parser.add_argument(
        "--impute-gdp",
        type=str,
        default=None,
        choices=[None, "mean", "knn", "linear", "rf"],
        help="Impute GDP values prior to training, or leave as nan (default option)",
    )
    parser.add_argument(
        "--standardise",
        type=str,
        default=None,
        choices=[None, "standard", "minmax", "robust"],
        help="Standardise feature data prior to fitting model, options are None (default, leave raw), standard (z-score), minmax (min-max normalisation to limit to 0-1 range), or robust (median and quantile version of z-score)",
    )
    parser.add_argument(
        "--target-transform",
        type=str,
        default=None,
        choices=[None, "log", "power"],
        help="Transform target variable(s) prior to fitting model - choices of None (default, leave raw), 'log', 'power' (Yeo-Johnson)",
    )
    parser.add_argument(
        "--log-run",
        action="store_true",
        help="Use MLflow to log training run params + scores, by default in a /models/mlruns directory where /models is contained in same parent folder as args.data",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Save trained models (pickled), by default in a /models directory contained in same parent folder as args.data",
    )

    try:
        args = parser.parse_args()
    except argparse.ArgumentError:
        parser.print_help()
        sys.exit(0)

    DATA_DIRECTORY = Path(args.data)
    if args.save_model:
        SAVE_DIRECTORY = DATA_DIRECTORY.parent / "models"
        SAVE_DIRECTORY.mkdir(exist_ok=True)

    ### Load data
    # TODO: link w Dani's data generating pipeline
    if args.cp2nbr:
        # Load all NGA data (including expanded data)
        # TODO: include option to run on expanded dataset
        raise NotImplementedError("Not yet implemented")
    else:
        # Load all NGA data, using only specified (perturbed) locations
        XY = pd.read_csv(Path(args.data) / args.clean_name)
    if args.country != "all":
        # Want to either use preexisting data in location specified,
        # else produce data from scratch
        import geopandas as gpd
        import h3.api.numpy_int as h3
        from shapely.geometry import Point

        world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
        # world[world.name == "Nigeria"].geometry.__geo_interface__['features'][0]['geometry']
        ctry_name = args.country.capitalize()
        ctry_geom = world[world.name == ctry_name].geometry.values[0]
        XY = XY[
            XY.hex_code.swifter.apply(
                lambda x: Point(h3.h3_to_geo(x)[::-1])
            ).swifter.apply(lambda pt: pt.within(ctry_geom))
        ]

    XY["name_commuting_zone"] = XY["name_commuting_zone"].astype("category")
    # TODO: consider including hex count threshold here
    # thr_df = pd.read_csv(thr_data)
    # thr_all = all_df.set_index('hex_code').loc[thr_df.hex_code].reset_index()
    #### Select features to use
    start_idx = XY.columns.tolist().index("LATNUM")
    X = XY.iloc[:, start_idx:].copy()

    if args.univ:
        # Remove country specific data - e.g. in case of Nigeria,
        # conflict and healthcare data, and FB connectivity data
        pass
    if args.interpretable:
        # Remove auto-encoder features for more interpretable models
        pass
    #### Select target variables
    if args.target != "all":
        if args.target == "av-severity":
            target_name = "sumpoor_sev"
        elif args.target == "av-prevalence":
            target_name = "deprived_sev"
        else:
            target_name = f"dep_{args.target}_sev"
        Y = XY[target_name].copy()
    else:
        good_idxs = ["housing", "water", "sanitation", "education"]
        Y = XY[
            list(map(lambda x: f"dep_{x}_sev", good_idxs))
            + ["sumpoor_sev", "deprived_sev"]
        ].copy()

    X = None
    Y = None
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    if args.model == "automl":
        model = AutoML()
        # automl_pipeline
        automl_settings = {
            "time_budget": 60,  # total running time in seconds
            "metric": "mse",  # primary metrics for regression can be chosen from: ['mae','mse','r2']
            "task": "regression",  # task type
            "estimator_list": ["xgboost", "catboost", "lgbm"],
            "log_file_name": "automl.log",  # flaml log file
            "seed": 42,  # random seed
        }
        pipeline_settings = {
            f"automl__{key}": value for key, value in automl_settings.items()
        }
    else:
        raise NotImplementedError("Model not implemented")

    if args.impute_gdp is not None:
        imp = IterativeImputer(max_iter=10, random_state=42)
        imputer = SimpleImputer()

    set_config(display="diagram")

    if args.standardise is not None:
        standardiser = StandardScaler()

    pipeline = Pipeline(
        [("imputer", imputer), ("standardiser", standardiser), ("model", model)]
    )
    if args.log_run:
        MLFLOW_DIR = DATA_DIRECTORY.parent / "models" / "mlruns"
        MLFLOW_DIR.mkdir(exist_ok=True)

        mlflow.set_tracking_uri(MLFLOW_DIR)
        client = mlflow.tracking.MlflowClient()
        try:
            # Create an experiment name, which must be unique and case sensitive
            experiment_id = client.create_experiment(
                f"{args.country}-{args.target}-{args.model}"
            )
            # experiment = client.get_experiment(experiment_id)
        except ValueError:
            assert f"{args.country}-{args.target}-{args.model}" in [
                exp.name for exp in client.list_experiments()
            ]
            experiment_id = [
                exp.experiment_id
                for exp in client.list_experiments()
                if exp.name == f"{args.country}-{args.target}-{args.model}"
            ][0]
        mlflow.start_run(experiment_id=experiment_id)
    if len(Y.shape) == 1:
        pipeline.fit(X_train, Y_train, **pipeline_settings)
    else:
        for col_idx in range(Y_train.shape[1]):
            pipeline.fit(X_train, Y_train.values[:, col_idx], **pipeline_settings)

    if args.model == "automl":
        # get automl object back
        automl = pipeline.steps[2][1]
        # Get the best config and best learner
        print("Best ML learner:", automl.best_estimator)
        print("Best hyperparmeter config:", automl.best_config)
        print(f"Best accuracy on validation data: {1 - automl.best_loss:.4g}")
        print(
            "Training duration of best run: {:.4g} s".format(
                automl.best_config_train_time
            )
        )

        # plot basic feature importances
        # plt.barh(automl.feature_names_in_, automl.feature_importances_)

        # compute different metrics on test set
        y_pred = pipeline.predict(X_test)
        y_test = Y_test.values[:, col_idx]
        r2_val = 1 - sklearn_metric_loss_score("r2", y_pred, y_test)
        print("r2", "=", r2_val)
        mse_val = sklearn_metric_loss_score("mse", y_pred, y_test)
        print("mse", "=", mse_val)
        mae_val = sklearn_metric_loss_score("mae", y_pred, y_test)
        print("mae", "=", mae_val)
        if args.log_run:
            mlflow.log_param(key="best_model", value=automl.best_estimator)
            mlflow.log_params(automl.best_config)
            mlflow.log_metric(key="r2_score", value=r2_val)
            mlflow.log_metric(
                key="rmse",
                value=np.sqrt(mse_val),
            )
            mlflow.log_metric(
                key="mae",
                value=mae_val,
            )
    if args.log_run:
        mlflow.end_run()
