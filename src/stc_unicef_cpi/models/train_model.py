import argparse
import pickle
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import swifter
from flaml import AutoML
from flaml.ml import sklearn_metric_loss_score
from sklearn import clone, set_config
from sklearn.compose import (
    ColumnTransformer,
    make_column_selector,
    make_column_transformer,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score  # , log_loss
from sklearn.model_selection import GroupKFold, KFold, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)
from tqdm.auto import tqdm

from stc_unicef_cpi.data.cv_loaders import HexSpatialKFold, StratifiedIntervalKFold
from stc_unicef_cpi.features.build_features import boruta_shap_ftr_select
from stc_unicef_cpi.utils.mlflow_utils import fetch_logged_data
from stc_unicef_cpi.utils.scoring import mae

# TODO: write proper warning handler to only suppress unhelpful msgs
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    ### Argument and global variables
    parser = argparse.ArgumentParser("High-res multi-dim CPI model training")
    DATA_DIRECTORY = Path("../../../data/processed")
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
        default="new_auto_thr_clean_nga.csv",
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
    # parser.add_argument(
    #     "--aug_data", action="store_true", help="Augment data with group features"
    # )
    parser.add_argument(
        "--ftr-impt",
        action="store_true",
        help="Investigate final model feature importance using BorutaShap",
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
            "catboost",
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
        "--eval-split-type",
        type=str,
        default="normal",
        choices=["normal", "stratified", "spatial"],
        help="Method to split test from training set, default is normal, choices are normal (fully random), stratified and spatial",
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
            "av-2-prevalence",
            "health",
            "nutrition",
            "av-3-prevalence",
            "av-4-prevalence",
        ],
        help="Target variable to use for training, default is all, choices are 'all' (train separate model for each of the following), 'av-severity' (average number of deprivations / child), 'av-prevalence' (average proportion of children with at least one deprivation), 'av-2-prevalence' (average proportion of children with at least two deprivations), proportion of children deprived in 'education', 'sanitation', 'housing', 'water'. May also pass 'health' or 'nutrition' but limited ground truth data increases model variance. Similarly may pass 'av-3-prevalence' or 'av-4-prevalence', but ~50pc of cell data is exactly zero for 3, and ~80pc for 4, so again causes modelling issues.",
    )
    parser.add_argument(
        "--impute",
        type=str,
        default="none",
        choices=["none", "mean", "median", "knn", "linear", "rf"],
        help="Impute missing values prior to training, or leave as nan (default option)",
    )
    parser.add_argument(
        "--standardise",
        type=str,
        default="none",
        choices=["none", "standard", "minmax", "robust"],
        help="Standardise feature data prior to fitting model, options are none (default, leave raw), standard (z-score), minmax (min-max normalisation to limit to 0-1 range), or robust (median and quantile version of z-score)",
    )
    parser.add_argument(
        "--target-transform",
        type=str,
        default="none",
        choices=["none", "log", "power"],
        help="Transform target variable(s) prior to fitting model - choices of none (default, leave raw), 'log', 'power' (Yeo-Johnson)",
    )
    parser.add_argument(
        "--log-run",
        action="store_true",
        help="Use MLflow to log training run params + scores, by default in a /models/mlruns directory where /models is contained in same parent folder as args.data",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Save trained models (joblib pickled), by default in a /models directory contained in same parent folder as args.data",
    )
    parser.add_argument(
        "--automl-warm-start",
        action="store_true",
        help="When possible, use best model configuration found from previous runs to initialise hyperparameter search for each model.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Produce scatter plot(s) of predicted vs actual values on test set",
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
    # - Either load clean dataset, or pass data directory with all data
    # files necessary to produce clean data
    if args.copy_to_nbrs:
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

    # TODO: consider including hex count threshold here
    # thr_df = pd.read_csv(thr_data)
    # thr_all = all_df.set_index('hex_code').loc[thr_df.hex_code].reset_index()
    #### Select features to use
    start_idx = XY.columns.tolist().index("LATNUM")
    X = XY.iloc[:, start_idx:].copy()

    if args.universal_data_only:
        # Remove country specific data - e.g. in case of Nigeria,
        # conflict and healthcare data, and FB connectivity data
        if args.country == "nigeria":
            nga_spec_cols = [
                "n_conflicts",
                "n_education",
                "n_health",
                "OSM_hospital",
                "OSM_school",
                "health_gv_osm",
                "school_gv_osm",
                "estimate_dau",
            ]
            X = X.drop(nga_spec_cols, axis=1)
        else:
            print(
                "NB no additional features exist for countries other than Nigeria, so no additional features are removed"
            )

    if args.interpretable:
        # Remove auto-encoder features for more interpretable models
        auto_cols = [col for col in X.columns if "auto_" in col]
        X = X.drop(auto_cols, axis=1)

    #### Select target variables
    if args.target != "all":
        if args.target == "av-severity":
            target_name = "sumpoor_sev"
        elif args.target == "av-prevalence":
            target_name = "deprived_sev"
        elif args.target == "av-2-prevalence":
            target_name = "dep_2_or_more_sev"
        elif args.target == "av-3-prevalence":
            target_name = "dep_3_or_more_sev"
        elif args.target == "av-4-prevalence":
            target_name = "dep_4_or_more_sev"
        else:
            if args.target in ["health", "nutrition"]:
                warnings.warn(
                    f"Target variable {args.target} has very minimal ground truth data - model extrapolation likely to be poor"
                )
            target_name = f"dep_{args.target}_sev"
        Y = XY[target_name].copy()
    else:
        # NB only evaluating on good idxs here, if want to do health / nutrition need to pass explicitly.
        good_idxs = ["housing", "water", "sanitation", "education", "2_or_more"]
        Y = XY[
            list(map(lambda x: f"dep_{x}_sev", good_idxs))
            + ["sumpoor_sev", "deprived_sev"]
        ].copy()

    categorical_features = X.select_dtypes(exclude=[np.number]).columns
    X[categorical_features] = X[categorical_features].astype("category")

    # generate train / test split
    # choices = ["normal", "stratified", "spatial"]
    if args.eval_split_type == "normal":
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=args.test_size, random_state=42
        )
    elif args.eval_split_type == "stratified":
        if len(Y.shape) == 1:
            train_idxs, test_idxs = next(
                StratifiedIntervalKFold(
                    n_splits=int(1 / args.test_size), random_state=42
                ).split(X, Y)
            )
            X_train, Y_train = X[train_idxs], Y[train_idxs]
            X_test, Y_test = X[test_idxs], Y[test_idxs]
        else:
            # must generate split for each column separately
            strat_X_train = {}
            strat_Y_train = {}
            strat_X_test = {}
            strat_Y_test = {}
            for col_idx in range(Y.shape[1]):
                train_idxs, test_idxs = next(
                    StratifiedIntervalKFold(
                        n_splits=int(1 / args.test_size), random_state=42
                    ).split(X, Y.iloc[:, col_idx])
                )
                X_train, y_train = X[train_idxs], Y.iloc[:, col_idx][train_idxs]
                X_test, y_test = X[test_idxs], Y.iloc[:, col_idx][test_idxs]
                strat_X_train[col_idx] = X_train
                strat_Y_train[col_idx] = y_train
                strat_X_test[col_idx] = X_test
                strat_Y_test[col_idx] = y_test
    elif args.eval_split_type == "spatial":
        train_idxs, test_idxs = next(
            HexSpatialKFold(n_splits=int(1 / args.test_size), random_state=42).split(
                X, Y
            )
        )
        X_train, Y_train = X[train_idxs], Y[train_idxs]
        X_test, Y_test = X[test_idxs], Y[test_idxs]
    # specify KFold strategy
    # choices = ["normal", "stratified", "spatial"]
    if args.cv_type == "normal":
        kfold = KFold(n_splits=args.nfolds, shuffle=True)
    elif args.cv_type == "stratified":
        kfold = StratifiedIntervalKFold(n_splits=args.nfolds, shuffle=True, n_cuts=5)
    elif args.cv_type == "spatial":
        # print(X.iloc[:,-1].head())
        kfold = GroupKFold(n_splits=args.nfolds)
        spatial_groups = HexSpatialKFold(n_splits=args.nfolds).get_spatial_groups(
            XY["hex_code"].loc[X_train.index]
        )
        try:
            assert len(spatial_groups) == len(X_train)
        except AssertionError:
            print(spatial_groups.shape, X_train.shape)
    else:
        raise ValueError("Invalid CV type")
    # target transforms
    # choices = ["none", "log", "power"]
    if args.target_transform != "none":
        if args.target_transform == "log":
            Y_train = np.log(Y_train)
            Y_test = np.log(Y_test)
        elif args.target_transform == "power":
            power_tf = PowerTransformer().fit(Y_train)
            Y_train = power_tf.transform(Y_train)
            Y_test = power_tf.transform(Y_test)
        else:
            raise ValueError("Invalid target transform")

    pipeline_settings = dict()
    if args.model == "automl":
        model = AutoML()
        # automl_pipeline
        automl_settings = {
            "time_budget": 300,  # total running time in seconds for each target
            "metric": "mse",  # primary metrics for regression can be chosen from: ['mae','mse','r2']
            "task": "regression",  # task type
            "estimator_list": [
                "xgboost",
                "lgbm",
                # "catboost", # for some reason in flaml code, failing for spatial CV
                "rf",
                "extra_tree",
            ],
            "log_file_name": "automl.log",  # flaml log file
            "seed": 42,  # random seed
            "eval_method": "cv",
            "split_type": kfold,
            "groups": spatial_groups if args.cv_type == "spatial" else None,
        }
        pipeline_settings = {
            f"model__{key}": value for key, value in automl_settings.items()
        }
    else:
        raise NotImplementedError("Model not implemented")

    # imputer setup
    # choices = ["none", "mean", "median", "knn", "linear", "rf"]
    # NB these imputers will be applied to all features, but other than
    # ('discrete_classification-proba_mean', 84)
    # ('GDP_PPP_2015', 282)
    # ('GDP_PPP_1990', 458)
    # ('GDP_PPP_2000', 458)
    # ('avg_signal', 890)
    # ('avg_d_kbps', 1098)
    # ('avg_u_kbps', 1098)
    # ('avg_lat_ms', 1098)
    # there aren't substantial amounts of missing data for other features (max of 14 records in clean dataset)
    # optionally add an indicator feature which marks imputed records
    add_indicator = True
    if args.impute == "none":
        num_imputer = None
    elif args.impute == "mean":
        # default strategy is mean
        num_imputer = SimpleImputer(add_indicator=add_indicator)
    elif args.impute == "median":
        num_imputer = SimpleImputer(strategy="median", add_indicator=add_indicator)
    elif args.impute == "knn":
        num_imputer = KNNImputer(n_neighbors=5, add_indicator=add_indicator)
    elif args.impute == "linear":
        # default estimator is BayesianRidge
        num_imputer = IterativeImputer(
            max_iter=10, random_state=42, add_indicator=add_indicator
        )
    elif args.impute == "rf":
        num_imputer = IterativeImputer(
            estimator=RandomForestRegressor(
                n_estimators=20, min_samples_split=5, min_samples_leaf=3
            ),
            max_iter=10,
            random_state=42,
            add_indicator=add_indicator,
        )
    # as only commuting_zn is cat, just use constant imputation for this (only 5 missing records)
    cat_imputer = SimpleImputer(strategy="constant", fill_value="Unknown")

    set_config(display="diagram")

    # feature standardisation setup
    # choices = ["none", "standard", "minmax", "robust"]
    if args.standardise == "none":
        standardiser = None
    elif args.standardise == "standard":
        num_stand = StandardScaler()
    elif args.standardise == "minmax":
        num_stand = MinMaxScaler()
    elif args.standardise == "robust":
        num_stand = RobustScaler()

        # if cat_encoder == "ohe":
        # only real categorical column is commuting zone, so could do encoding here - however, interested really in GBMs, (LightGBM, XGBoost, CatBoost), all of which have
        # native support for categorical columns, so can neglect
        # If was bigger problem, could e.g. try category-encoders lib, https://contrib.scikit-learn.org/category_encoders/index.html
        # cat_enc = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=5)

    num_tf = Pipeline(steps=[("imputer", num_imputer), ("standardiser", num_stand)])
    cat_tf = Pipeline(
        steps=[
            ("imputer", cat_imputer),
            #  ("encoder", cat_enc)
        ]
    )
    col_tf = make_column_transformer(
        (num_tf, make_column_selector(dtype_include=np.number)),
        (cat_tf, make_column_selector(dtype_exclude=np.number)),
    )

    pipeline = Pipeline([("preprocessor", col_tf), ("model", model)])

    univ_data = "univ" if args.universal_data_only else "all"
    ip_data = "ip" if args.interpretable else "nip"
    if len(Y.shape) == 1:
        pipeline_desc = f"best_cfg-{args.country}-{args.target}-{args.cv_type}-{args.universal_data_only}-{univ_data}-{ip_data}-{args.impute}-{args.standardise}-{args.target_transform}.pkl"
        if args.automl_warm_start:
            try:
                with open(DATA_DIRECTORY.parent / "models" / pipeline_desc, "rb") as f:
                    start_pts = pickle.load(f)  # type: ignore
            except FileNotFoundError:
                print("No warm start available.")
        pipeline_settings["model__starting_points"] = start_pts
        pipeline.fit(
            X_train.reset_index(drop=True),
            Y_train.reset_index(drop=True),
            **pipeline_settings,
        )
    else:
        pipelines = [clone(pipeline) for _ in range(Y.shape[1])]
        for col_idx, pipeline in enumerate(pipelines):
            col = Y.columns[col_idx]
            if args.automl_warm_start:
                try:
                    pipeline_desc = f"best_cfg-{args.country}-{col}-{args.cv_type}-{args.universal_data_only}-{univ_data}-{ip_data}-{args.impute}-{args.standardise}-{args.target_transform}.pkl"
                    with open(
                        DATA_DIRECTORY.parent / "models" / pipeline_desc, "rb"
                    ) as f:
                        start_pts = pickle.load(f)  # type: ignore
                except FileNotFoundError:
                    print("No warm start available.")
            if args.eval_split_type == "stratified":
                X_train = strat_X_train[col_idx]
                y_train = strat_Y_train[col_idx]
            else:
                y_train = Y_train[col]
            pipeline.fit(
                X_train.reset_index(drop=True),
                y_train.reset_index(drop=True),
                **pipeline_settings,
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
        except:
            assert f"{args.country}-{args.target}-{args.model}" in [
                exp.name for exp in client.list_experiments()
            ]
            experiment_id = [
                exp.experiment_id
                for exp in client.list_experiments()
                if exp.name == f"{args.country}-{args.target}-{args.model}"
            ][0]
        mlflow.start_run(experiment_id=experiment_id)
    if args.model == "automl":
        # get automl object back
        if len(Y.shape) == 1:
            pipelines = [pipeline]
        for col_idx, pipeline in enumerate(pipelines):
            automl = pipeline.steps[1][1]
            # Get the best config and best learner
            if len(Y.shape) == 1:
                col = Y.name
            else:
                col = Y.columns[col_idx]
            print(f"Best ML learner for {col}:", automl.best_estimator)
            print("Best hyperparameter config:", automl.best_config)
            print(f"Best accuracy on validation data: {1 - automl.best_loss:.4g}")
            print(
                "Training duration of best run: {:.4g} s".format(
                    automl.best_config_train_time
                )
            )
            if args.ftr_impt:
                if len(Y.shape) > 1:
                    y = Y.iloc[:, col_idx]
                else:
                    y = Y
                ftr_names = pipeline[:-1].get_feature_names_out()
                ftr_names = [
                    name.split("__")[1] if "__" in name else name for name in ftr_names
                ]

                X_tf = pd.DataFrame(
                    pipeline[:-1].transform(X), columns=ftr_names, index=X.index
                )
                # ordinal encode categorical columns so fit method always works
                # but leave as type category so rest works out
                categorical_features = X_tf.select_dtypes(exclude=[np.number]).columns
                try:
                    X_tf[categorical_features] = pd.concat(
                        [
                            X_tf[cat_col].astype("category").cat.codes
                            for cat_col in categorical_features
                        ],
                        axis=1,
                    ).astype("category")
                except:
                    print(X_tf.columns)
                print(
                    X_tf[
                        [col for col in X_tf.columns if "commuting_zone" in col]
                    ].head()
                )
                # pipeline.steps[0][1].transform(X)
                ftr_subset = boruta_shap_ftr_select(
                    X_tf,
                    y,
                    base_model=clone(automl.model.estimator),
                    plot=True,
                    n_trials=100,
                    sample=False,
                    train_or_test="test",
                    normalize=True,
                    verbose=True,
                    incl_tentative=True,
                )
                print("Best ftr subset estimated to be")
                print(ftr_subset)

            # plot basic feature importances
            # plt.barh(automl.feature_names_in_, automl.feature_importances_)

            # compute different metrics on test set
            if len(Y.shape) > 1:
                if args.eval_split_type == "stratified":
                    X_test = strat_X_test[col_idx]
                    y_test = strat_Y_test[col_idx]
                else:
                    y_test = Y_test.values[:, col_idx]
            y_pred = pipeline.predict(X_test)
            r2_val = 1 - sklearn_metric_loss_score("r2", y_pred, y_test)
            print("r2", "=", r2_val)
            mse_val = sklearn_metric_loss_score("mse", y_pred, y_test)
            print("mse", "=", mse_val)
            mae_val = sklearn_metric_loss_score("mae", y_pred, y_test)
            print("mae", "=", mae_val)

            if args.plot:
                fig, ax = plt.subplots(dpi=200)
                sns.scatterplot(x=y_pred, y=y_test)
                ax.set_title(
                    f"{col}: R2 = {r2_val:.3f}, MSE = {mse_val:.3f}, MAE = {mae_val:.3f}"
                )
                ax.set_xlabel("Predicted value")
                ax.set_ylabel("True value")
                plt.show()
                FIG_DIR = DATA_DIRECTORY.parent / "figures"
                FIG_DIR.mkdir(exist_ok=True)
                if not args.log_run:
                    fig.savefig(
                        FIG_DIR / f"{col}_{args.model}.png", bbox_inches="tight"
                    )

            if args.log_run:
                with mlflow.start_run(
                    experiment_id=experiment_id, run_name=col, nested=True
                ) as run:
                    mlflow.set_tags(
                        {
                            "cv_type": args.cv_type,
                            "imputation": args.impute,
                            "standardisation": args.standardise,
                            "target_transform": args.target_transform,
                            "interpretable": args.interpretable,
                            "universal": args.universal_data_only,
                        }
                    )
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
                    if args.plot:
                        mlflow.log_figure(fig, f"figures/{col}_{args.model}.png")
                    if args.ftr_impt:
                        with open(
                            DATA_DIRECTORY.parent
                            / "models"
                            / f"{col}_{args.model}_ftr_subset.txt",
                            "w",
                        ) as f:
                            f.write(str(ftr_subset))
                        mlflow.log_artifact(
                            DATA_DIRECTORY.parent
                            / "models"
                            / f"{col}_{args.model}_ftr_subset.txt"
                        )
            if args.save_model:
                if args.target != "all":
                    model_desc = f"{args.country}-{args.target}-{args.cv_type}-{args.universal_data_only}-{univ_data}-{ip_data}-{args.impute}-{args.standardise}-{args.target_transform}"
                    pipeline_desc = f"best_cfg-{model_desc}.pkl"
                    with open(
                        SAVE_DIRECTORY / f"{model_desc}.pkl",
                        "wb",
                    ) as f:
                        joblib.dump(pipeline, f)
                    with open(SAVE_DIRECTORY / pipeline_desc, "wb") as f:
                        pickle.dump(automl.best_config_per_estimator, f)
                else:
                    # TODO: change model save description
                    model_desc = f"{args.country}-{col}-{args.cv_type}-{args.universal_data_only}-{univ_data}-{ip_data}-{args.impute}-{args.standardise}-{args.target_transform}"
                    pipeline_desc = f"best_cfg-{model_desc}.pkl"
                    with open(SAVE_DIRECTORY / f"{model_desc}.pkl", "wb") as f:
                        joblib.dump(pipeline, f)

                    with open(SAVE_DIRECTORY / pipeline_desc, "wb") as f:
                        pickle.dump(automl.best_config_per_estimator, f)

    if args.log_run:
        mlflow.end_run()
        # Marina - two stage model as suggested by Lyudmila?
