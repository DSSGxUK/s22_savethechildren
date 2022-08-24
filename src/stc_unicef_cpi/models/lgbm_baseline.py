import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
import argparse
import sys
import warnings
from pathlib import Path
from pprint import pprint

import lightgbm as lgb
import mlflow
import optuna  # pip install optuna
import optuna.integration.lightgbm as lgb_optuna
from flaml import AutoML
from optuna.integration import LightGBMPruningCallback
from sklearn.compose import (
    ColumnTransformer,
    make_column_selector,
    make_column_transformer,
)
from sklearn.impute import SimpleImputer
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
optuna.logging.set_verbosity(
    optuna.logging.ERROR
)  # prints errors but no warning or higher level info

# TODO: Use poetry to properly save environment for transfer


def adjusted_rsquared(r2, n, p):
    return 1 - (1 - r2) * ((n - 1) / (n - p - 1))


# Helper function


def get_card_split(df, cols, n=11):
    """
    Splits categorical columns into 2 lists based on cardinality (i.e # of unique values)
    Parameters
    ----------
    df : Pandas DataFrame
        DataFrame from which the cardinality of the columns is calculated.
    cols : list-like
        Categorical columns to list
    n : int, optional (default=11)
        The value of 'n' will be used to split columns.
    Returns
    -------
    card_low : list-like
        Columns with cardinality < n
    card_high : list-like
        Columns with cardinality >= n
    """
    cond = df[cols].nunique() > n
    card_high = cols[cond]
    card_low = cols[~cond]
    return card_low, card_high


def basic_model_pipeline(
    model, num_scaler="robust", cat_encoder="ohe", imputer="simple"
):
    """From given sklearn style model make simple pipeline, along with suitable transformers
    Alt construction to below that takes advantage of sklearn fns

    Args:
        model (_type_): _description_
        num_scaler (str, optional): _description_. Defaults to 'robust'.
        cat_encoder (str, optional): _description_. Defaults to 'ohe'.
        imputer (str, optional): _description_. Defaults to 'simple'.
    """
    # TODO: generalise choice of transformations both here and in basic_preprocessor below
    if num_scaler == "robust":
        num_tf = RobustScaler()
    if cat_encoder == "ohe":
        cat_tf = OneHotEncoder()
    if imputer == "simple":
        miss_tf = SimpleImputer()
    col_transformer = make_column_transformer(
        (num_tf, make_column_selector(dtype_include=np.number)),
        (cat_tf, make_column_selector(dtype_exclude=np.number)),
    )
    pipeline = make_pipeline(miss_tf, col_transformer, model)
    return pipeline


def basic_preprocessor(X_train, model_type="lgb"):
    numeric_features = X_train.select_dtypes(include=[np.number]).columns
    categorical_features = X_train.select_dtypes(include=["object"]).columns

    # NB while should generally encode categorical features in some way as follows,
    # LGBM actually allows direct support of categorical pandas columns
    categorical_low, categorical_high = get_card_split(X_train, categorical_features)

    if model_type != "lgb":
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_transformer_low = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("encoding", OneHotEncoder(handle_unknown="ignore", sparse=False)),
            ]
        )

        categorical_transformer_high = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                # 'OrdianlEncoder' Raise a ValueError when encounters an unknown value. Check https://github.com/scikit-learn/scikit-learn/pull/13423
                ("encoding", OrdinalEncoder()),
            ]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", numeric_transformer, numeric_features),
                ("categorical_low", categorical_transformer_low, categorical_low),
                ("categorical_high", categorical_transformer_high, categorical_high),
            ]
        )
        return preprocessor
    else:
        X_train[categorical_features] = X_train[categorical_features].astype("category")


def train_model(
    X_train,
    Y_train,
    X_test,
    Y_test,
    log_run=True,
    target_name="",
    model="lgb",
    experiment_name="nga-cpi",
):
    """Train baseline model

    :param X_train: _description_
    :type X_train: _type_
    :param Y_train: _description_
    :type Y_train: _type_
    :param X_test: _description_
    :type X_test: _type_
    :param Y_test: _description_
    :type Y_test: _type_
    :param log_run: _description_, defaults to True
    :type log_run: bool, optional
    :param target_name: _description_, defaults to ""
    :type target_name: str, optional
    :param model: _description_, defaults to "lgb"
    :type model: str, optional
    :param experiment_name: _description_, defaults to "nga-cpi"
    :type experiment_name: str, optional
    :return: _description_
    :rtype: _type_
    """
    models = {}
    scores = np.empty(len(Y_train.columns))
    for idx, col in tqdm(enumerate(Y_train.columns)):
        if log_run:
            client = mlflow.tracking.MlflowClient()
            try:
                # Create an experiment name, which must be unique and case sensitive
                experiment_id = client.create_experiment(
                    f"{experiment_name}: {target_name}"
                )
                # experiment = client.get_experiment(experiment_id)
            except ValueError:
                assert f"{experiment_name}: {target_name}" in [
                    exp.name for exp in client.list_experiments()
                ]
                experiment_id = [
                    exp.experiment_id
                    for exp in client.list_experiments()
                    if exp.name == f"{experiment_name}: {target_name}"
                ][0]
            mlflow.autolog(
                log_models=False,
                exclusive=True,
                registered_model_name=f"{experiment_name}:{target_name}_{model}",
            )
            with mlflow.start_run(experiment_id=experiment_id) as run:
                print(run.info)
                # TODO: finish mlflow logging for automl properly
                y_test = Y_test[col]
                preds = model.predict(X_test)
                mae_score = mae(y_test, preds)
                mlflow.log_metric("MAE", mae_score)
        else:
            if model == "lgb":
                lgb_reg = lgb.LGBMRegressor
                model = lgb_reg
                basic_preprocessor(X_train)
            elif model == "automl":
                model = AutoML()
            else:
                model = Pipeline(
                    steps=[
                        ("preprocessor", basic_preprocessor(X_train, model_type=None)),
                        ("regressor", model),
                    ]
                )
            if model == "automl":
                model.fit(X_train, Y_train[col], task="regression", metric="mae")
            else:
                model.fit(X_train, Y_train[col])
            Y_pred = model.predict(X_test)
            r_squared = r2_score(Y_test[col], Y_pred)
            adj_rsquared = adjusted_rsquared(
                r_squared, X_test.shape[0], X_test.shape[1]
            )
            rmse = np.sqrt(mean_squared_error(Y_test[col], Y_pred))
            print(f"For target {col}:")
            print(f"R^2 = {r_squared}, Adj.R^2 = {adj_rsquared}, RMSE = {rmse}")
            scores[idx] = rmse
            models[col] = model
    return models, scores


def lgbmreg_optuna(
    X_train,
    X_test,
    y_train,
    y_test,
    log_run=True,
    target_name="test",
    logging_level=optuna.logging.ERROR,
    experiment_name="nga-cpi",
    tracking_uri="../models/mlruns",
):
    """Use optuna / FLAML to train tuned LGBMRegressor
    NB expect target y to be a vector due to computational expense, and desire to log runs separately
    If need be run in loop
    Assume feature engineering etc already performed if desired

    Note some thoughts in various blog posts e.g. here
    https://towardsdatascience.com/kagglers-guide-to-lightgbm-hyperparameter-tuning-with-optuna-in-2021-ed048d9838b5

    Or just directly from docs
    https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html

    Main params to target:

    * num_leaves (max. limit should be 2^(max_depth) according to docs)
      - number of decision points in tree, given max_depth relatively
        easy to choose, but expensive so choose conservative range
        e.g. (20,3000)
    * max_depth
      - number of levels, more makes more complex and prone to overfit,
        too few and will underfit. Kaggle finds values of 3-12 works
        well for most datasets
    * min_data_in_leaf
      - min num observations that fit dec. crit. of each leaf,
        should be >100 for larger datasets as helps prevent overfitting
    * n_estimators
      - Number of decision trees used - larger will be slower but should
            be more accurate
    * learning_rate
      - step size param of gradient descent at each iteration, with
            typical values between 0.01 and 0.3, sometimes lower. Perfect
            setup w n_estimators is many trees w early stopping and low
            lr
    * max_bin
      - default already 255, likely to cause overfitting if increased
    * reg_alpha or _lambda
      - L1 / L2 regularisation - good search range usually (0,100) for
            both
    * min_gain_to_split
      - conservative search range is (0,15), can help regularisation
    * bagging_fraction and feature_fraction
      - proportion of training samples (within (0,1), needs bagging_freq
            set to an integer also) and proportion of features (also in (0,1)),
            respectively used to train each tree. Both can again help with
            overfitting
    * objective
      - the learning objective used, which can be custom (!)

    Additionally use MLflow to log the run unless specified not to


    Args:
        X_train (_type_): _description_
        X_test (_type_): _description_
        y_train (_type_): _description_
        y_test (_type_): _description_
        log_run (bool, optional): _description_. Defaults to True.
        target_name (str, optional): _description_. Defaults to "test".
        logging_level (_type_, optional): _description_. Defaults to optuna.logging.ERROR.
        experiment_name (str, optional): _description_. Defaults to "nga-cpi".

    Returns:
        _type_: _description_
    """
    optuna.logging.set_verbosity(logging_level)
    study = optuna.create_study(
        direction="minimize", study_name=f"LGBM Regressor: {target_name}"
    )

    def optim_func(trial):
        return objective(trial, X_train, y_train)

    study.optimize(optim_func, n_trials=20, callbacks=[callback])

    print(f"\tBest value over val folds (MAE): {study.best_value:.5f}")
    print("\tBest params:")
    for key, value in study.best_params.items():
        print(f"\t\t{key}: {value}")
    best_model = study.user_attrs["best_model"]

    print(
        f"MAE of best model on test set: {mae(y_test,best_model.predict(X_test)):.5f}"
    )
    if log_run:
        """Enables (or disables) and configures autologging from
        LightGBM to MLflow. Logs the following:

            - parameters specified in lightgbm.train.
            - metrics on each iteration (if valid_sets specified).
            - metrics at the best iteration (if early_stopping_rounds
              specified or early_stopping callback is set).
            - feature importance (both “split” and “gain”) as JSON
              files and plots.
            - trained model (unless include log_models=False flag),
              including:
                - an example of valid input.
                - inferred signature of the inputs and outputs of the model.

        """
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient()
        try:
            # Create an experiment name, which must be unique and case sensitive
            experiment_id = client.create_experiment(
                f"{experiment_name}: {target_name}"
            )
            # experiment = client.get_experiment(experiment_id)
        except ValueError:
            assert f"{experiment_name}: {target_name}" in [
                exp.name for exp in client.list_experiments()
            ]
            experiment_id = [
                exp.experiment_id
                for exp in client.list_experiments()
                if exp.name == f"{experiment_name}: {target_name}"
            ][0]
        mlflow.lightgbm.autolog(
            registered_model_name=f"{experiment_name}:{target_name}_{model}"
        )
        # NB this assumes using solely lgbm, not e.g. lgbm contained
        # inside an sklearn pipeline in which case should use
        # mlflow.sklearn.autolog(...)
        train_set = lgb.Dataset(X_train, label=y_train)
        with mlflow.start_run(experiment_id=experiment_id) as run:
            print(run.info)
            # train model
            params = study.best_params
            # log params - shouldn't be necessary but looping seems to cause issues
            mlflow.log_params(params)
            params.update({"verbosity": -1})
            model = lgb.train(
                params,
                train_set,
                # num_boost_round=10,
                valid_sets=[train_set],
                valid_names=["train"],
            )

            # evaluate model
            preds = model.predict(X_test)
            loss = mae(y_test, preds)

            # log metrics
            mlflow.log_metrics({"MAE": loss})

            print(f"Logged data and model in run: {run.info.run_id}")

            # show logged data
            for key, data in fetch_logged_data(run.info.run_id).items():
                print(f"\n---------- logged {key} ----------")
                pprint(data)
        # disable autologging so doesn't cause problems for next run w optuna
        mlflow.lightgbm.autolog(disable=True)
        return model, loss
    else:
        return best_model, study.best_value


def lgbmreg_optunaCV(
    X_train,
    X_test,
    y_train,
    y_test,
    log_run=True,
    target_name="test",
    logging_level=optuna.logging.ERROR,
    experiment_name="nga-cpi",
):
    """Use optuna default tuner CV instead of above definition - only optimises

        - lambda_l1
        - lambda_l2
        - num_leaves
        - feature_fraction
        - bagging_fraction
        - bagging_freq
        - min_child_samples

    Additionally use MLflow to log the run unless specified not to

    Args:
        X_train (_type_): _description_
        X_test (_type_): _description_
        y_train (_type_): _description_
        y_test (_type_): _description_
        log_run (bool, optional): _description_. Defaults to True.
        target_name (str, optional): _description_. Defaults to "test".
        logging_level (_type_, optional): _description_. Defaults to optuna.logging.ERROR.
        experiment_name (_type_, optional): _description_. Defaults to "nga-cpi".

    Returns:
        _type_: _description_
    """
    optuna.logging.set_verbosity(logging_level)
    study = optuna.create_study(
        direction="minimize", study_name=f"LGBM Regressor {target_name}"
    )
    dtrain = lgb_optuna.Dataset(X_train, y_train)
    params = {
        "objective": "regression",
        "metric": "l2",
        "boosting_type": "gbdt",
    }
    tuner = lgb_optuna.LightGBMTunerCV(
        params,
        dtrain,
        folds=KFold(n_splits=5),
        study=study,
        # callbacks=[LightGBMPruningCallback(trial, "l2")],
        callbacks=[lgb.early_stopping(100)],
        return_cvbooster=True,
    )
    tuner.run()

    print(f"\tBest value over val folds (MAE): {tuner.best_score:.5f}")
    print("\tBest params:")
    for key, value in tuner.best_params.items():
        print(f"\t\t{key}: {value}")
    best_model = tuner.get_best_booster()
    try:
        preds = best_model.predict(X_test)
        if len(preds.shape) == 1:
            print(f"MAE of best model on test set: {mae(y_test,preds):.5f}")
        elif len(preds.shape) == 2:
            print(
                f"MAE of best model on test set: {np.mean([mae(y_test,pred) for pred in preds]):.5f}"
            )
    except AttributeError:
        print(
            f"MAE of best model on test set: {np.mean([mae(y_test,pred) for pred in preds]):.5f}"
        )
    except ValueError:
        print("Tried to print MAE... failed")
        return best_model, (tuner.best_score, best_model.predict(X_test))

    if log_run:
        """Enables (or disables) and configures autologging from
        LightGBM to MLflow. Logs the following:
            - parameters specified in lightgbm.train.
            - metrics on each iteration (if valid_sets specified).
            - metrics at the best iteration (if early_stopping_rounds
              specified or early_stopping callback is set).
            - feature importance (both “split” and “gain”) as JSON
              files and plots.
            - trained model (unless include log_models=False flag),
              including:
                - an example of valid input.
                - inferred signature of the inputs and outputs of the model.
        """
        client = mlflow.tracking.MlflowClient()
        try:
            # Create an experiment name, which must be unique and case sensitive
            experiment_id = client.create_experiment(
                f"{experiment_name}: {target_name}"
            )
            # experiment = client.get_experiment(experiment_id)
        except:
            assert f"{experiment_name}: {target_name}" in [
                exp.name for exp in client.list_experiments()
            ]
            experiment_id = [
                exp.experiment_id
                for exp in client.list_experiments()
                if exp.name == f"{experiment_name}: {target_name}"
            ][0]
        mlflow.lightgbm.autolog(
            registered_model_name=f"{experiment_name}:{target_name}_lgbm"
        )
        # NB this assumes using solely lgbm, not e.g. lgbm contained
        # inside an sklearn pipeline in which case should use
        # mlflow.sklearn.autolog(...)
        train_set = lgb.Dataset(X_train, label=y_train)
        with mlflow.start_run(experiment_id=experiment_id) as run:
            print(run.info)
            # train model
            params = tuner.best_params
            # log params - shouldn't be necessary but looping seems to cause issues
            mlflow.log_params(params)
            params.update({"verbosity": -1})
            model = lgb.train(
                params,
                train_set,
                # num_boost_round=10,
                valid_sets=[train_set],
                valid_names=["train"],
            )

            # evaluate model
            preds = model.predict(X_test)
            loss = mae(y_test, preds)

            # log metrics
            mlflow.log_metrics({"MAE": loss})

            print(f"Logged data and model in run: {run.info.run_id}")

            # show logged data
            for key, data in fetch_logged_data(run.info.run_id).items():
                print(f"\n---------- logged {key} ----------")
                pprint(data)
        # disable autologging so doesn't cause problems for next run w optuna
        mlflow.lightgbm.autolog(disable=True)
        return model, loss
    else:
        return best_model, study.best_value


# ex lgbm optuna code modified from blog above
def objective(trial, X, y):
    param_grid = {
        # "device_type": trial.suggest_categorical("device_type", ['gpu']),
        "n_estimators": trial.suggest_categorical("n_estimators", [1000, 10000]),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 4, 400, step=4),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 2, 20, step=1),
        "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
        "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
        "bagging_fraction": trial.suggest_float(
            "bagging_fraction", 0.2, 0.95, step=0.1
        ),
        "bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
        "feature_fraction": trial.suggest_float(
            "feature_fraction", 0.3, 1.0, step=0.05
        ),
        "num_boost_round": trial.suggest_int("num_boost_round", 3, 10),
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    cv_scores = np.empty(5)
    for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = lgb.LGBMRegressor(**param_grid, verbosity=-1)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            eval_metric="l2",
            early_stopping_rounds=100,
            callbacks=[
                LightGBMPruningCallback(trial, "l2")
            ],  # Add a pruning callback that detects unpromising hyperparameter sets before
            # training them on data, saving a lot of time - l1 is alt for mae
        )
        preds = model.predict(X_test)
        cv_scores[idx] = mae(y_test, preds)
    trial.set_user_attr(key="best_model", value=model)
    return np.mean(cv_scores)


def callback(study, trial):
    if study.best_trial.number == trial.number:
        study.set_user_attr(key="best_model", value=trial.user_attrs["best_model"])


def flaml_multireg(
    X,
    Y,
    log_run=True,
    time_budget=60,
    scorer=lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
):
    # split into train and test data
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.30, random_state=42
    )

    if log_run:
        experiment = mlflow.set_experiment("flaml_cpi")
        experiment_id = experiment.experiment_id
        with mlflow.start_run(experiment_id=experiment_id) as run:
            print(run.info)
            # train the model - by default AutoML for regression will try 'lgbm', 'rf', 'catboost', 'xgboost', 'extra_tree', so good base selection
            model = MultiOutputRegressor(
                AutoML(task="regression", metric="rmse", time_budget=60)
            )
            model.fit(X_train, Y_train)

            # predict
            preds = model.predict(X_test)
            mlflow.sklearn.log_model(model, "automl")
            # score
            scores = scorer(Y_test, preds)
            mlflow.log_metric("mean_score", scores.mean())
    else:
        # train the model
        model = MultiOutputRegressor(
            AutoML(task="regression", metric="rmse", time_budget=60)
        )
        model.fit(X_train, Y_train)

        # predict
        preds = model.predict(X_test)

        # score
        scores = scorer(Y_test, preds)
    return model, scores


# Stacking model code ex
# from sklearn.ensemble import StackingRegressor

# # Decision trees
# from catboost import CatBoostRegressor
# from xgboost import XGBRegressor

# # # Neural networks
# # from tensorflow.keras.models import Sequential
# # from tensorflow.keras.layers import Dense
# # from tensorflow.keras.layers import Dropout
# # from tensorflow.keras import regularizers
# # # Wrapper to make neural network compitable with StackingRegressor
# # from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
# # Linear model as meta-learn
# from sklearn.linear_model import LinearRegression


# def get_stacking(model_list=None):
#     """A stacking model that consists of CatBoostRegressor,
#     XGBRegressor, and a linear model, or a specified list of models"""
#     # First we create a list called "level0", which consists of our base models"
#     # These models will get passed down to the meta-learner later
#     level0 = list()
#     if model_list == None:
#         level0.append(("cat", CatBoostRegressor(verbose=False)))
#         level0.append(("cat2", CatBoostRegressor(verbose=False, learning_rate=0.0001)))
#         level0.append(("xgb", XGBRegressor()))
#         level0.append(("xgb2", XGBRegressor(max_depth=5, learning_rate=0.0001)))
#         level0.append(("linear", LinearRegression()))
#     else:
#         model_names = [
#             str(type(mod["regressor"])).strip("'>").split(".")[-1] for mod in model_list
#         ]
#         [
#             level0.append((model_name, model))
#             for model_name, model in zip(model_names, model_list)
#         ]

#     # The "meta-learner" designated as the level1 model
#     # Linear Regression often performs best
#     # but feel free to experiment with other models
#     level1 = LinearRegression()
#     # Create the stacking ensemble
#     model = StackingRegressor(
#         estimators=level0, final_estimator=level1, cv=5, verbose=1
#     )
#     return model
# # Create stacking model
# model = get_stacking()
# model.fit(X_train, Y_train.iloc[:, 0].values.ravel())
# # Creating a temporary dataframe so we can see how each of our models performed
# temp = pd.DataFrame(Y_test.iloc[:, 0])
# # The stacked models predictions, which should perform the best
# temp["stacking_prediction"] = model.predict(X_test)
# # Get each model in the stacked model to see how they individually perform
# for m in model.named_estimators_:
#     temp[m] = model.named_estimators_[m].predict(X_test)
# # See how each of our models correlate with our target
# print(temp.corr()[Y_test.columns[0]])
# # See what our meta-learner is thinking (the linear regression)
# for coef in zip(model.named_estimators_, model.final_estimator_.coef_):
#     print(coef)

if __name__ == "__main__":
    ### Argument and global variables
    SAVE_DIRECTORY = Path("../../../models")
    SAVE_DIRECTORY.mkdir(exist_ok=True)
    DATA_DIRECTORY = Path("../../../data")
    parser = argparse.ArgumentParser("High-res multi-dim CPI model training")
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        help="Pathway to data",
        default=DATA_DIRECTORY,
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save models and scores",
    )
    parser.add_argument(
        "--aug_data", action="store_true", help="Augment data with group features"
    )
    parser.add_argument(
        "--subsel_data", action="store_true", help="Use feature subset selection"
    )
    parser.add_argument(
        "-ip",
        "--interpretable",
        action="store_true",
        help="Make model (more) interpretable - no matter other flags, use only base features so can explain",
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
        default="lgbm",
        choices=[
            "lgbm",
            "automl",
            # "xgb",
            # "huber",
            # "krr",
        ],
        help="Choice of model to train (and tune)",
    )
    try:
        args = parser.parse_args()
    except argparse.ArgumentError:
        parser.print_help()
        sys.exit(0)
    # access w e.g. args.aug_data etc.
    X = None
    Y = None
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    col_idx = 5
    best_model = {}
    maes = np.zeros(len(Y_train.columns))
    for i, col in enumerate(Y_train.columns):
        print()
        print("#" * 12, f"At target {col}", "#" * 12)
        idx_mod, mae_val = lgbmreg_optuna(
            X_train,
            X_test,
            Y_train[col],  # Y_train.iloc[:, col_idx],
            Y_test[col],  # Y_test.iloc[:, col_idx],
            log_run=True,
            target_name=col,  # Y_train.columns[col_idx],
        )
        best_model[col] = idx_mod
        maes[i] = mae_val
    print(f"Mean MAE across all targets: {maes.mean()}")
