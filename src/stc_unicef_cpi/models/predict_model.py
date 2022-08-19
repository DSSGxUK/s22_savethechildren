# TODO: Finish script to use pretrained model to extrapolate for unseen areas

import argparse
import pickle
import sys
from pathlib import Path

import h3.api.numpy_int as h3
import joblib
import numpy as np
import pandas as pd
import swifter

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "High-res multi-dim CPI pretrained model predictions"
    )
    parser.add_argument(
        "--country",
        type=str,
        help="Choice of which country to predict for - options are 'all', 'nigeria' or 'senegal'",
        choices=[
            "all",
            "nigeria",
            "senegal",
            "togo",
            "benin",
            "guinea",
            "cameroon",
            "liberia",
            "sierra leone",
            "burkina faso",
        ],
    )
    MODEL_DIR = Path.cwd().parent.parent.parent / "data" / "models"
    DATA_DIR = MODEL_DIR.parent / "processed"

    parser.add_argument(
        "--data-dir",
        type=str,
        default=DATA_DIR,
        help="Pathway to processed data directory",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=MODEL_DIR,
        help="Pathway to pretrained model directory",
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
        type=str,
        default="false",
        choices=["true", "false"],
        help="Use only universal data (i.e. no country-specific data) - only applicable if --country!=all",
    )
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
        "--copy-to-nbrs",
        "-cp2nbr",
        type=str,
        default="false",
        choices=["true", "false"],
        help="Use model trained on expanded dataset",
    )
    parser.add_argument(
        "--resolution", "-res", type=int, default=7, help="Resolution of h3 grid"
    )

    # population threshold
    try:
        args = parser.parse_args()
    except argparse.ArgumentError:
        parser.print_help()
        sys.exit(0)

    # load model
    univ_data = "univ" if args.universal_data_only == "true" else "all"
    ip_data = "ip" if args.interpretable else "nip"

    MODEL_DIR = Path(args.model_dir)
    # model_name = f"{args.country}-{args.target}-{args.cv_type}-{args.universal_data_only}-{univ_data}-{ip_data}-{args.impute}-{args.standardise}-{args.target_transform}.pkl"
    if args.target != "all":
        if "av-" not in args.target:
            target_name = args.target
        else:
            pass

        model_name = f"{args.country}-dep_{args.target}_sev-{args.cv_type}-{args.universal_data_only}-{univ_data}-{ip_data}-{args.impute}-{args.standardise}-{args.target_transform}-{args.copy_to_nbrs}_res{args.resolution}.pkl"
        with open(MODEL_DIR / model_name, "rb") as f:
            # model = pickle.load(f)
            model = joblib.load(f)
        models = {target_name: model}
    else:
        model_pattern = f"{args.country}-*-{args.cv_type}-{args.universal_data_only}-{univ_data}-{ip_data}-{args.impute}-{args.standardise}-{args.target_transform}-{args.copy_to_nbrs}_res{args.resolution}.pkl"
        model_names = MODEL_DIR.glob(model_pattern)
        # print(model_pattern)
        models = {}
        for model_path in model_names:
            target_name = (
                str(model_path.stem)
                .replace(f"{args.country}-", "")
                .replace(
                    f"-{args.cv_type}-{args.universal_data_only}-{univ_data}-{ip_data}-{args.impute}-{args.standardise}-{args.target_transform}-{args.copy_to_nbrs}_res{args.resolution}",
                    "",
                )
            )
            with open(model_path, "rb") as f:
                # model = pickle.load(f)
                model = joblib.load(f)
            models[target_name] = model

    # load data
    if args.country != "all":
        data_path = Path(DATA_DIR) / f"hexes_{args.country}_res{args.resolution}_*.csv"
        data_path = next(
            Path(data_path.parent).expanduser().glob(data_path.name)
        )  # threshold on 'ground-truth' doesn't matter
        # so just take first found
        XY = pd.read_csv(data_path)
    else:
        print("Using all available data - currently not generalisable:")
        print("will look for all data in form")
        print("(expanded_/hexes_)[country]_res[args.resolution]_thres[threshold].csv,")
        print(f"in specified directory: {DATA_DIR}")
        clean_name = f"hexes_*_res{args.resolution}_*.csv"
        all_data = list(Path(DATA_DIR).expanduser().glob(clean_name))
        try:
            XY = pd.read_csv(all_data[0])
        except Exception:
            XY = pd.read_csv(all_data)
        if len(all_data) > 1:
            XY = pd.concat(
                [XY, *list(map(pd.read_csv, all_data[1:]))], ignore_index=True, axis=0
            )
        # arbitrarily remove duplicate hexes
        XY.drop_duplicates(subset=["hex_code"], inplace=True)

    survey_idx = XY.columns.tolist().index("survey")
    X = XY.iloc[:, :survey_idx]
    # add in lat longs as ftrs
    latlongs = X.hex_code.swifter.apply(lambda x: h3.h3_to_geo(x))
    X["lat"] = latlongs.str[0]
    X["long"] = latlongs.str[1]
    print("Data loaded")
    output = X[["hex_code", "population"]].copy()
    preds = {}
    for target_name, model in models.items():
        print(f"Predicting for {target_name}...")
        preds = model.predict(X)
        # with open(f"tmp_{target}_preds.npy", "wb") as f:
        #     np.save(f, preds)
        output = pd.concat([output, pd.DataFrame(preds, columns=[target_name])], axis=1)
    OUTPUT_DIR = DATA_DIR.parent / "predictions"
    OUTPUT_DIR.mkdir(exist_ok=True)
    output.to_csv(
        OUTPUT_DIR
        / f"preds_{args.country}_res{args.resolution}_expanded-{args.copy_to_nbrs}.csv",
        index=False,
    )
