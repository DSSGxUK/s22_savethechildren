# TODO: Finish script to use pretrained model to extrapolate for unseen areas

import argparse
import pickle
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "High-res multi-dim CPI pretrained model predictions"
    )
    parser.add_argument(
        "--country",
        type=str,
        help="Choice of which country to predict for - options are 'all', 'nigeria' or 'senegal'",
        choices=["all", "nigeria", "senegal"],
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Pathway to pretrained model",
    )

    try:
        args = parser.parse_args()
    except argparse.ArgumentError:
        parser.print_help()
        sys.exit(0)

    with open(args.model, "rb") as f:
        model = pickle.load(f)
