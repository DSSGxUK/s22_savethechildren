import featuretools as ft
import lightgbm as lgb
import numpy as np
import pandas as pd
import woodwork as ww
from BorutaShap import BorutaShap


def add_group_features(*dfs, join_on=""):
    """TODO: fix for our data

    From tuning using optuna in notebook, suggests that adding these features is indeed useful
    - seem to get slightly better CV performance using the full augmented dataset, along with
    similar performance for subset of features selected using BorutaShap (below)
    However, when assessing generalisation performance on completely new data, the cross-validated
    tuned model seems to significantly overfit on the base data, while performing much better on
    both fully augmented and subset of augmented data. Best generalised performance seems to be
    on subselection of features, as one might expect.

    :param dfs: any number of pandas dataframes to join for group features
    :type: dfs: pd.DataFrame
    :param join_on: column for entity sets to join on, defaults to ""
    :type join_on: str, optional
    :return: _description_
    :rtype: _type_
    """
    # responses = (
    #     X.melt(var_name="facet", ignore_index=False)
    #     .reset_index()
    #     .rename(columns={"index": "u_idx"})
    # )

    # dataframes = {
    #     "questions": (questions[["facet/big5"]], "facet/big5"),
    #     "responses": (responses, "index"),
    # }
    surveys, extras = dfs
    es = ft.EntitySet(id="cpi_es")

    es.add_dataframe(
        dataframe_name="surveys",
        dataframe=surveys,
        index=join_on,
        logical_types={
            col: ww.logical_types.Categorical
            for col in surveys.columns
            if hasattr(col, "cat")
        },
    )
    es.add_dataframe(
        dataframe_name="extras",
        dataframe=extras,
        index=join_on,
        logical_types={
            col: ww.logical_types.Categorical
            for col in extras.columns
            if hasattr(col, "cat")
        },
    )
    es.normalize_dataframe(
        base_dataframe_name="surveys",
        new_dataframe_name="hexes",
        index="hex_code",
        # copy_columns=["facet", "value"],
    )
    # relationships = [("questions", "facet/big5", "responses", "facet")]
    # es.add_relationship(*relationships[0])
    # es.add_interesting_values(
    #     dataframe_name="questions",
    #     values={"fac_domain": questions.fac_domain.unique().tolist()},
    # )
    # es.add_interesting_values(
    #     dataframe_name="users", values={"facet": es["users"].facet.unique().tolist()}
    # )
    feature_matrix, feature_defs = ft.dfs(
        entityset=es,
        target_dataframe_name="hexes",
        #   agg_primitives=["count"],
        #   where_primitives=["count"],
        #   trans_primitives=[]
        max_depth=2,
        where_primitives=["mean", "std", "min", "max", "skew"],
    )
    full_eng_ftr_data = pd.concat(
        [
            extras,
            feature_matrix[
                [
                    col
                    for col in feature_matrix.columns
                    if "WHERE" in col and "domain" in col
                ]
            ],
        ],
        axis=1,
    ).dropna(axis=1)
    return full_eng_ftr_data


#####################################
###### (iii) Feature selection ######
#####################################


def boruta_shap_ftr_select(
    X,
    y,
    base_model=lgb.LGBMRegressor(),
    plot=True,
    n_trials=100,
    sample=False,
    train_or_test="test",
    normalize=True,
    verbose=True,
    incl_tentative=True,
):
    """Simple wrapper to BorutaShap feature selection to also show feature plot
    (more interesting at this point)

    :param X: _description_
    :type X: _type_
    :param y: _description_
    :type y: _type_
    :param base_model: _description_, defaults to lgb.LGBMRegressor()
    :type base_model: _type_, optional
    :param plot: show feature importance plot, defaults to True
    :type plot: bool, optional
    :param n_trials: _description_, defaults to 100
    :type n_trials: int, optional
    :param sample: if true then a row-wise sample of the data will be used to calculate the feature importance values, defaults to False
    :type sample: bool, optional
    :param train_or_test: Decides whether the feature importance should be calculated on out of sample data - see the dicussion here https://compstat-lmu.github.io/iml_methods_limitations/pfi-data.html#introduction-to-test-vs.training-data, defaults to "test"
    :type train_or_test: str, optional
    :param normalize: if true the importance values will be normalized using the z-score formula, defaults to True
    :type normalize: bool, optional
    :param verbose: a flag indicator to print out all the rejected or accepted features, defaults to True
    :type verbose: bool, optional
    :param incl_tentative: _description_, defaults to True
    :type incl_tentative: bool, optional
    :return: _description_
    :rtype: _type_
    """

    ftr_selector = BorutaShap(
        model=base_model, importance_measure="shap", classification=False
    )
    ftr_selector.fit(
        X=X,
        y=y,
        n_trials=n_trials,
        sample=sample,
        train_or_test=train_or_test,
        normalize=normalize,
        verbose=verbose,
    )
    if plot:
        ftr_selector.plot(which_features="all")
    # return subset of original data w selected ftrs
    if incl_tentative:
        subset = ftr_selector.starting_X[ftr_selector.accepted + ftr_selector.tentative]
    else:
        subset = ftr_selector.Subset()
    return subset
