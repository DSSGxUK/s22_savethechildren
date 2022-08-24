Getting started
===============

.. _installation:

Installation
------------

To use stc_unicef_cpi, first clone the repo:

.. code-block:: console

   (.venv) $ git clone git@github.com:DSSGxUK/s22_savethechildren.git

then inside the top-level directory of the repo, run

.. code-block:: console

   (.venv) $ pip install .

to allow imports from other python scripts.

We will publish to PyPI to allow direct installation shortly!


.. _data:

Getting data
----------------

To obtain the necessary data for a given country,
you can use the script ``make_dataset.py`` in ``./src/stc_unicef_cpi/data``.

This has the following command line arguments:

.. code-block:: console

    (.venv) $ python make_dataset.py --help
    usage: High-res multi-dim CPI dataset creation [-h] [-c COUNTRY] [-r RESOLUTION] [--force] [--force-download] [--add-auto]

    optional arguments:
    -h, --help            show this help message and exit
    -c COUNTRY, --country COUNTRY
                            Country to make dataset for, default is Nigeria
    -r RESOLUTION, --resolution RESOLUTION
                            H3 resolution level, default is 7
    --force               Force recreation of dataset, without redownloading unless necessary
    --force-download, -fdl
                            Force (re)download of dataset
    --add-auto            Generate autoencoder features also

The ``country`` argument uses a fuzzy search so don't worry about getting the name exactly right!
We recommend using resolution 6 or 7, as these appear to be a reasonable tradeoff between high-resolution
and small survey sample sizes. We would caution that adding autoencoder features can take a considerable time
to both train and make predictions for full countries (especially without a GPU!), often for minimal improvement,
but in some cases can help considerably.




.. _model:

Training a model
----------------

Once the dataset has been created successfully, training a model is as simple as
running the script ``train_model.py`` in ``./src/stc_unicef_cpi/model``.

This has the following command line arguments:

.. code-block:: console

    (.venv) $ python train_model.py --help
    usage: High-res multi-dim CPI model training [-h] [-d DATA] [--clean-name CLEAN_NAME] [--resolution RESOLUTION] [--threshold THRESHOLD] [--country COUNTRY] [--prefix PREFIX] [-ip {true,false}] [--universal-data-only {true,false}] [--copy-to-nbrs {true,false}] [--model {lgbm,automl,catboost}]
                                                 [--test-size TEST_SIZE] [--nfolds NFOLDS] [--cv-type {normal,stratified,spatial}] [--eval-split-type {normal,stratified,spatial}]
                                                 [--target {all,education,sanitation,housing,water,av-severity,av-prevalence,av-2-prevalence,health,nutrition,av-3-prevalence,av-4-prevalence}] [--target-transform {none,log,power}] [--ncores NCORES] [--impute {none,mean,median,knn,linear,rf}]
                                                 [--standardise {none,standard,minmax,robust}] [--automl-warm-start] [--plot] [--ftr-impt] [--log-run] [--save-model]

    optional arguments:
    -h, --help            show this help message and exit
    -d DATA, --data DATA  Pathway to data directory
    --clean-name CLEAN_NAME
                            Name of clean dataset inside data directory
    --resolution RESOLUTION, -res RESOLUTION
                            Resolution of h3 grid, defaults to 7
    --threshold THRESHOLD, -thres THRESHOLD
                            Threshold for minimum number of surveys per hex, defaults to 30
    --country COUNTRY     Choice of which country to use for training - options are 'all' in which case all currently available data is used, or the name of a specific country for which data is available
    --prefix PREFIX       Prefix to name the saved models / checkpoints
    -ip {true,false}, --interpretable {true,false}
                            Make model (more) interpretable - no matter other flags, use only base (non auto-encoder) features so can explain
    --universal-data-only {true,false}, -univ {true,false}
                            Use only universal data (i.e. no country-specific data) - only applicable if --country!=all
    --copy-to-nbrs {true,false}, -cp2nbr {true,false}
                            Use expanded dataset, where 'ground-truth' values are copied to neighbouring cells
    --model {lgbm,automl,catboost}
                            Choice of model to train (and tune)
    --test-size TEST_SIZE
                            Proportion of data to exclude for test evaluation, default is 0.2
    --nfolds NFOLDS       Number of folds of training set for cross validation, default is 5
    --cv-type {normal,stratified,spatial}
                            Type of CV to use, default is normal, choices are normal (fully random), stratified and spatial
    --eval-split-type {normal,stratified,spatial}
                            Method to split test from training set, default is normal, choices are normal (fully random), stratified and spatial
    --target {all,education,sanitation,housing,water,av-severity,av-prevalence,av-2-prevalence,health,nutrition,av-3-prevalence,av-4-prevalence}
                            Target variable to use for training, default is all, choices are 'all' (train separate model for each of the following), 'av-severity' (average number of deprivations / child), 'av-prevalence' (average proportion of children with at least one deprivation), 'av-2-prevalence' (average
                            proportion of children with at least two deprivations), proportion of children deprived in 'education', 'sanitation', 'housing', 'water'. May also pass 'health' or 'nutrition' but limited ground truth data increases model variance. Similarly may pass 'av-3-prevalence' or
                            'av-4-prevalence', but ~50pc of cell data is exactly zero for 3, and ~80pc for 4, so again causes modelling issues.
    --target-transform {none,log,power}
                            Transform target variable(s) prior to fitting model - choices of none (default, leave raw), 'log', 'power' (Yeo-Johnson)
    --ncores NCORES       Number of cores to use, defaults to 4
    --impute {none,mean,median,knn,linear,rf}
                            Impute missing values prior to training, or leave as nan (default option)
    --standardise {none,standard,minmax,robust}
                            Standardise feature data prior to fitting model, options are none (default, leave raw), standard (z-score), minmax (min-max normalisation to limit to 0-1 range), or robust (median and quantile version of z-score)
    --automl-warm-start   When possible, use best model configuration found from previous runs to initialise hyperparameter search for each model.
    --plot                Produce scatter plot(s) of predicted vs actual values on test set
    --ftr-impt            Investigate final model feature importance using BorutaShap
    --log-run             Use MLflow to log training run params + scores, by default in a /models/mlruns directory where /models is contained in same parent folder as args.data
    --save-model          Save trained models (joblib pickled), by default in a /models directory contained in same parent folder as args.data

- If no argument is passed to ``--data``, by default the script will look in ``./data/processed``,
  where the output of ``make_dataset.py`` will save by default.
- If no argument is passed to ``--clean-name``, by default the script will look for dataset files in this location
  in the form ``(expanded_/hexes_)[country]_res[args.resolution]_thres[arg.threshold].csv``, which again is the form
  in which ``make_dataset.py`` outputs by default.
- As in ``make_dataset.py``, default resolution is 7, and threshold is 30, then default country is 'all' (i.e. use all available data).
- ``--interpretable``, ``--universal-data-only`` and ``copy-to-nbrs`` all have 'true' or 'false' as options, default being 'false'.
  Details are as in the help, and from initial experiments it would seem that naively expanding data does not generally improve
  model performance, though it can for some cases.
- While LGBM and Catboost are listed as options for the model, these are not currently implemented suitably for all other arguments.
  LGBM is included in the set of models for the automl option anyway, and catboost would be were it not for conflicts in other packages.
  The default dataset only has a single categorical parameter, so catboost did not seem to outperform other alternatives hence this is
  not a priority. AutoML here refers to FLAML from Microsoft - a package for cost-efficient automatic hyperparameter tuning.
- The method for splitting both the test set from the overall dataset (``--eval-split-type``), and for splitting validation sets from the
  train set (``--cv-type``) can be chosen separately to each other, from the options 'normal' (fully random),
  'stratified' (using target values) and 'spatial' (using location information). This is important depending on how you want to evaluate
  the model -- in particular effectively as interpolation in areas (i.e. countries) where you have data ('normal' eval split best), or
  generalisation to completely new areas, for which 'spatial' eval split is likely better. For splitting the training set on the other hand
  it's more important to just look at performance. We find that 'spatial' often seems to provide the best overall models, likely as it
  finds more robust choices of hyperparameter.
- Due to minimal ground truth data, setting ``--target`` to 'all' (the default) will only actually train models for a subset of the
  indices - in particular neglecting 'health' and 'nutrition'. These neglected indices may still have models trained for them by
  specific request. The metric for cross validation is chosen to be mean squared error, but in final evaluation MSE, MAE and R:sup:`2`
  are all reported.
- If ``--impute`` is left as 'none' (the default), then currently errors may be thrown for some model choices. This is to be resolved.
- The argument ``--plot`` will also by default save figures in a ``./data/figures`` directory, and as an artifact for MLflow if
  ``--log-run`` is also passed.
- Other options are straightforward as described in the help text.
- All options can be tested for country choice of 'all', 'nigeria' and 'senegal' by running ``bash model_training.sh``.





.. _predict:

Making predictions
----------------
