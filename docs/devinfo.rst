Developer information
=====================



.. _dev_getting_started:

Getting started
===============

To set up pre-commit to run, just ``pip install pre-commit``, navigate to top-level directory, then run ``pre-commit install`` to be good to go.

Github workflow to automatically register issues from ``# TODO:`` comments already set up: for extra functionality see below.

While ``black`` will autoformat scripts on commit, if you want this to run locally you should ``pip install black[jupyter]`` to allow formatting of exploratory notebooks also.

Install locally
---------------

To allow import of functions locally through e.g. ``stc_unicef_cpi.subpkg.module``, navigate to top-level directory then just run editable install (``pip install -e .``).

TODO Options
------------

This section is copied from the `original TODO repo README <https://github.com/alstr/todo-to-issue-action>`_.

Unless specified otherwise, options should be on their own line, below the initial TODO declaration.

``assignees:``
^^^^^^^^^^^^^^

Comma-separated list of usernames to assign to the issue.

``labels:``
^^^^^^^^^^^

Comma-separated list of labels to add to the issue. If any of the labels do not already exist, they will be created. The `todo` label is automatically added to issues to help the action efficiently retrieve them in the future.

``milestone:``
^^^^^^^^^^^^^^

Milestone ID to assign to the issue. Only a single milestone can be specified and this must already have been created.

Other Options
^^^^^^^^^^^^^

Reference
^^^^^^^^^

As per the `Google Style Guide <https://google.github.io/styleguide/cppguide.html#TODO_Comments>`_, you can provide a reference after the TODO label. This will be included in the issue title for searchability.

.. code-block:: python

    def hello_world():
        # TODO(alstr) Come up with a more imaginative greeting
        print("Hello world!")


Don't include parentheses within the identifier itself.

Project Organization
====================

.. code-block:: console

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.cfg           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src/stc_unicef_cpi                <- Source code for use in this project, see below for script details.
    │   ├── data           <- Scripts to download or generate data
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


Project based on the `cookiecutter data science project template <"https://drivendata.github.io/cookiecutter-data-science/">`_.


Script breakdown
----------------

Main package code is found in `src/stc_unicef_cpi`, with the structure

.. code-block:: console

    |-- data
    |   |-- ResnetWithPCA.py <- Use pretrained Resnet to extract features, use PCA to compress
    |   |-- cv_loaders.py <- Dataloaders for TIFF images, and cross validation utils (especially spatial)
    |   |-- get_cell_tower_data.py <- Obtain cell tower data
    |   |-- get_econ_data.py <- Obtain economic data (GDP, PPP, Elec. Consump.)
    |   |-- get_facebook_data.py <- Obtain FB data (deprecated due to time and feature utility, but for audience info using marketing API)
    |   |-- get_osm_data.py <- Obtain Open Street Maps data (specifically road density)
    |   |-- get_satellite_data.py <- Obtain satellite data (from GEE)
    |   |-- get_speedtest_data.py <- Obtain speedtest data (from Ookla)
    |   |-- make_dataset.py <- Combine into model-ready dataset
    |   |-- make_gee_dataset.py <- Obtain GEE data (deprecated, see `get_satellite_data.py`)
    |   |-- process_geotiff.py <- Utils for processing GeoTIFF files
    |   |-- process_netcdf.py <- Utils for NetCDF files
    |   |-- process_to_torch.py <- (deprecated) dataloaders for PyTorch
    |   `-- test_gee_data_download.js <- Example javascript file for code.earthengine.google.com
    |-- features
    |   |-- autoencoder_features.py <- Train an autoencoder on image dataset, use to extract features
    |   `-- build_features.py <- (deprecated) construct additional features
    |-- models
    |   |-- inflated_vals_2stg.py <- Two-stage models for inflated values (classification followed by regression)
    |   |-- lgbm_baseline.py <- LGBM baseline
    |   |-- mobnet_TL.py <- MobNet transfer learning (future work)
    |   |-- predict_model.py <- Make predictions
    |   `-- train_model.py <- Train overall model
    |-- utils
    |   |-- constants.py <- Constants
    |   |-- general.py <- General
    |   |-- geospatial.py <- Geospatial specific
    |   |-- mlflow_utils.py <- MLflow utils
    |   `-- scoring.py <- Scoring metrics
    `-- visualization
        `-- visualize.py <- Visualisation utils for model + predictions



.. _dev_main_comps:

Main components
---------------





.. _dev_contrib_guidelines:


Contribution guidelines
-----------------------
