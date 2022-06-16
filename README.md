stc_unicef_cpi
==============================

A package to produce a high-resolution multi-dimensional child poverty index.

To set up pre-commit to run, just `pip install pre-commit`, navigate to top-level directory, then run `pre-commit install` to be good to go.

Github workflow to automatically register issues from `# TODO:` comments already set up: for extra functionality see below.

While `black` will autoformat scripts on commit, if you want this to run locally you should `pip install black[jupyter]` to allow formatting of exploratory notebooks also.

## TODO Options

This section is copied from the [original TODO repo README](https://github.com/alstr/todo-to-issue-action).

Unless specified otherwise, options should be on their own line, below the initial TODO declaration.

### `assignees:`

Comma-separated list of usernames to assign to the issue.

### `labels:`

Comma-separated list of labels to add to the issue. If any of the labels do not already exist, they will be created. The `todo` label is automatically added to issues to help the action efficiently retrieve them in the future.

### `milestone:`

Milestone ID to assign to the issue. Only a single milestone can be specified and this must already have been created.

### Other Options

#### Reference

As per the [Google Style Guide](https://google.github.io/styleguide/cppguide.html#TODO_Comments), you can provide a reference after the TODO label. This will be included in the issue title for searchability.

```python
def hello_world():
    # TODO(alstr) Come up with a more imaginative greeting
    print("Hello world!")
```

Don't include parentheses within the identifier itself.

# Project Organization
------------

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
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
