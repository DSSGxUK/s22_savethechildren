stc_unicef_cpi
==============================

A package to produce a high-resolution multi-dimensional child poverty index visualisation.

### Partners

This project is a collaboration between [Save The Children United Kingdom (STC)](https://www.savethechildren.org.uk/what-we-do), [United Nations International Children's Emergency Fund (UNICEF)](https://www.unicef.org/about-unicef) and [Data Science for Social Good Fellowship UK](https://warwick.ac.uk/research/data-science/warwick-data/dssgx/). 

### Challenge 

The World Bank defines [Global Poverty](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwiJy4yTgoP6AhV-QUEAHQy7C2cQFnoECAQQAw&url=https%3A%2F%2Fwww.compassion.com%2Fpoverty%2Fglobal-poverty-definition.htm%23%3A~%3Atext%3DGlobal%2520poverty%2520is%2520defined%2520as%2Cdefined%2520by%2520the%2520World%2520Bank.&usg=AOvVaw1IvLvvBCUkZxRtVl5eM7km) as the number of people worldwide who live on less than USD $1.90 a day. However, this definition is not ideal  for assessing poverty amongst children, who have different needs in areas such as education and nutrition, and who are not able to earn a living in the same way adults do.

So, UNICEF and Save The Children have proposed a new definition of child poverty in 6 equally-weighted factors: **access to water, sanitation, healthcare, housing, nutrition and education**. Our goal is to help these organisations predict and assess child poverty in developing nations, using open source data formats, to produce a **map visualisation** that will accurately support policy design and local resource planning. 

### Solution

![image](https://drive.google.com/uc?export=view&id=1OAGcy5YSbTtj6C7w8Jq-jCp_9z1dhd_G)

Our team produced a high-resolution map of child poverty using several countries in Africa as proof of concept. 

We also produced [`stc_unicef_cpi`](https://stc-unicef-cpi.readthedocs.io/en/latest/index.html), an open-source Python library for the Data Science for International Development community that estimates high-resolution, multi-dimensional child poverty using DHS (and soon MICS!) geocoded survey data.


## About the Project

![image](https://drive.google.com/uc?export=view&id=1qoqWZ5xVvmLgkpFWacRILa2hmx4AIOxm)

#### Data

Our team has used public data from Google Earth Engine (GEE) on precipitation, elevation, land usage census, Facebook's ads connectivity graph, open telecommunications tower networks, and Facebook's Relative Wealth Index (RWI). We validated this against annual surveys from the [Demographic and Health Surveys (DHS) dataset](https://dhsprogram.com/) as our ground truth. 

#### Methods 

![image](https://drive.google.com/uc?export=view&id=1H5-MTQ3E-Ave7JS0YsJLhsCJ7s58-DNG)

![image](https://drive.google.com/uc?export=view&id=1nt6c8zEsPbJ0iZDHh6kJZfCw9itvIGib)

We used machine learning and deep learning techniques to achieve a very high resolution of 5km2, beyond most public research which is only aggregated at state levels, which will be helpful for targetted aid planning and policy research in both organisations.

![image](https://drive.google.com/uc?export=view&id=11Y4PEZxyYCr0705pyAwNT0Z8qDu7DRSD)

#### Architecture 

Beyond this map, the team has also done excellent work in creating a python package and reproducible scripts for parsing public census data and satellite images, that may be helpful for many other internal projects, such as the Climate Mobility and Children piece, or the Children on the Move research.

## Getting started

To set up pre-commit to run, just `pip install pre-commit`, navigate to top-level directory, then run `pre-commit install` to be good to go.

Github workflow to automatically register issues from `# TODO:` comments already set up: for extra functionality see below.

While `black` will autoformat scripts on commit, if you want this to run locally you should `pip install black[jupyter]` to allow formatting of exploratory notebooks also.

### Install locally

To allow import of functions locally through e.g. `stc_unicef_cpi.subpkg.module`, navigate to top-level directory then just run editable install (`pip install -e .`).


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


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


## Script breakdown
Main package code is found in `src/stc_unicef_cpi`, with the structure

```
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
```
