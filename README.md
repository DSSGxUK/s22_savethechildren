Microestimates of Multidimensional Child Poverty
==============================

A Python package to produce high-resolution multi-dimensional child poverty estimates.

### Partners

This project is a collaboration between [Save The Children United Kingdom (STC)](https://www.savethechildren.org.uk/what-we-do), [United Nations International Children's Emergency Fund (UNICEF)](https://www.unicef.org/about-unicef) and [Data Science for Social Good Fellowship UK](https://warwick.ac.uk/research/data-science/warwick-data/dssgx/). 

### Challenge 

The World Bank defines [Global Poverty](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwiJy4yTgoP6AhV-QUEAHQy7C2cQFnoECAQQAw&url=https%3A%2F%2Fwww.compassion.com%2Fpoverty%2Fglobal-poverty-definition.htm%23%3A~%3Atext%3DGlobal%2520poverty%2520is%2520defined%2520as%2Cdefined%2520by%2520the%2520World%2520Bank.&usg=AOvVaw1IvLvvBCUkZxRtVl5eM7km) as the number of people worldwide who live on less than USD $1.90 a day.  However, depending solely on economic definitions is not ideal for assessing poverty amongst **children**, who may be deprived across different areas such as education or nutrition. Thus, UNICEF and Save The Children have proposed a new definition of child poverty based on deprivations in any of 6 factors: **water, sanitation, healthcare, housing, nutrition and education**.  

The aim of our project is to help these organisations predict multi-dimensional child poverty in developing nations at a granular level, through use of open source data. These predictions can then be used to create **geospatial visualisations in the form of poverty maps**, that will accurately support policy design and local resource planning.

### Solution

![image](https://drive.google.com/uc?export=view&id=1OAGcy5YSbTtj6C7w8Jq-jCp_9z1dhd_G)

Our team produced a high-resolution map of child poverty that shows the proportion of children facing deprivations in a given area, along with the corresponding prediction interval. 

![image](https://drive.google.com/uc?export=view&id=1EuPo2dVybmNHaUYWsjRmNTDGGlkSUyJI)

We also produced [`stc_unicef_cpi`](https://stc-unicef-cpi.readthedocs.io/en/latest/index.html), an open-source Python library for the Data Science for International Development community that estimates high-resolution, multi-dimensional child poverty using DHS (and soon MICS!) geocoded survey data as ground truth.


## About the Project

![image](https://drive.google.com/uc?export=view&id=1qoqWZ5xVvmLgkpFWacRILa2hmx4AIOxm)

#### Data

Our team has used public data from Google Earth Engine (GEE) on precipitation, elevation, land usage census, etc; Facebook's ads connectivity graph, open telecommunications tower networks, and [Facebook's Relative Wealth Index (RWI)](https://dataforgood.facebook.com/dfg/tools/relative-wealth-index). Surveys from the [Demographic and Health Surveys (DHS) dataset](https://dhsprogram.com/) serve as our ground truth. 

#### Methods 

<!---

![image](https://drive.google.com/uc?export=view&id=1H5-MTQ3E-Ave7JS0YsJLhsCJ7s58-DNG)

![image](https://drive.google.com/uc?export=view&id=1FpOTT0kBKKjcJa011Uk5TNNKyL1RkUne)

![image](https://drive.google.com/uc?export=view&id=11Y4PEZxyYCr0705pyAwNT0Z8qDu7DRSD)

--->

We make use of Uber's [H3 spatial indexing](https://www.uber.com/en-IN/blog/h3/) to tessellate over geographical locations. The hexagonal shape is chosen due to uniformity of neighbors and reduced sampling bias from edge effects, which is attributed to a high perimeter:area ratio. Our experiments are performed at [H3 resolution](https://h3geo.org/docs/core-library/restable/) 7 (each hexagon has area ~5.16km2)

The target dimensions are averaged at the hexagonal level. To account for geographic displacement introduced by DHS, data for each hexagon is copied to its neighbors (1 for urban areas and 2 for rural areas). Feature extraction from satelltite images is performed using a convolutional autoencoder:

![image](https://drive.google.com/uc?export=view&id=1v6vLy6C9g46K-xr71Z8BoHohBUWHZ0gw)

For the prediction task, we use machine learning, along with spatial cross validation, to train and tune our models. Uncertainity estimation is done through generation of 90% prediction intervals. Visualization of predictions is performed using [kepler.gl](https://github.com/keplergl/kepler.gl).

Further, these scripts have been bundled into a Python package, with reproducible scripts for parsing public census data and satellite images, that may be helpful for many other internal projects, such as the Climate Mobility and Children piece, or the Children on the Move research.

<!---
We used machine learning and deep learning techniques to achieve a very high resolution of 5km2, beyond most public research which is only aggregated at state levels, which will be helpful for targetted aid planning and policy research in both organisations.

Beyond this map, the team has also done excellent work in creating a python package and reproducible scripts for parsing public census data and satellite images, that may be helpful for many other internal projects, such as the Climate Mobility and Children piece, or the Children on the Move research.
--->

## Getting started

To use [`stc_unicef_cpi`](https://stc-unicef-cpi.readthedocs.io/en/latest/index.html), first clone the repo:

```
git clone git@github.com:DSSGxUK/s22_savethechildren.git
```

Then inside the top-level directory of the repo, run

``` 
pip install .
```

to allow imports from other python scripts.

For more information on getting data, training a model and making predictions, [click here](https://stc-unicef-cpi.readthedocs.io/en/latest/getting-started.html#getting-started).


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
|   |-- prediction_intervals.py <- Generate prediction intervals using the dataset and trained model/pipeline
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

## Contributors

- [John Fitzergerald](https://github.com/fitzgeraldja) 
- [Daniela Pinto Veizaga](https://github.com/dapivei)
- [Arpita Saggar](https://github.com/Arpita2512)
- [Marina Vicini](https://github.com/marinavicini)

In collaboration with Project Manager: [Valerie Lim](https://github.com/valerielim) and Technical Mentor: [Mihir Mehta](https://github.com/mihirpsu) 

