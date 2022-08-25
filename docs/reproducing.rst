Reproducing results
===================

.. _reproducing data:

Reproducing data
----------------

After obtaining necessary permissions for data access (particularly GEE, after which you will need to run ``earthengine authenticate``
on the machine you are using to run the package), to produce all data for countries considered, first navigate to the ``data/``
subpackage then run the script ``bash make_countries.sh``.

The initial run will fail, as currently you will need to manually go to the Google Drive of the authenticated account, and
manually place all data in a ``external/gee`` directory within the top-level ``data/`` directory. This will be fixed in the future.
Within your drive, the GEE data for each country should be placed in a ``gee/{country}`` directory for better organisation, but currently
on download the data for all countries should be placed in a single directory - this may change in future versions.
It may take some time for the GEE data to appear in your Drive - you can check the status of
tasks `here <https://code.earthengine.google.com/tasks>`_.

Once all data is available locally, run ``bash make_countries.sh`` one more time, and on completion you should have all data necessary!


.. _reproducing models:

Reproducing models
------------------

Once the datasets have been created, to train models with the same pipeline as we used, for the same countries,
navigate to the ``models/`` subpackage then simply run ``bash train_countries.sh``.




.. _reproducing predictions:

Reproducing predictions
-----------------------

Finally to use these models to make all the predictions, then just run ``bash predict_countries.sh``. By default these predictions
will be saved in the ``predictions/`` directory within the top-level ``data/`` directory of the repo.


Additional steps
-----------------

Prediction intervals are not automatically generated as part of the pipeline. They will be incorporated soon, but in the meantime you
may follow steps as in ``./notebooks/prediction_intervals.ipynb`` if desired.
