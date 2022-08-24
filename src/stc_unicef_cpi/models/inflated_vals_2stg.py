from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from flaml import AutoML
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
    clone,
    is_classifier,
    is_regressor,
)
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class InflatedValsRegressor(BaseEstimator, RegressorMixin):
    """A meta regressor for datasets with inflated values, i.e. the
    targets contain certain values with much higher frequency than others.

    `InflatedValsRegressor` consists of a classifier and a regressor.

        - The classifier's task is to find of if the target is an inflated value or not.
        - The regressor's task is to output a prediction whenever the classifier indicates that the there should be a non-zero prediction.

    The regressor is only trained on examples where the target is not an inflated value,
    which makes it easier for it to focus.

    At prediction time, the classifier is first asked if the output should be one of the
    inflated values. Depending on the mode selected, either

        (i) output that value, or
        (ii) use the estimated class probabilities to weight the output.

    If not predicted to be an inflated value, in case (i) ask the regressor
    for its prediction and output it.


    Examples

    .. code-block:: python

        import numpy as np
        from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

        np.random.seed(0)
        X = np.random.randn(10000, 4)
        y = ((X[:, 0] > 0) & (X[:, 1] > 0)) * np.abs(X[:, 2] * X[:, 3] ** 2)
        z = InflatedValsRegressor(
            classifier=ExtraTreesClassifier(random_state=0),
            regressor=ExtraTreesRegressor(random_state=0),
        )
        z.fit(X, y)
        # InflatedValsRegressor(classifier=ExtraTreesClassifier(random_state=0),
        #                     regressor=ExtraTreesRegressor(random_state=0))
        z.predict(X)[:5]
        # array([4.91483294, 0.        , 0.        , 0.04941909, 0.        ])

    """

    def __init__(self, classifier: ClassifierMixin, regressor: RegressorMixin) -> None:
        """Initialise

        :param classifier: A classifier that answers the question "Should the output be an inflated value?".
        For the second mode (weighted output), the classifier must have a `predict_proba` method.
        :type classifier: ClassifierMixin
        :param regressor: A regressor for predicting the target, particularly if not an inflated value.
        In the strict mode, its prediction is only used if `classifier` says that
        the output is not an inflated value.
        :type regressor: RegressorMixin
        """
        self.classifier = classifier
        self.regressor = regressor

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        inflated_vals: list[float] | np.ndarray = [0],
        sample_weight: np.ndarray | None = None,
        allow_nan: bool = True,
        cls_fit_kwargs: dict | None = None,
        reg_fit_kwargs: dict | None = None,
    ) -> InflatedValsRegressor:
        """
        Fit the model.

        :param X: The training data in shape (n_samples, n_features).
        :type X: Union[np.ndarray, pd.DataFrame]
        :param y: The target values, 1-dimensional.
        :type y: Union[np.ndarray, pd.Series]
        :param inflated_vals: Inflated values, defaults to [0]
        :type inflated_vals: Union[List[float], np.ndarray], optional
        :param sample_weight: Individual weights for each sample, defaults to None
        :type sample_weight: Optional[np.ndarray], optional
        :raises ValueError: If `classifier` is not a classifier or `regressor` is not a regressor.
        :return: Fitted regressor.
        :rtype: InflatedValsRegressor
        """
        X, y = check_X_y(X, y, force_all_finite="allow-nan" if allow_nan else True)
        inflated_vals = sorted(inflated_vals)
        self.infl_vals = np.array(inflated_vals)
        try:
            assert len(np.unique(inflated_vals)) == len(inflated_vals)
        except AssertionError:
            raise ValueError("inflated_vals must be unique")

        y_cls = self.get_cls_labels(y, inflated_vals)
        self._check_n_features(X, reset=True)
        if not is_classifier(self.classifier):
            if type(self.classifier) != type(AutoML()):
                raise ValueError(
                    f"`classifier` has to be a classifier. Received instance of {type(self.classifier)} instead."
                )
        if not is_regressor(self.regressor):
            if type(self.regressor) != type(AutoML()):
                raise ValueError(
                    f"`regressor` has to be a regressor. Received instance of {type(self.regressor)} instead."
                )

        try:
            check_is_fitted(self.classifier)
            self.classifier_ = self.classifier
        except NotFittedError:
            self.classifier_ = clone(self.classifier)
            if type(self.classifier) != type(AutoML()):
                self.classifier_.fit(
                    X, y_cls, sample_weight=sample_weight, **cls_fit_kwargs
                )
            else:
                self.classifier_.fit(X_train=X, y_train=y_cls, **cls_fit_kwargs)

        non_inflated_indices = np.where(
            # ~np.isin(self.classifier_.predict(X), self.infl_cls)
            ~np.isin(y_cls, self.infl_cls)
        )[0]

        if non_inflated_indices.size > 0:
            try:
                check_is_fitted(self.regressor)
                self.regressor_ = self.regressor
            except NotFittedError:
                self.regressor_ = clone(self.regressor)
                if type(self.regressor) != type(AutoML()):
                    self.regressor_.fit(
                        X[non_inflated_indices],
                        y[non_inflated_indices],
                        sample_weight=sample_weight[non_inflated_indices]
                        if sample_weight is not None
                        else None,
                        **reg_fit_kwargs,
                    )
                else:
                    self.regressor_.fit(
                        X_train=X[non_inflated_indices],
                        y_train=y[non_inflated_indices],
                        **reg_fit_kwargs,
                    )
        else:
            raise ValueError(
                "The predicted training labels are all zero, making the regressor obsolete. Change the classifier or use a plain regressor instead."
            )

        return self

    def predict(
        self,
        X: np.ndarray | pd.DataFrame,
        weighted: bool = False,
        allow_nan: bool = True,
    ) -> np.ndarray:
        """Make predictions.

        :param X: Samples to get predictions of, shape (n_samples, n_features).
        :type X: Union[np.ndarray, pd.DataFrame]
        :param weighted: Weight output, or use strict class predictions, defaults to False
        :type weighted: bool, optional
        :return: The predicted values.
        :rtype: np.ndarray, shape (n_samples,)
        """
        check_is_fitted(self)
        X = check_array(X, force_all_finite="allow-nan" if allow_nan else True)
        self._check_n_features(X, reset=False)

        if not weighted:
            output = np.zeros(len(X))
            cls_preds = self.classifier_.predict(X)
            infl_idxs = np.isin(cls_preds, self.infl_cls)
            non_inflated_indices = np.where(
                ~infl_idxs
                # ~np.isin(y_cls, infl_cls)
            )[0]
            if self.infl_cls[0] == 0:
                output[infl_idxs] = self.infl_vals[
                    (cls_preds[infl_idxs] // 2).astype(int)
                ]
            else:
                output[infl_idxs] = self.infl_vals[
                    ((cls_preds[infl_idxs] - 1) // 2).astype(int)
                ]

            if non_inflated_indices.size > 0:
                output[non_inflated_indices] = self.regressor_.predict(
                    X[non_inflated_indices]
                )
        else:
            cls_probs = self.classifier_.predict_proba(X)
            reg_output = self.regressor_.predict(X)
            reg_cls = self.get_cls_labels(reg_output, self.infl_vals, init=False)
            output_norm = (
                cls_probs[:, self.infl_cls].sum(axis=1)
                + cls_probs[np.arange(cls_probs.shape[0]), reg_cls.astype(int)]
            )
            output = (
                (cls_probs[:, self.infl_cls] * self.infl_vals).sum(
                    axis=1
                )  # inflated vals contrib
                + cls_probs[np.arange(cls_probs.shape[0]), reg_cls.astype(int)]
                * reg_output  # regressor contrib
            ) / output_norm  # normalise

        return output

    def get_cls_labels(
        self,
        y: np.ndarray | pd.Series,
        inflated_vals: list[float] | np.ndarray,
        init=True,
    ) -> pd.Series:
        """Get class labels of targets, y, according to inflated values passed

        :param y: Target values
        :type y: Union[np.ndarray, pd.Series]
        :param inflated_vals: Inflated values
        :type inflated_vals: Union[List[float], np.ndarray]
        :param init: Initialise, defaults to True
        :type init: bool, optional
        :return: Class labels
        :rtype: pd.Series
        """
        y_cls = np.zeros_like(y)
        # in general, n unique values partition \mathbb{R} into n+1 parts
        # so overall would have 2n + 1 (ordinal) classes - each of these
        # intervals + each of the inflated values themselves
        n_cls = 2 * len(inflated_vals) + 1
        # but of course might have boundaries as inflated values, so must handle this
        n_cls -= (inflated_vals[0] == y.min()) * 1 + (inflated_vals[-1] == y.max()) * 1
        # assign class labels
        if init:
            self.n_cls = n_cls
            if len(inflated_vals) > 1:
                for i, (iv1, iv2) in enumerate(
                    zip(inflated_vals[:-1], inflated_vals[1:])
                ):
                    if inflated_vals[0] == y.min():
                        y_cls[y == iv1] = 2 * i
                        y_cls[np.logical_and(y > iv1, y < iv2)] = 2 * i + 1
                        # cls labels corresponding to inflated vals
                        self.infl_cls = np.arange(0, n_cls, 2)
                    else:
                        if i == 0:
                            y_cls[y < inflated_vals[0]] = 0
                        y_cls[y == iv1] = 2 * i + 1
                        y_cls[np.logical_and(y > iv1, y < iv2)] = 2 * (i + 1)
                        self.infl_cls = np.arange(1, n_cls, 2)
                if inflated_vals[-1] != y.max():
                    y_cls[y > inflated_vals[-1]] = n_cls - 1
                else:
                    y_cls[y == inflated_vals[-1]] = n_cls - 1

            else:
                if inflated_vals[0] == y.min():
                    y_cls[y == inflated_vals[0]] = 0
                    y_cls[y != inflated_vals[0]] = 1
                    self.infl_cls = np.array([0])
                elif inflated_vals[0] == y.max():
                    y_cls[y != inflated_vals[0]] = 0
                    y_cls[y == inflated_vals[0]] = 1
                    self.infl_cls = np.array([1])
                else:
                    y_cls[y < inflated_vals[0]] = 0
                    y_cls[y == inflated_vals[0]] = 1
                    y_cls[y > inflated_vals[0]] = 2
                    self.infl_cls = np.array([1])
        else:
            if self.infl_cls[0] == 0:
                if len(inflated_vals) > 1:
                    for i, (iv1, iv2) in enumerate(
                        zip(inflated_vals[:-1], inflated_vals[1:])
                    ):
                        y_cls[y == iv1] = 2 * i
                        y_cls[np.logical_and(y > iv1, y < iv2)] = 2 * i + 1
                    if inflated_vals[-1] != y.max():
                        y_cls[y > inflated_vals[-1]] = self.infl_cls[-1] + 1
                    else:
                        y_cls[y == inflated_vals[-1]] = self.infl_cls[-1]
                else:
                    y_cls[y == inflated_vals[0]] = 0
                    y_cls[y != inflated_vals[0]] = 1
            else:
                if len(inflated_vals) > 1:
                    y_cls[y < inflated_vals[0]] = 0
                    for i, (iv1, iv2) in enumerate(
                        zip(inflated_vals[:-1], inflated_vals[1:])
                    ):
                        y_cls[y == iv1] = 2 * i + 1
                        y_cls[np.logical_and(y > iv1, y < iv2)] = 2 * (i + 1)
                    if inflated_vals[-1] != y.max():
                        y_cls[y > inflated_vals[-1]] = self.infl_cls[-1] + 1
                    else:
                        y_cls[y == inflated_vals[-1]] = self.infl_cls[-1]
                else:
                    y_cls[y < inflated_vals[0]] = 0
                    y_cls[y == inflated_vals[0]] = 1
                    y_cls[y > inflated_vals[0]] = 2
        return pd.Series(y_cls).astype("category")
