from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

if TYPE_CHECKING:
    import pandas as pd

from airbornegeo import logger


def _resolve_fallback_degree(degree: int, n_points: int) -> int | None:
    """
    Determine the highest polynomial degree that can be fit without creating an
    underdetermined system.

    A polynomial of degree ``d`` has ``d + 1`` coefficients (including the
    intercept), so at least ``d + 1`` valid observations are required. If fewer
    observations are available, the requested degree is reduced to the highest
    supported value.

    Parameters
    ----------
    degree : int
        Requested polynomial degree.
    n_points : int
        Number of valid observations available for fitting.

    Returns
    -------
    int or None
        Degree that can be safely fit. Returns ``None`` if no valid observations
        are available.
    """
    if n_points <= 0:
        return None

    return min(degree, n_points - 1)


def trend(
    data_to_fit: pd.DataFrame,
    cols_to_fit: list[str],
    data_to_predict: pd.DataFrame,
    cols_to_predict: list[str],
    degree: int,
    intersection_weight_col: str | None = None,
) -> pd.DataFrame:
    """
    Fit a polynomial trend using scikit-learn and predict values at new locations.

    The polynomial degree is automatically reduced if there are insufficient
    observations to constrain the requested model.

    Parameters
    ----------
    data_to_fit : pandas.DataFrame
        Data used to fit the polynomial.
    cols_to_fit : list[str]
        Two column names specifying the independent and dependent variables,
        respectively.
    data_to_predict : pandas.DataFrame
        DataFrame containing the coordinates where the fitted trend should be
        evaluated.
    cols_to_predict : list[str]
        Two column names specifying the predictor variable and the output column
        that will receive the predicted values.
    degree : int
        Requested polynomial degree.
    intersection_weight_col : str or None, optional
        Column containing sample weights for weighted least-squares fitting.

    Returns
    -------
    pandas.DataFrame
        Copy of ``data_to_predict`` containing the predicted trend.
    """
    fit_df = data_to_fit.copy()
    predict_df = data_to_predict.copy()

    # drop rows with NaN in the fit columns before counting available points
    fit_df = fit_df.dropna(subset=cols_to_fit)
    n_points = len(fit_df)

    resolved_degree = _resolve_fallback_degree(degree, n_points)

    if resolved_degree is None:
        msg = "No valid points available to fit a trend"
        raise UserWarning(msg)

    if resolved_degree != degree:
        logger.debug(
            "Reducing polynomial degree from %d to %d (%d valid point(s)).",
            degree,
            resolved_degree,
            n_points,
        )

    pipeline = Pipeline(
        [
            (
                "polynomial_features",
                PolynomialFeatures(
                    degree=resolved_degree,
                    include_bias=True,
                ),
            ),
            (
                "linear_regression",
                LinearRegression(),
            ),
        ]
    )

    sample_weight = (
        fit_df[intersection_weight_col].to_numpy()
        if intersection_weight_col is not None
        else None
    )

    pipeline.fit(
        fit_df[cols_to_fit[0]].to_numpy()[:, np.newaxis],
        fit_df[cols_to_fit[1]].to_numpy(),
        linear_regression__sample_weight=sample_weight,
    )

    predict_df[cols_to_predict[1]] = pipeline.predict(
        predict_df[cols_to_predict[0]].to_numpy()[:, np.newaxis]
    )

    return predict_df
