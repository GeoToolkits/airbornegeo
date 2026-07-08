import numpy as np
import pandas as pd
import pytest

from airbornegeo.trend import _resolve_fallback_degree, trend


@pytest.mark.parametrize(
    ("degree", "n_points", "expected"),
    [
        (3, 5, 3),
        (3, 2, 1),
        (3, 4, 3),
        (3, 0, None),
        (3, -1, None),
    ],
)
def test_resolve_fallback_degree(degree, n_points, expected):
    """The requested degree should be reduced to what n_points can support, or None if there are no points."""
    assert _resolve_fallback_degree(degree, n_points) == expected


def test_trend_linear_fit_reproduces_line():
    """A degree-1 fit to perfectly linear data should reproduce that line exactly at new points."""
    n = 20
    x = np.linspace(0, 10, n)
    y = 3 * x + 2
    fit_df = pd.DataFrame({"x": x, "y": y})
    predict_df = pd.DataFrame({"x": np.linspace(0, 10, 5)})
    result = trend(fit_df, ["x", "y"], predict_df, ["x", "pred"], degree=1)
    assert result.pred.to_numpy() == pytest.approx(3 * predict_df.x.to_numpy() + 2)


def test_trend_reduces_degree_when_insufficient_points():
    """2 points can only support a degree-1 fit, even though degree=5 is requested."""
    fit_df = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})
    result = trend(
        fit_df, ["x", "y"], pd.DataFrame({"x": [0.5]}), ["x", "pred"], degree=5
    )
    assert result.pred.tolist() == pytest.approx([0.5])


def test_trend_no_valid_points_raises_userwarning():
    """Fitting with no non-NaN points should raise UserWarning."""
    fit_df = pd.DataFrame({"x": [np.nan, np.nan], "y": [np.nan, np.nan]})
    with pytest.raises(UserWarning, match="No valid points available to fit a trend"):
        trend(fit_df, ["x", "y"], pd.DataFrame({"x": [0.5]}), ["x", "pred"], degree=1)


def test_trend_nan_rows_excluded_before_counting_points():
    """2 valid points plus 1 NaN row should behave as if only the 2 valid points existed."""
    fit_df = pd.DataFrame({"x": [0.0, 1.0, np.nan], "y": [0.0, 1.0, 5.0]})
    result = trend(
        fit_df, ["x", "y"], pd.DataFrame({"x": [0.5]}), ["x", "pred"], degree=5
    )
    assert result.pred.tolist() == pytest.approx([0.5])


def test_trend_uniform_weights_match_unweighted():
    """Sample weights that are all equal to 1 should give the same fit as no weighting."""
    n = 20
    x = np.linspace(0, 10, n)
    y = 3 * x + 2
    fit_df = pd.DataFrame({"x": x, "y": y, "w": np.ones(n)})
    predict_df = pd.DataFrame({"x": np.linspace(0, 10, 5)})
    weighted = trend(
        fit_df,
        ["x", "y"],
        predict_df,
        ["x", "pred"],
        degree=1,
        intersection_weight_col="w",
    )
    unweighted = trend(fit_df, ["x", "y"], predict_df, ["x", "pred"], degree=1)
    assert weighted.pred.to_numpy() == pytest.approx(unweighted.pred.to_numpy())


def test_trend_downweighting_outlier_pulls_fit_toward_true_trend():
    """Heavily downweighting an outlier point should pull the fit closer to the true trend."""
    n = 20
    x = np.linspace(0, 10, n)
    y = 3 * x + 2
    fit_df = pd.DataFrame({"x": x, "y": y, "w": np.ones(n)})
    fit_df.loc[0, "y"] = 1000.0
    predict_df = pd.DataFrame({"x": np.linspace(0, 10, 5)})

    unweighted = trend(fit_df, ["x", "y"], predict_df, ["x", "pred"], degree=1)
    fit_df.loc[0, "w"] = 0.001
    downweighted = trend(
        fit_df,
        ["x", "y"],
        predict_df,
        ["x", "pred"],
        degree=1,
        intersection_weight_col="w",
    )

    true_trend = 3 * predict_df.x.to_numpy() + 2
    unweighted_error = np.abs(unweighted.pred.to_numpy() - true_trend).sum()
    downweighted_error = np.abs(downweighted.pred.to_numpy() - true_trend).sum()
    assert downweighted_error < unweighted_error


def test_trend_does_not_mutate_input_dataframes():
    """trend() should not add or remove columns from the caller's fit/predict dataframes."""
    fit_df = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0]})
    predict_df = pd.DataFrame({"x": [0.5]})
    fit_cols_before = fit_df.columns.tolist()
    predict_cols_before = predict_df.columns.tolist()
    trend(fit_df, ["x", "y"], predict_df, ["x", "pred"], degree=1)
    assert fit_df.columns.tolist() == fit_cols_before
    assert predict_df.columns.tolist() == predict_cols_before


def test_trend_returns_copy_of_predict_df_with_new_column():
    """The result should be a copy of predict_df with only the x and predicted-value columns."""
    fit_df = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0]})
    predict_df = pd.DataFrame({"x": [0.5, 1.5]})
    result = trend(fit_df, ["x", "y"], predict_df, ["x", "pred"], degree=1)
    assert list(result.columns) == ["x", "pred"]
    assert len(result) == 2
