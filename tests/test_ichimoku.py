import numpy as np
import pandas as pd
import pytest

from indicators import Ichimoku


@pytest.fixture
def sample_df():
    """100本のサンプルOHLCデータ"""
    rng = np.random.default_rng(123)
    n = 100
    close = 100.0 + np.cumsum(rng.normal(0, 0.3, n))
    high = close + rng.uniform(0.1, 0.5, n)
    low = close - rng.uniform(0.1, 0.5, n)
    dates = pd.bdate_range("2025-01-01", periods=n)
    return pd.DataFrame({"high": high, "low": low, "close": close}, index=dates)


class TestIchimokuCalculate:
    def test_output_columns(self, sample_df):
        result = Ichimoku().calculate(sample_df)
        expected = {"tenkan_sen", "kijun_sen", "senkou_span_a", "senkou_span_b", "chikou_span"}
        assert expected.issubset(result.columns)

    def test_tenkan_sen_period(self, sample_df):
        ichi = Ichimoku(tenkan_period=9)
        result = ichi.calculate(sample_df)
        assert result["tenkan_sen"].isna().sum() == 8  # 9期間 → 最初の8行がNaN

    def test_kijun_sen_period(self, sample_df):
        ichi = Ichimoku(kijun_period=26)
        result = ichi.calculate(sample_df)
        assert result["kijun_sen"].isna().sum() == 25

    def test_senkou_span_b_period(self, sample_df):
        ichi = Ichimoku(senkou_b_period=52, kijun_period=26)
        result = ichi.calculate(sample_df)
        # 52期間 rolling + 26期間 shift → 最初の77行がNaN
        assert result["senkou_span_b"].isna().sum() == 77

    def test_chikou_span_shift(self, sample_df):
        ichi = Ichimoku(chikou_shift=26)
        result = ichi.calculate(sample_df)
        # 末尾26行がNaN (未来へのシフト)
        assert result["chikou_span"].isna().sum() == 26

    def test_midpoint_formula(self, sample_df):
        result = Ichimoku(tenkan_period=9).calculate(sample_df)
        idx = 20
        h9 = sample_df["high"].iloc[idx - 8 : idx + 1].max()
        l9 = sample_df["low"].iloc[idx - 8 : idx + 1].min()
        expected = (h9 + l9) / 2
        assert abs(result["tenkan_sen"].iloc[idx] - expected) < 1e-10

    def test_original_data_unchanged(self, sample_df):
        original = sample_df.copy()
        Ichimoku().calculate(sample_df)
        pd.testing.assert_frame_equal(sample_df, original)


class TestIchimokuSignals:
    def test_signal_columns(self, sample_df):
        signals = Ichimoku().signals(sample_df)
        expected = {
            "tenkan_kijun_cross",
            "price_vs_cloud",
            "cloud_color",
            "chikou_vs_price",
            "composite_score",
        }
        assert expected.issubset(signals.columns)

    def test_cross_values(self, sample_df):
        signals = Ichimoku().signals(sample_df)
        assert set(signals["tenkan_kijun_cross"].unique()).issubset({-1, 0, 1})

    def test_composite_score_range(self, sample_df):
        signals = Ichimoku().signals(sample_df)
        assert signals["composite_score"].min() >= -4
        assert signals["composite_score"].max() <= 4

    def test_custom_parameters(self, sample_df):
        ichi = Ichimoku(tenkan_period=7, kijun_period=22, senkou_b_period=44, chikou_shift=22)
        result = ichi.calculate(sample_df)
        assert result["tenkan_sen"].isna().sum() == 6
        signals = ichi.signals(sample_df)
        assert len(signals) == len(sample_df)
