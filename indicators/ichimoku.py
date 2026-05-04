import pandas as pd
import numpy as np


class Ichimoku:
    """一目均衡表 (Ichimoku Kinko Hyo) インジケーター"""

    def __init__(
        self,
        tenkan_period: int = 9,
        kijun_period: int = 26,
        senkou_b_period: int = 52,
        chikou_shift: int = 26,
    ):
        self.tenkan_period = tenkan_period
        self.kijun_period = kijun_period
        self.senkou_b_period = senkou_b_period
        self.chikou_shift = chikou_shift

    @staticmethod
    def _midpoint(high: pd.Series, low: pd.Series, period: int) -> pd.Series:
        return (high.rolling(period).max() + low.rolling(period).min()) / 2

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        一目均衡表の全線を計算する。

        Parameters
        ----------
        df : DataFrame
            'high', 'low', 'close' カラムを含むOHLCデータ。

        Returns
        -------
        DataFrame
            元データに以下のカラムを追加したもの:
            - tenkan_sen    : 転換線
            - kijun_sen     : 基準線
            - senkou_span_a : 先行スパンA
            - senkou_span_b : 先行スパンB
            - chikou_span   : 遅行スパン
        """
        result = df.copy()
        h, l, c = result["high"], result["low"], result["close"]

        result["tenkan_sen"] = self._midpoint(h, l, self.tenkan_period)
        result["kijun_sen"] = self._midpoint(h, l, self.kijun_period)

        result["senkou_span_a"] = (
            (result["tenkan_sen"] + result["kijun_sen"]) / 2
        ).shift(self.kijun_period)

        result["senkou_span_b"] = self._midpoint(
            h, l, self.senkou_b_period
        ).shift(self.kijun_period)

        result["chikou_span"] = c.shift(-self.chikou_shift)

        return result

    def signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        売買シグナルを生成する。

        シグナル種類:
        - tenkan_kijun_cross : 転換線と基準線のクロス (1=買い, -1=売り, 0=なし)
        - price_vs_cloud     : 価格と雲の関係 (1=雲の上, -1=雲の下, 0=雲の中)
        - cloud_color        : 雲の色 (1=陽雲/強気, -1=陰雲/弱気)
        - chikou_vs_price    : 遅行スパンと過去価格の位置関係 (1=上, -1=下)
        """
        calc = self.calculate(df)
        signals = pd.DataFrame(index=calc.index)

        # 転換線・基準線クロス
        prev_tenkan = calc["tenkan_sen"].shift(1)
        prev_kijun = calc["kijun_sen"].shift(1)
        signals["tenkan_kijun_cross"] = 0
        signals.loc[
            (prev_tenkan <= prev_kijun) & (calc["tenkan_sen"] > calc["kijun_sen"]),
            "tenkan_kijun_cross",
        ] = 1
        signals.loc[
            (prev_tenkan >= prev_kijun) & (calc["tenkan_sen"] < calc["kijun_sen"]),
            "tenkan_kijun_cross",
        ] = -1

        # 価格 vs 雲
        cloud_top = calc[["senkou_span_a", "senkou_span_b"]].max(axis=1)
        cloud_bottom = calc[["senkou_span_a", "senkou_span_b"]].min(axis=1)
        signals["price_vs_cloud"] = 0
        signals.loc[calc["close"] > cloud_top, "price_vs_cloud"] = 1
        signals.loc[calc["close"] < cloud_bottom, "price_vs_cloud"] = -1

        # 雲の色 (先行スパンA > B → 陽雲)
        signals["cloud_color"] = np.where(
            calc["senkou_span_a"] > calc["senkou_span_b"], 1, -1
        )

        # 遅行スパン vs 過去の価格
        past_close = calc["close"].shift(self.chikou_shift)
        signals["chikou_vs_price"] = np.where(
            calc["close"] > past_close, 1, -1
        )

        # 総合スコア (-4 ~ +4)
        signals["composite_score"] = (
            signals["tenkan_kijun_cross"]
            + signals["price_vs_cloud"]
            + signals["cloud_color"]
            + signals["chikou_vs_price"]
        )

        return signals
