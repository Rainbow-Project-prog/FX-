"""
一目均衡表の使用例

サンプルデータ(ランダムウォーク)で一目均衡表を計算し、
チャート画像を出力する。
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from indicators import Ichimoku
from indicators.ichimoku_chart import plot_ichimoku


def generate_sample_ohlc(n_days: int = 200, seed: int = 42) -> pd.DataFrame:
    """ランダムウォークでサンプルOHLCデータを生成する。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start="2025-01-01", periods=n_days)

    close = 150.0 + np.cumsum(rng.normal(0, 0.5, n_days))
    high = close + rng.uniform(0.1, 1.0, n_days)
    low = close - rng.uniform(0.1, 1.0, n_days)
    open_ = close + rng.normal(0, 0.3, n_days)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close},
        index=dates,
    )


def main():
    df = generate_sample_ohlc()
    ichi = Ichimoku()

    # 一目均衡表の計算
    result = ichi.calculate(df)
    print("=== 一目均衡表 (直近5行) ===")
    print(
        result[
            ["close", "tenkan_sen", "kijun_sen", "senkou_span_a", "senkou_span_b", "chikou_span"]
        ].tail()
    )

    # シグナル生成
    signals = ichi.signals(df)
    print("\n=== シグナル (直近5行) ===")
    print(signals.tail())

    # チャート保存
    output_path = os.path.join(os.path.dirname(__file__), "..", "output", "ichimoku_chart.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plot_ichimoku(df, ichi, title="USD/JPY Ichimoku (Sample)", save_path=output_path)
    print(f"\nChart saved to: {output_path}")


if __name__ == "__main__":
    main()
