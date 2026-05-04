import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from .ichimoku import Ichimoku


def plot_ichimoku(
    df: pd.DataFrame,
    ichimoku: Ichimoku | None = None,
    title: str = "Ichimoku Kinko Hyo",
    figsize: tuple = (16, 8),
    save_path: str | None = None,
) -> plt.Figure:
    """一目均衡表のチャートを描画する。"""
    ichi = ichimoku or Ichimoku()
    calc = ichi.calculate(df)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title, fontsize=14)

    # ローソク足の代わりに終値ライン
    ax.plot(calc.index, calc["close"], color="black", linewidth=1.2, label="Close")

    # 転換線 (赤)
    ax.plot(
        calc.index, calc["tenkan_sen"],
        color="red", linewidth=0.8, alpha=0.9, label="Tenkan-sen (9)",
    )

    # 基準線 (青)
    ax.plot(
        calc.index, calc["kijun_sen"],
        color="blue", linewidth=0.8, alpha=0.9, label="Kijun-sen (26)",
    )

    # 遅行スパン (緑)
    ax.plot(
        calc.index, calc["chikou_span"],
        color="green", linewidth=0.8, alpha=0.7, label="Chikou Span",
    )

    # 雲 (先行スパンA/B の間を塗りつぶし)
    ax.plot(
        calc.index, calc["senkou_span_a"],
        color="orange", linewidth=0.5, alpha=0.6, label="Senkou Span A",
    )
    ax.plot(
        calc.index, calc["senkou_span_b"],
        color="purple", linewidth=0.5, alpha=0.6, label="Senkou Span B",
    )

    ax.fill_between(
        calc.index,
        calc["senkou_span_a"],
        calc["senkou_span_b"],
        where=calc["senkou_span_a"] >= calc["senkou_span_b"],
        color="orange",
        alpha=0.25,
    )
    ax.fill_between(
        calc.index,
        calc["senkou_span_a"],
        calc["senkou_span_b"],
        where=calc["senkou_span_a"] < calc["senkou_span_b"],
        color="purple",
        alpha=0.25,
    )

    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)

    return fig
