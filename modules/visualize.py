from __future__ import annotations

from typing import Iterable
import warnings

from matplotlib import font_manager
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import streamlit as st


def configure_chinese_font() -> None:
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
    ]
    installed = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            plt.rcParams["font.sans-serif"] = [name, *plt.rcParams.get("font.sans-serif", [])]
            break
    plt.rcParams["axes.unicode_minus"] = False
    warnings.filterwarnings("ignore", message="Glyph .* missing from font")


def image_grid(images: Iterable[Image.Image], captions: Iterable[str], columns: int = 4) -> None:
    cols = st.columns(columns)
    for index, (image, caption) in enumerate(zip(images, captions)):
        with cols[index % columns]:
            st.image(image, caption=caption, use_container_width=True)


def plot_history(history: list[dict], title: str, primary_key: str, secondary_key: str | None = None):
    configure_chinese_font()
    df = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ax.plot(df["epoch"], df[primary_key], marker="o", color="#2a9d8f", label=primary_key)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(primary_key)
    ax.grid(True, alpha=0.25)

    if secondary_key and secondary_key in df:
        ax2 = ax.twinx()
        ax2.plot(df["epoch"], df[secondary_key], marker="s", color="#e76f51", label=secondary_key)
        ax2.set_ylabel(secondary_key)
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="best")
    else:
        ax.legend(loc="best")

    fig.tight_layout()
    return fig


def metric_table(rows: list[dict]) -> None:
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
