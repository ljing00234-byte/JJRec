#!/usr/bin/env python3
"""Generate report figures from the local my-agent data.

White canvas, dark blue-gray type, light blue grids, teal/orange/purple series,
direct labels, restrained legends, and 3750 x 1530 wide exports.

Usage:
    .venv/bin/python scripts/generate_visualizations.py
    .venv/bin/python scripts/generate_visualizations.py --out figures
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/my-agent-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle


# ---------------------------------------------------------------------------
# Visual system
INK = "#172235"
SLATE = "#5f6c84"
LIGHT_SLATE = "#9aa6b9"
GRID = "#d8e3ef"
WHITE = "#ffffff"
TEAL = "#07879a"
TEAL_LIGHT = "#cfe8ec"
ORANGE = "#fb6a2a"
ORANGE_DARK = "#c74508"
ORANGE_LIGHT = "#fde2d4"
PURPLE = "#5d5ddd"
PURPLE_LIGHT = "#e8e8fb"
BLUE = "#347fd1"
BLUE_LIGHT = "#88b1e2"
GRAY_FILL = "#eef1f5"

WIDE = (12.5, 5.1)  # 300 dpi -> 3750 x 1530
DPI = 300


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial Unicode MS", "Arial", "DejaVu Sans"],
            "font.size": 11.5,
            "font.weight": "normal",
            "axes.unicode_minus": False,
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "axes.edgecolor": SLATE,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "xtick.color": SLATE,
            "ytick.color": SLATE,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.75,
            "axes.axisbelow": True,
            "legend.frameon": False,
            "legend.fontsize": 9.5,
            "lines.linewidth": 2.2,
            "lines.markersize": 6.5,
            "savefig.dpi": DPI,
            "savefig.bbox": None,
            "savefig.pad_inches": 0.08,
        }
    )


def clean_axes(ax: plt.Axes, *, grid_axis: str = "both") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.grid(axis=grid_axis)


def panel_title(ax: plt.Axes, title: str, subtitle: str | None = None) -> None:
    ax.set_title(title, loc="left", pad=10 if not subtitle else 22)
    if subtitle:
        ax.text(
            0,
            1.015,
            subtitle,
            transform=ax.transAxes,
            color=SLATE,
            fontsize=9.5,
            va="bottom",
        )


def footer(fig: plt.Figure, text: str) -> None:
    fig.text(0.01, 0.012, text, color=SLATE, fontsize=8.0, ha="left", va="bottom")


def save(fig: plt.Figure, out_dir: Path, name: str) -> Path:
    path = out_dir / name
    fig.savefig(path, dpi=DPI, bbox_inches=None, facecolor=WHITE)
    plt.close(fig)
    print(f"  OK  {name}  ({path.stat().st_size:,} bytes)")
    return path


# ---------------------------------------------------------------------------
# Data summaries
def summarize_standard(path: Path) -> dict:
    usecols = [
        "user_id",
        "video_id",
        "date",
        "is_click",
        "is_like",
        "is_follow",
        "is_comment",
        "is_forward",
        "is_hate",
        "long_view",
        "is_profile_enter",
        "duration_ms",
    ]
    signal_cols = [
        "is_click",
        "long_view",
        "is_profile_enter",
        "is_like",
        "is_comment",
        "is_hate",
        "is_follow",
        "is_forward",
    ]
    rows = 0
    users: set[int] = set()
    videos: set[int] = set()
    daily = defaultdict(lambda: {"rows": 0, "positives": 0})
    signals = {col: 0 for col in signal_cols}
    user_counts: dict[int, int] = {}
    first_duration_by_video: dict[int, float] = {}

    for chunk in pd.read_csv(path, usecols=usecols, chunksize=600_000):
        rows += len(chunk)
        users.update(chunk["user_id"].astype("int64").unique().tolist())
        videos.update(chunk["video_id"].astype("int64").unique().tolist())
        for col in signal_cols:
            signals[col] += int(chunk[col].sum())

        grouped = chunk.groupby("date", sort=False)["long_view"].agg(["size", "sum"])
        for date, row in grouped.iterrows():
            daily[int(date)]["rows"] += int(row["size"])
            daily[int(date)]["positives"] += int(row["sum"])

        counts = chunk["user_id"].value_counts()
        for user_id, count in counts.items():
            uid = int(user_id)
            user_counts[uid] = user_counts.get(uid, 0) + int(count)

        firsts = chunk[["video_id", "duration_ms"]].drop_duplicates("video_id")
        for video_id, duration_ms in firsts.itertuples(index=False):
            first_duration_by_video.setdefault(int(video_id), float(duration_ms))

    dates = sorted(daily)
    daily_rows = np.array([daily[d]["rows"] for d in dates], dtype=float)
    daily_rates = np.array(
        [daily[d]["positives"] / daily[d]["rows"] for d in dates], dtype=float
    )
    duration_seconds = np.fromiter(first_duration_by_video.values(), dtype=float) / 1000.0

    return {
        "rows": rows,
        "n_users": len(users),
        "n_videos": len(videos),
        "dates": pd.to_datetime([str(d) for d in dates]),
        "daily_rows": daily_rows,
        "daily_rates": daily_rates,
        "signal_rates": {col: count / rows for col, count in signals.items()},
        "user_counts": np.fromiter(user_counts.values(), dtype=int),
        "duration_seconds": duration_seconds,
    }


def summarize_history(path: Path) -> dict:
    hist_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    total_rows = 0
    for chunk in pd.read_csv(
        path, usecols=["hist_long_view_rate", "long_view"], chunksize=750_000
    ):
        total_rows += len(chunk)
        valid = chunk["hist_long_view_rate"].notna()
        hist_parts.append(chunk.loc[valid, "hist_long_view_rate"].to_numpy("float32"))
        label_parts.append(chunk.loc[valid, "long_view"].to_numpy("int8"))

    hist = np.concatenate(hist_parts)
    labels = np.concatenate(label_parts)
    edges = np.quantile(hist, np.linspace(0, 1, 11))
    bins = np.searchsorted(edges[1:-1], hist, side="right")
    means = []
    observed = []
    counts = []
    for index in range(10):
        mask = bins == index
        means.append(float(hist[mask].mean()))
        observed.append(float(labels[mask].mean()))
        counts.append(int(mask.sum()))

    return {
        "means": np.array(means),
        "observed": np.array(observed),
        "counts": np.array(counts),
        "correlation": float(np.corrcoef(hist, labels)[0, 1]),
        "missing": total_rows - len(hist),
    }


# ---------------------------------------------------------------------------
# Figure 01: governed data pipeline and temporal split
def draw_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    value: str,
    edge: str,
    face: str,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.035",
        linewidth=1.6,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height * 0.63, title, ha="center", va="center", color=INK)
    ax.text(
        x + width / 2,
        y + height * 0.30,
        value,
        ha="center",
        va="center",
        color=edge,
        fontsize=13,
        fontweight="bold",
    )


def figure_data_pipeline(summary: dict, quarantined_rows: int, out_dir: Path) -> Path:
    duplicate_rows = 94_224
    raw_rows = summary["rows"] + quarantined_rows + duplicate_rows
    train_rows = int(summary["daily_rows"][:14].sum())
    test_rows = int(summary["daily_rows"][14:].sum())

    fig = plt.figure(figsize=WIDE)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.03, 1.52], wspace=0.24)
    flow = fig.add_subplot(gs[0, 0])
    timeline = fig.add_subplot(gs[0, 1])

    panel_title(flow, "从原始日志到 Silver 层", "剔除与隔离分开记录，所有行都可核对")
    flow.set_xlim(0, 1)
    flow.set_ylim(0, 1)
    flow.axis("off")
    draw_box(flow, 0.05, 0.65, 0.37, 0.20, "原始 standard 日志", f"{raw_rows / 1e6:.3f}M 行", SLATE, GRAY_FILL)
    draw_box(flow, 0.58, 0.65, 0.37, 0.20, "Silver 主数据", f"{summary['rows'] / 1e6:.3f}M 行", TEAL, TEAL_LIGHT)
    flow.annotate("", xy=(0.58, 0.75), xytext=(0.42, 0.75), arrowprops={"arrowstyle": "->", "color": SLATE, "lw": 1.8})
    draw_box(flow, 0.05, 0.25, 0.37, 0.20, "完全重复（剔除）", f"−{duplicate_rows:,}", ORANGE, ORANGE_LIGHT)
    draw_box(flow, 0.58, 0.25, 0.37, 0.20, "时长缺失视频（隔离）", f"−{quarantined_rows:,}", ORANGE_DARK, ORANGE_LIGHT)
    flow.annotate("", xy=(0.235, 0.45), xytext=(0.235, 0.65), arrowprops={"arrowstyle": "->", "color": LIGHT_SLATE, "lw": 1.4})
    flow.annotate("", xy=(0.765, 0.45), xytext=(0.765, 0.65), arrowprops={"arrowstyle": "->", "color": LIGHT_SLATE, "lw": 1.4})
    flow.text(0.50, 0.08, f"核对：{summary['rows']:,} + {quarantined_rows:,} + {duplicate_rows:,} = {raw_rows:,}", ha="center", color=SLATE, fontsize=9.2)

    panel_title(timeline, "时间外切分", "同一条全局历史轴；切分只决定训练或评估")
    start = pd.Timestamp("2022-04-08")
    split = pd.Timestamp("2022-04-22")
    end = pd.Timestamp("2022-05-09")
    x0 = mdates.date2num(start)
    xs = mdates.date2num(split)
    x1 = mdates.date2num(end)
    timeline.broken_barh([(x0, xs - x0)], (1.22, 0.48), facecolors=BLUE_LIGHT)
    timeline.broken_barh([(xs, x1 - xs)], (1.22, 0.48), facecolors=BLUE)
    timeline.text((x0 + xs) / 2, 1.46, f"训练 · 04-08—04-21\n{train_rows / 1e6:.3f}M 行", ha="center", va="center", color=INK, fontsize=10.5)
    timeline.text((xs + x1) / 2, 1.46, f"测试 · 04-22—05-08\n{test_rows / 1e6:.3f}M 行", ha="center", va="center", color=WHITE, fontsize=10.5)
    timeline.broken_barh([(x0, x1 - x0)], (0.38, 0.34), facecolors=TEAL_LIGHT, edgecolors=TEAL, linewidth=1.2)
    timeline.text((x0 + x1) / 2, 0.55, "历史特征：只看当前时刻之前的全量记录", ha="center", va="center", color=TEAL, fontsize=10.5, fontweight="bold")
    timeline.axvline(split, color=SLATE, linestyle=(0, (4, 3)), linewidth=1.3)
    timeline.text(split + pd.Timedelta(hours=5), 1.83, "训练 / 测试边界", color=SLATE, fontsize=9.2, ha="left")
    timeline.set_xlim(start - pd.Timedelta(days=1), end + pd.Timedelta(days=1))
    timeline.set_ylim(0.05, 2.08)
    timeline.set_yticks([1.46, 0.55])
    timeline.set_yticklabels(["建模数据", "历史可见性"])
    timeline.set_xticks(
        pd.to_datetime(["2022-04-12", "2022-04-19", "2022-04-26", "2022-05-03", "2022-05-08"])
    )
    timeline.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    clean_axes(timeline, grid_axis="x")
    timeline.spines["left"].set_visible(False)
    timeline.tick_params(axis="y", length=0)

    footer(fig, "数据：data_cleaned/standard_cleaned.csv、quarantined_removed_videos.csv；清洗口径：WORK_LOG.md 决策 15–17")
    fig.subplots_adjust(left=0.035, right=0.99, top=0.87, bottom=0.15)
    return save(fig, out_dir, "01_data_pipeline.png")


# ---------------------------------------------------------------------------
# Figure 02: daily volume, prevalence, and label sparsity
def figure_daily_profile(summary: dict, out_dir: Path) -> Path:
    fig = plt.figure(figsize=WIDE)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 0.95], hspace=0.42, wspace=0.36)
    ax_rows = fig.add_subplot(gs[0, :2])
    ax_rate = fig.add_subplot(gs[1, :2], sharex=ax_rows)
    ax_signals = fig.add_subplot(gs[:, 2])

    dates = summary["dates"]
    colors = [BLUE_LIGHT if date < pd.Timestamp("2022-04-22") else BLUE for date in dates]
    ax_rows.bar(dates, summary["daily_rows"] / 1000, width=0.82, color=colors)
    ax_rows.axvline(pd.Timestamp("2022-04-22"), color=SLATE, linestyle=(0, (4, 3)), linewidth=1.2)
    panel_title(ax_rows, "每日数据规模", f"31 天 · {summary['rows'] / 1e6:.3f}M 行 · {summary['n_users']:,} 用户")
    ax_rows.set_ylabel("交互行（千）")
    clean_axes(ax_rows, grid_axis="y")
    ax_rows.tick_params(axis="x", labelbottom=False)

    rates = summary["daily_rates"] * 100
    overall = float(np.average(rates, weights=summary["daily_rows"]))
    ax_rate.plot(dates, rates, color=TEAL, marker="o", markersize=4.6)
    ax_rate.axhline(overall, color=SLATE, linestyle=(0, (3, 3)), linewidth=1.1)
    ax_rate.text(dates[-1], overall + 0.25, f"全期 {overall:.2f}%", color=SLATE, fontsize=8.8, ha="right")
    ax_rate.axvline(pd.Timestamp("2022-04-22"), color=SLATE, linestyle=(0, (4, 3)), linewidth=1.2)
    panel_title(ax_rate, "long_view 每日正例率", "训练段与测试段的日级波动保持在窄区间")
    ax_rate.set_ylabel("正例率")
    ax_rate.yaxis.set_major_formatter(lambda x, pos: f"{x:.0f}%")
    ax_rate.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax_rate.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    clean_axes(ax_rate, grid_axis="y")

    labels = ["点击", "长播", "进主页", "点赞", "评论", "负反馈", "关注", "转发"]
    keys = ["is_click", "long_view", "is_profile_enter", "is_like", "is_comment", "is_hate", "is_follow", "is_forward"]
    values = np.array([summary["signal_rates"][key] * 100 for key in keys])
    y = np.arange(len(labels))[::-1]
    point_colors = [SLATE, TEAL, SLATE, LIGHT_SLATE, LIGHT_SLATE, LIGHT_SLATE, LIGHT_SLATE, LIGHT_SLATE]
    for yi, value, color in zip(y, values, point_colors):
        ax_signals.plot([0, value], [yi, yi], color=color, linewidth=2.2)
        ax_signals.scatter([value], [yi], s=64 if color == TEAL else 48, color=color, zorder=3)
        ax_signals.text(41.5, yi, f"{value:.2f}%" if value >= 1 else f"{value:.3f}%", color=color if color == TEAL else SLATE, fontsize=9.5, ha="right", va="center", fontweight="bold" if color == TEAL else "normal")
    ax_signals.set_yticks(y)
    ax_signals.set_yticklabels(labels)
    ax_signals.set_xlim(0, 43)
    ax_signals.xaxis.set_major_formatter(lambda x, pos: f"{x:.0f}%")
    ax_signals.set_xlabel("全期交互行占比")
    panel_title(ax_signals, "行为标签稀疏度", "长播是密集信号；显式互动普遍稀疏")
    clean_axes(ax_signals, grid_axis="x")

    footer(fig, "数据：data_cleaned/standard_cleaned.csv（按日聚合；行为率按交互行计算）")
    fig.subplots_adjust(left=0.065, right=0.985, top=0.87, bottom=0.14)
    return save(fig, out_dir, "02_daily_data_profile.png")


# ---------------------------------------------------------------------------
# Figure 03: duration distribution and history-signal relationship
def figure_duration_history(summary: dict, history: dict, out_dir: Path) -> Path:
    fig, (ax_duration, ax_history) = plt.subplots(1, 2, figsize=WIDE, gridspec_kw={"wspace": 0.30})

    duration = summary["duration_seconds"]
    bins = np.logspace(np.log10(max(0.25, duration.min())), np.log10(duration.max()), 55)
    ax_duration.hist(duration, bins=bins, color=BLUE_LIGHT, edgecolor="none")
    ax_duration.set_xscale("log")
    median = float(np.median(duration))
    p99 = float(np.quantile(duration, 0.99))
    ax_duration.axvline(18, color=SLATE, linestyle=(0, (4, 3)), linewidth=1.4)
    ax_duration.axvline(median, color=ORANGE, linewidth=2.4)
    ax_duration.axvline(p99, color=ORANGE_DARK, linestyle=(0, (5, 3)), linewidth=2.0)
    ymax = ax_duration.get_ylim()[1]
    ax_duration.text(18 * 0.94, ymax * 0.94, "18 秒长播规则", color=SLATE, fontsize=9.2, ha="right", va="top", rotation=90)
    ax_duration.text(median * 1.05, ymax * 0.88, f"中位数 {median:.1f} 秒", color=ORANGE, fontsize=11.5, fontweight="bold", rotation=90, va="top")
    ax_duration.text(p99 * 1.06, ymax * 0.88, f"P99 {p99:.0f} 秒", color=ORANGE_DARK, fontsize=11.5, fontweight="bold", rotation=90, va="top")
    panel_title(ax_duration, "视频时长高度偏态", f"{summary['n_videos'] / 1e6:.3f}M 个唯一视频；横轴使用对数刻度")
    ax_duration.set_xlabel("视频时长（秒，对数）")
    ax_duration.set_ylabel("唯一视频数")
    ax_duration.yaxis.set_major_formatter(lambda x, pos: f"{x / 1000:.0f}k" if x >= 1000 else f"{x:.0f}")
    clean_axes(ax_duration, grid_axis="both")

    x = history["means"] * 100
    y = history["observed"] * 100
    limit = max(x.max(), y.max()) * 1.08
    ax_history.plot([0, limit], [0, limit], color=SLATE, linestyle=(0, (4, 3)), linewidth=1.2, label="完全一致")
    ax_history.plot(x, y, color=TEAL, marker="o", markersize=7.0, label="测试集十分位")
    for idx, (xx, yy) in enumerate(zip(x, y), start=1):
        if idx in (1, 5, 10):
            ax_history.text(xx + 1.0, yy - 0.3, f"D{idx}\n{yy:.1f}%", color=INK, fontsize=9.2, va="center")
    ax_history.text(0.97, 0.08, f"行级相关系数  r = {history['correlation']:.3f}\n每组约 {int(np.median(history['counts'])):,} 行", transform=ax_history.transAxes, color=SLATE, fontsize=10.0, ha="right", va="bottom")
    panel_title(ax_history, "历史长播率映射未来长播概率", f"测试集按历史长播率等频分成 10 组；仅 {history['missing']} 行无历史")
    ax_history.set_xlabel("组内平均历史长播率")
    ax_history.set_ylabel("组内实际 long_view 正例率")
    ax_history.xaxis.set_major_formatter(lambda value, pos: f"{value:.0f}%")
    ax_history.yaxis.set_major_formatter(lambda value, pos: f"{value:.0f}%")
    ax_history.set_xlim(0, limit)
    ax_history.set_ylim(0, limit)
    ax_history.set_aspect("equal", adjustable="box")
    ax_history.legend(loc="upper left")
    clean_axes(ax_history)

    footer(fig, "数据：standard_cleaned.csv（每个视频首次出现的 duration_ms）与 model2_features_test.csv（严格前序历史特征）")
    fig.subplots_adjust(left=0.075, right=0.985, top=0.87, bottom=0.15)
    return save(fig, out_dir, "03_duration_and_history_signal.png")


# ---------------------------------------------------------------------------
# Figure 04: model progression across four metrics
def figure_model_progression(out_dir: Path) -> Path:
    names = ["Model1\n122 个静态特征", "Model2\n+ 历史长播率", "序列模型\nGRU · 窗口 50"]
    colors = [PURPLE, TEAL, ORANGE]
    metrics = [
        ("AUC ↑", [0.643795, 0.738182, 0.771101], (0.62, 0.79), "+0.094", "+0.033"),
        ("PR-AUC ↑", [0.375541, 0.499545, 0.547503], (0.35, 0.57), "+0.124", "+0.048"),
        ("Log Loss ↓", [0.567868, 0.519266, 0.492565], (0.475, 0.585), "−0.049", "−0.027"),
        ("Brier ↓", [0.192142, 0.172884, 0.163774], (0.155, 0.199), "−0.019", "−0.009"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=WIDE, gridspec_kw={"wspace": 0.30})
    for ax, (title, values, ylim, delta_a, delta_b) in zip(axes, metrics):
        x = np.arange(3)
        ax.plot(x, values, color=LIGHT_SLATE, linewidth=1.6, zorder=1)
        for xx, value, color in zip(x, values, colors):
            ax.scatter(xx, value, s=90, color=color, edgecolor=WHITE, linewidth=1.2, zorder=3)
            offset = (ylim[1] - ylim[0]) * 0.045
            va = "bottom" if "↑" in title else "top"
            yy = value + offset if va == "bottom" else value - offset
            ax.text(xx, yy, f"{value:.3f}", ha="center", va=va, color=color, fontsize=10.0, fontweight="bold")
        ax.annotate(delta_a, xy=(0.5, (values[0] + values[1]) / 2), xytext=(0.5, ylim[1] - (ylim[1] - ylim[0]) * 0.09), ha="center", color=TEAL, fontsize=9.2, fontweight="bold", arrowprops={"arrowstyle": "-", "color": GRID})
        ax.annotate(delta_b, xy=(1.5, (values[1] + values[2]) / 2), xytext=(1.5, ylim[1] - (ylim[1] - ylim[0]) * 0.18), ha="center", color=ORANGE, fontsize=9.2, fontweight="bold", arrowprops={"arrowstyle": "-", "color": GRID})
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=8.7)
        ax.set_ylim(*ylim)
        ax.set_title(title, pad=10)
        clean_axes(ax, grid_axis="y")
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)

    fig.suptitle("模型能力递进：一个历史信号带来主跃迁，序列结构继续增益", fontsize=17, fontweight="bold", color=INK, y=0.98)
    footer(fig, "结果：WORK_LOG.md 决策 19、23；同一 04-21｜04-22 时间切分测试集")
    fig.subplots_adjust(left=0.035, right=0.99, top=0.82, bottom=0.20)
    return save(fig, out_dir, "04_model_progression.png")


# ---------------------------------------------------------------------------
# Figure 05: multi-cut temporal robustness
def figure_temporal_robustness(out_dir: Path) -> Path:
    split_labels = ["04-15\n训练 2.65M 行", "04-21\n训练 4.61M 行", "04-29\n训练 7.28M 行"]
    metrics = [
        ("AUC 提升", [0.0988, 0.0944, 0.0938]),
        ("PR-AUC 提升", [0.1292, 0.1240, 0.1215]),
        ("Log Loss 降低", [0.0508, 0.0486, 0.0479]),
        ("Brier 降低", [0.0202, 0.0193, 0.0188]),
    ]
    fig, axes = plt.subplots(1, 4, figsize=WIDE, gridspec_kw={"wspace": 0.31})
    x = np.arange(3)
    for ax, (title, values_list) in zip(axes, metrics):
        values = np.array(values_list)
        spread = max(float(np.ptp(values)), float(values.mean()) * 0.035)
        lower = max(0, float(values.min() - spread * 1.4))
        upper = float(values.max() + spread * 1.8)
        ax.axhspan(values.min(), values.max(), color=TEAL_LIGHT, alpha=0.72, zorder=0)
        ax.plot(x, values, color=TEAL, marker="o", markersize=7.5)
        for xx, value in zip(x, values):
            ax.text(xx, value + (upper - lower) * 0.055, f"+{value:.4f}", ha="center", color=INK, fontsize=9.3, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(split_labels, fontsize=8.6)
        ax.set_ylim(lower, upper)
        ax.set_title(title)
        ax.yaxis.set_major_formatter(lambda value, pos: f"{value:.3f}")
        clean_axes(ax, grid_axis="y")
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)

    fig.suptitle("三刀时间验证：Model2 对 Model1 的改进方向与幅度都稳定", fontsize=17, fontweight="bold", color=INK, y=0.98)
    fig.text(0.5, 0.105, "浅青条带 = 三次切分的观测范围；损失指标已转换为“降低量”，因此四个面板均为越高越好。", color=SLATE, fontsize=9.2, ha="center")
    footer(fig, "结果：WORK_LOG.md 决策 20；每个切分都重新拟合训练期编码器，避免跨时间信息泄漏")
    fig.subplots_adjust(left=0.035, right=0.99, top=0.81, bottom=0.24)
    return save(fig, out_dir, "05_temporal_robustness.png")


# ---------------------------------------------------------------------------
# Figure 06: domain shift
def figure_domain_shift(out_dir: Path) -> Path:
    domains = ["标准测试集", "随机曝光域"]
    colors = [BLUE, ORANGE]
    prevalence = np.array([0.277786, 0.086252])
    auc = np.array([0.738185, 0.681952])
    pr_auc = np.array([0.499453, 0.158395])

    fig, axes = plt.subplots(1, 3, figsize=WIDE, gridspec_kw={"wspace": 0.34})
    x = np.arange(2)

    ax = axes[0]
    ax.bar(x, prevalence * 100, color=colors, width=0.58)
    for xx, value, color in zip(x, prevalence, colors):
        ax.text(xx, value * 100 + 0.9, f"{value:.1%}", ha="center", color=color, fontweight="bold")
    ax.set_ylim(0, 32)
    ax.set_xticks(x, domains)
    ax.set_ylabel("long_view 正例率")
    ax.yaxis.set_major_formatter(lambda value, pos: f"{value:.0f}%")
    panel_title(ax, "标签基准率改变", "随机曝光仅为标准域的 31.05%")
    clean_axes(ax, grid_axis="y")

    ax = axes[1]
    ax.bar(x, auc, color=colors, width=0.58)
    ax.axhline(0.5, color=SLATE, linestyle=(0, (4, 3)), linewidth=1.2)
    ax.text(1.46, 0.505, "随机排序 0.5", color=SLATE, fontsize=8.8, ha="right", va="bottom")
    for xx, value, color in zip(x, auc, colors):
        ax.text(xx, value + 0.012, f"{value:.3f}", ha="center", color=color, fontweight="bold")
    ax.annotate("−0.056", xy=(0.5, auc.mean()), xytext=(0.5, 0.785), ha="center", color=ORANGE_DARK, fontweight="bold", arrowprops={"arrowstyle": "-", "color": GRID})
    ax.set_ylim(0.46, 0.81)
    ax.set_xticks(x, domains)
    panel_title(ax, "AUC 真实下降", "对正例比例不敏感，但仍明显高于 0.5")
    clean_axes(ax, grid_axis="y")

    ax = axes[2]
    ax.bar(x, pr_auc, color=colors, width=0.58, label="模型 PR-AUC")
    ax.bar(x, prevalence, color=WHITE, edgecolor=colors, linewidth=1.8, width=0.28, label="无技巧基线 = 正例率")
    for xx, score, base, color in zip(x, pr_auc, prevalence, colors):
        ax.text(xx, score + 0.018, f"PR-AUC {score:.3f}", ha="center", color=color, fontweight="bold")
        ax.text(xx, base / 2, f"基线\n{base:.3f}", ha="center", va="center", color=color, fontsize=8.7)
        ax.text(xx, -0.047, f"{score / base:.2f}× 基线", ha="center", color=INK, fontsize=9.2, fontweight="bold")
    ax.set_ylim(-0.07, 0.57)
    ax.set_xticks(x, domains)
    panel_title(ax, "PR-AUC 随基准率缩放", "绝对值下降，但相对无技巧基线的倍数近似不变")
    clean_axes(ax, grid_axis="y")

    fig.suptitle("随机曝光域：规律仍存在，但标准推荐环境提供了部分排序优势", fontsize=17, fontweight="bold", color=INK, y=0.98)
    footer(fig, "结果：WORK_LOG.md 决策 22；同一个 Model2 从标准训练段直接迁移到随机曝光日志，不重新训练")
    fig.subplots_adjust(left=0.065, right=0.985, top=0.80, bottom=0.20)
    return save(fig, out_dir, "06_domain_shift.png")


# ---------------------------------------------------------------------------
# Figure 07: sequence-window trade-off
def figure_window_tradeoff(out_dir: Path) -> Path:
    windows = np.array([20, 50, 100])
    colors = [PURPLE, TEAL, ORANGE]
    data = {
        "AUC ↑": ([0.769201, 0.771101, 0.771612], 0.738182),
        "PR-AUC ↑": ([0.543981, 0.547503, 0.548034], 0.499545),
        "Log Loss ↓": ([0.494077, 0.492565, 0.494160], 0.519266),
        "Brier ↓": ([0.164367, 0.163774, 0.164415], 0.172884),
    }
    training_seconds = np.array([235.2, 557.2, 2608.4])

    fig = plt.figure(figsize=WIDE)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.04], hspace=0.48, wspace=0.34)
    perf_axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    cost_ax = fig.add_subplot(gs[:, 2])

    for ax, (title, (values_list, baseline)) in zip(perf_axes, data.items()):
        values = np.array(values_list)
        ax.axvspan(42, 58, color=TEAL_LIGHT, alpha=0.65, zorder=0)
        ax.axhline(baseline, color=SLATE, linestyle=(0, (4, 3)), linewidth=1.1)
        ax.plot(windows, values, color=LIGHT_SLATE, linewidth=1.4)
        for window, value, color in zip(windows, values, colors):
            ax.scatter(window, value, s=70, color=color, edgecolor=WHITE, linewidth=1.0, zorder=3)
            ax.text(window, value, f" {value:.4f}", color=color, fontsize=8.5, va="bottom", ha="left")
        all_values = np.append(values, baseline)
        span = max(float(all_values.max() - all_values.min()), 0.003)
        ax.set_ylim(float(all_values.min() - span * 0.18), float(all_values.max() + span * 0.28))
        ax.set_xlim(12, 108)
        ax.set_xticks(windows)
        ax.set_title(title, loc="left")
        if ax in perf_axes[2:]:
            ax.set_xlabel("历史窗口（时间点）")
        else:
            ax.tick_params(axis="x", labelbottom=False)
        ax.text(0.98, 0.06, f"Model2 基线 {baseline:.4f}", transform=ax.transAxes, color=SLATE, fontsize=8.0, ha="right")
        clean_axes(ax, grid_axis="y")

    cost_ax.bar(np.arange(3), training_seconds / 60, color=colors, width=0.60)
    for xx, seconds, color in zip(np.arange(3), training_seconds, colors):
        cost_ax.text(xx, seconds / 60 + 1.1, f"{seconds / 60:.1f} 分", ha="center", color=color, fontweight="bold")
    cost_ax.set_xticks(np.arange(3), ["窗口 20", "窗口 50", "窗口 100"])
    cost_ax.set_ylabel("5 轮训练耗时（分钟）")
    panel_title(cost_ax, "训练成本快速增长", "窗口 100 比窗口 50 慢 4.7×，排序收益仅 +0.0005 AUC")
    clean_axes(cost_ax, grid_axis="y")
    cost_ax.annotate("当前平衡点", xy=(1, training_seconds[1] / 60), xytext=(0.55, 30), color=TEAL, fontsize=10.0, fontweight="bold", arrowprops={"arrowstyle": "->", "color": TEAL, "lw": 1.2})

    fig.suptitle("序列窗口：50 个时间点是当前效果与成本的平衡点", fontsize=17, fontweight="bold", color=INK, y=0.98)
    footer(fig, "结果：WORK_LOG.md 决策 24；三种窗口保持相同集合上限、模型结构与 5 轮训练预算")
    fig.subplots_adjust(left=0.065, right=0.985, top=0.80, bottom=0.16)
    return save(fig, out_dir, "07_window_tradeoff.png")


def count_data_rows(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(block.count(b"\n") for block in iter(lambda: handle.read(16 * 1024 * 1024), b"")) - 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    data_dir = root / "data_cleaned"
    out_dir = (args.out or (root / "figures")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    configure_style()

    required = [
        data_dir / "standard_cleaned.csv",
        data_dir / "quarantined_removed_videos.csv",
        data_dir / "model2_features_test.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required data files:\n" + "\n".join(missing))

    print("Summarizing local data...")
    standard = summarize_standard(required[0])
    history = summarize_history(required[2])
    quarantined_rows = count_data_rows(required[1])

    print("Rendering figures...")
    figure_data_pipeline(standard, quarantined_rows, out_dir)
    figure_daily_profile(standard, out_dir)
    figure_duration_history(standard, history, out_dir)
    figure_model_progression(out_dir)
    figure_temporal_robustness(out_dir)
    figure_domain_shift(out_dir)
    figure_window_tradeoff(out_dir)
    print(f"Done: {out_dir}")


if __name__ == "__main__":
    main()
