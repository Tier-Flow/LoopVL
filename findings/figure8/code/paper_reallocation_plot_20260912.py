from __future__ import annotations

import argparse

import hashlib

import json

import math

import textwrap

from dataclasses import dataclass

from pathlib import Path

from typing import Any

import matplotlib

import matplotlib.pyplot as plt

import numpy as np

from matplotlib import colors, font_manager

from matplotlib.cm import ScalarMappable

from matplotlib.collections import LineCollection, PatchCollection

from matplotlib.patches import Rectangle

from PIL import Image, ImageOps

TEXT = "#20313D"

MUTED = "#647482"

TARGET = "#13937D"

PROMOTED_TARGET = "#E45324"

PROMOTED_OTHER = "#75BAC5"

CMAP = "magma"

DIFF_CMAP = "RdBu_r"

TRANSLATIONS = {
    "s08": ("大正方形是什么颜色？", "粉色"),
    "s09": ("紫色圆在哪里？", "左上"),
    "s22": ("青色方块有几个？", "2"),
    "s29": ("框内的两字符代码是什么？", "B6"),
    "s31": ("框内的两字符代码是什么？", "X8"),
}

COLOR_ZH = {"red": "红色", "green": "绿色", "blue": "蓝色", "yellow": "黄色",
            "orange": "橙色", "purple": "紫色", "pink": "粉色", "cyan": "青色",
            "gray": "灰色", "black": "黑色", "white": "白色"}

SHAPE_ZH = {"square": "方块", "circle": "圆", "star": "星形", "triangle": "三角形"}

POSITION_ZH = {"top left": "左上", "top right": "右上", "bottom left": "左下", "bottom right": "右下"}

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def normalize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(array).all() or np.any(array < -1e-10):
        raise ValueError("Attention contains negative or non-finite values")
    array = np.maximum(array, 0)
    if array.sum() <= 0:
        raise ValueError("No visual attention in this snapshot")
    return array / array.sum()

def average_ranks(values: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    starts = ends - counts
    return ((starts + ends - 1) / 2)[inverse]

def fit_question(question: str) -> str:
    # Chinese headings: break before the final clause, with at most two lines.
    if "\n" in question or len(question) <= 13:
        return question
    if any("\u4e00" <= c <= "\u9fff" for c in question):
        split = 10 if len(question) <= 20 else math.ceil(len(question) / 2)
        return question[:split] + "\n" + question[split:]
    return "\n".join(textwrap.wrap(question, width=29))

@dataclass
class Case:
    sid: str
    sample: dict[str, Any]
    first: np.ndarray
    second: np.ndarray
    target: np.ndarray | None
    gh: int
    gw: int
    image: np.ndarray
    image_path: Path
    state_path: Path
    source_definition: dict[str, Any]

    @property
    def question(self) -> str:
        if self.sample.get("question_zh"):
            return self.sample["question_zh"]
        if self.sid in TRANSLATIONS:
            return TRANSLATIONS[self.sid][0]
        family = self.sample.get("family")
        shape = SHAPE_ZH.get(self.sample.get("target_shape"), "目标图形")
        color = COLOR_ZH.get(self.sample.get("target_color"), "")
        if family == "position":
            return f"{color}{shape}在哪里？"
        if family == "count":
            return f"{color}{shape}有几个？"
        if family == "color":
            return f"大{shape}是什么颜色？"
        if family == "ocr":
            return "框内的两字符代码是什么？"
        return self.sample.get("prompt", self.sid)

    @property
    def answer(self) -> str:
        if self.sample.get("answer_zh"):
            return str(self.sample["answer_zh"])
        if self.sid in TRANSLATIONS:
            return TRANSLATIONS[self.sid][1]
        value = str(self.sample.get("answer", ""))
        return COLOR_ZH.get(value, POSITION_ZH.get(value, value))

    @property
    def promoted(self) -> np.ndarray:
        return ((self.first <= np.quantile(self.first, .5)) &
                (self.second >= np.quantile(self.second, .75)))

    @property
    def extent(self) -> tuple[float, float, float, float]:
        height, width = self.image.shape[:2]
        factor = max(height, width)
        x0, y0 = (1 - width / factor) / 2, (1 - height / factor) / 2
        return x0, 1 - x0, 1 - y0, y0

def case_metrics(case: Case) -> dict[str, Any]:
    a, b, promoted = case.first, case.second, case.promoted
    high = b >= np.quantile(b, .75)
    ra, rb = average_ranks(a), average_ranks(b)
    info: dict[str, Any] = {
        "id": case.sid, "sample": case.sample,
        "question_zh": case.question, "answer_zh": case.answer,
        "grid": [case.gh, case.gw], "visual_token_count": int(a.size),
        "promoted_count": int(promoted.sum()), "second_high_count": int(high.sum()),
        "promoted_fraction_of_second_high": float(promoted.sum() / high.sum()),
        "promoted_attention_share_first": float(a[promoted].sum()),
        "promoted_attention_share_second": float(b[promoted].sum()),
        "rank_correlation_spearman": float(np.corrcoef(ra, rb)[0, 1]),
        "promoted_token_indices": np.flatnonzero(promoted).tolist(),
        "first_attention": a.tolist(), "second_attention": b.tolist(),
        "source": case.source_definition,
        "state_path": str(case.state_path), "state_sha256": sha256(case.state_path),
        "image_path": str(case.image_path), "image_sha256": sha256(case.image_path),
    }
    if case.target is not None:
        selected = promoted & case.target
        target_first, target_second = float(a[case.target].sum()), float(b[case.target].sum())
        info.update({
            "target_attention_share_first": target_first,
            "target_attention_share_second": target_second,
            "target_attention_share_delta": target_second - target_first,
            "target_promoted_count": int(selected.sum()),
            "target_promoted_attention_share_first": float(a[selected].sum()),
            "target_promoted_attention_share_second": float(b[selected].sum()),
            "target_promoted_fraction_of_target_attention_second": float(b[selected].sum() / max(target_second, 1e-15)),
            "target_promoted_token_indices": np.flatnonzero(selected).tolist(),
        })
    return info

def image_base(ax: Any, case: Case, light: bool = False) -> None:
    rgb = case.image.astype(np.float32) / 255
    gray = rgb @ np.array([.2126, .7152, .0722], dtype=np.float32)
    neutral = .56 * rgb + .44 * gray[..., None]
    if light:
        neutral = .65 * neutral + .35
    else:
        neutral = .40 * neutral + .10
    ax.imshow(neutral, extent=case.extent, interpolation="nearest", zorder=0)
    ax.set(xlim=(0, 1), ylim=(1, 0), aspect="equal")
    crop = case.sample.get('display_crop')
    if crop:
        # Display-only crop; all rankings, statistics and token locations still
        # come from the entire original model input, with no attention rerun.
        x0, x1, y1, y0 = case.extent
        ih, iw = case.image.shape[:2]
        l,t,r,b = crop
        ax.set(xlim=(x0+l/iw*(x1-x0), x0+r/iw*(x1-x0)),
               ylim=(y0+b/ih*(y1-y0), y0+t/ih*(y1-y0)))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#DCE2E6")
        spine.set_linewidth(.55)

def target_box(ax: Any, case: Case) -> None:
    # Optional semantic annotations for new real images. Old single-bbox
    # figures retain exactly the original drawing behavior below.
    if case.sample.get("target_bboxes") or case.sample.get("target_polygons"):
        from matplotlib.patches import Polygon
        x0, x1, y1, y0 = case.extent
        ih, iw = case.image.shape[:2]
        for left, top, right, bottom in case.sample.get("target_bboxes", []):
            ax.add_patch(Rectangle(
                (x0 + left / iw * (x1-x0), y0 + top / ih * (y1-y0)),
                (right-left)/iw*(x1-x0), (bottom-top)/ih*(y1-y0),
                fill=False, edgecolor=TARGET, linewidth=1.10, zorder=7))
        for polygon in case.sample.get("target_polygons", []):
            points = [(x0+x/iw*(x1-x0), y0+y/ih*(y1-y0)) for x,y in polygon]
            ax.add_patch(Polygon(points, closed=True, fill=False,
                                 edgecolor=TARGET, linewidth=1.10, zorder=7))
    bbox = case.sample.get("target_bbox")
    if not bbox:
        return
    x0, x1, y1, y0 = case.extent
    ih, iw = case.image.shape[:2]
    left, top, right, bottom = bbox
    ax.add_patch(Rectangle(
        (x0 + left / iw * (x1 - x0), y0 + top / ih * (y1 - y0)),
        (right - left) / iw * (x1 - x0), (bottom - top) / ih * (y1 - y0),
        fill=False, edgecolor=TARGET, linewidth=1.10, zorder=7))

def token_geometry(case: Case) -> tuple[np.ndarray, np.ndarray]:
    x0, x1, y1, y0 = case.extent
    return np.linspace(x0, x1, case.gw + 1), np.linspace(y0, y1, case.gh + 1)

def attention_map(ax: Any, case: Case, values: np.ndarray, style: str,
                  norm: colors.Normalize, signed: bool = False) -> None:
    image_base(ax, case, light=signed)
    grid = values.reshape(case.gh, case.gw)
    cmap = plt.get_cmap(DIFF_CMAP if signed else CMAP)
    if signed:
        strength = np.clip(np.abs(grid) / max(float(norm.vmax), 1e-15), 0, 1)
    else:
        strength = np.clip(norm(grid), 0, 1)
    if style == "smooth":
        # Interpolation is display-only. Promotions and all numbers use the
        # original token grid. A floating-point resize avoids quantizing data.
        height, width = case.image.shape[:2]
        rendered = np.asarray(Image.fromarray(grid.astype(np.float32), mode="F").resize(
            (width, height), Image.Resampling.BILINEAR))
        scale = (np.clip(np.abs(rendered) / max(float(norm.vmax), 1e-15), 0, 1)
                 if signed else np.clip(norm(rendered), 0, 1))
        rgba = cmap(norm(rendered))
        rgba[..., 3] = (.83 * np.power(scale, .55) if signed
                        else .88 * np.power(scale, .55))
        ax.imshow(rgba, extent=case.extent, interpolation="nearest", zorder=2)
    elif style == "token_grid":
        rgba = cmap(norm(grid))
        rgba[..., 3] = .88 * np.power(strength, .55)
        ax.imshow(rgba, extent=case.extent, interpolation="nearest", zorder=2)
        xs, ys = token_geometry(case)
        lines = [[(x, ys[0]), (x, ys[-1])] for x in xs]
        lines += [[(xs[0], y), (xs[-1], y)] for y in ys]
        ax.add_collection(LineCollection(lines, colors="#FFFFFF", linewidths=.23,
                                         alpha=.34, zorder=3))
    elif style == "token_dots":
        xs, ys = token_geometry(case)
        x, y = np.meshgrid((xs[:-1] + xs[1:]) / 2, (ys[:-1] + ys[1:]) / 2)
        rgba = cmap(norm(grid)).reshape(-1, 4)
        rgba[:, 3] = .13 + .84 * strength.ravel() ** .65
        # Radius is bounded by token-cell size in display points.
        width_points = ax.get_position().width * ax.figure.get_figwidth() * 72
        maximum_area = (.70 * width_points * (xs[-1] - xs[0]) / case.gw) ** 2
        areas = 1.0 + maximum_area * strength.ravel()
        ax.scatter(x.ravel(), y.ravel(), s=areas, c=rgba, linewidths=.18,
                   edgecolors="#FFFFFF70", zorder=3)
    else:
        raise ValueError(style)
    target_box(ax, case)

def promoted_map(ax: Any, case: Case) -> None:
    image_base(ax, case, light=True)
    xs, ys = token_geometry(case)
    promoted = case.promoted.reshape(case.gh, case.gw)
    target = (case.target.reshape(case.gh, case.gw) if case.target is not None
              else np.zeros_like(promoted))
    for inside, color, alpha in [(False, PROMOTED_OTHER, .42), (True, PROMOTED_TARGET, .90)]:
        selection = promoted & (target if inside else ~target)
        patches = [Rectangle((xs[col], ys[row]), xs[col + 1] - xs[col],
                             ys[row + 1] - ys[row])
                   for row, col in np.argwhere(selection)]
        if patches:
            ax.add_collection(PatchCollection(patches, facecolor=color,
                                             edgecolor="white", linewidth=.42,
                                             alpha=alpha, zorder=4 if inside else 2))
    target_box(ax, case)
