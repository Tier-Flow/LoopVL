from __future__ import annotations

import argparse

from collections import OrderedDict

import copy

import hashlib

import io

import json

from pathlib import Path

import pickle

import shutil

import zipfile

import matplotlib

import matplotlib.pyplot as plt

from matplotlib import colors

from matplotlib.cm import ScalarMappable

from matplotlib.lines import Line2D

from matplotlib.patches import Patch, Rectangle

import numpy as np

from PIL import Image, ImageOps

import paper_reallocation_plot_20260912 as spatial

ORDER8 = ['s08', 's22', 's31', 's29', 's21', 'rw_q10', 'new_r075', 'new_r093']

ORDER6 = ['s08', 's22', 's29', 'rw_q10', 'new_r075', 'new_r093']

HEADERS = {
    's08': ('What color is the\nlarge square?', 'Pink'),
    's22': ('How many\ncyan squares?', '2'),
    's31': ('What code is\ninside the box?', 'X8'),
    's29': ('What code is\ninside the box?', 'B6'),
    's21': ('How many\norange circles?', '1'),
    'rw_q10': ('How many debris pieces\nare beside the white van?', '3'),
    'new_r075': ('Which way does\nthe remote point?', 'Right'),
    'new_r093': ('Which way does\nthe arrow point?', 'Up'),
}

BLUE, ORANGE = '#276B88', '#E66B32'

def style_map(ax):
    # Scale line weights to small publication panels, without changing geometry.
    for patch in ax.patches:
        patch.set_linewidth(.55)
    for collection in ax.collections:
        collection.set_linewidth(.14)
        if isinstance(collection, matplotlib.collections.LineCollection):
            collection.set_alpha(.17)
    for spine in ax.spines.values():
        spine.set_linewidth(.35)

def scatter(ax, case, record, first_column):
    a = spatial.average_ranks(case.first) / (case.first.size - 1) * 100
    b = spatial.average_ranks(case.second) / (case.second.size - 1) * 100
    promoted = case.promoted
    ax.add_patch(Rectangle((0, 75), 50, 25, facecolor=ORANGE, alpha=.06, lw=0))
    ax.plot([0, 100], [0, 100], color='#D5DFE5', lw=.45, zorder=0)
    ax.axvline(50, lw=.4, ls=(0, (3, 3)), color=ORANGE, alpha=.7)
    ax.axhline(75, lw=.4, ls=(0, (3, 3)), color=ORANGE, alpha=.7)
    ax.scatter(a[~promoted], b[~promoted], s=1.8, color=BLUE,
               alpha=.62, linewidths=0, rasterized=True)
    ax.scatter(a[promoted], b[promoted], s=4.1, facecolors='white',
               edgecolors=ORANGE, linewidths=.42, rasterized=True)
    ax.set(xlim=(0, 100), ylim=(0, 100), aspect='equal',
           xticks=[0, 50, 100], yticks=[0, 50, 100])
    ax.tick_params(length=1.8, width=.35, colors=spatial.MUTED, labelsize=5.2, pad=1.6)
    if not first_column:
        ax.set_yticklabels([])
    for spine in ax.spines.values():
        spine.set_linewidth(.35)
        spine.set_color('#B9C8D1')
    ax.set_title(f"Promoted {100 * record['promoted_fraction_of_second_high']:.1f}%",
                 color=ORANGE, fontsize=6.1, pad=4)
    ax.text(.96, .05, f"$\\rho$ = {record['rank_correlation_spearman']:.2f}",
            transform=ax.transAxes, ha='right', va='bottom', fontsize=5.4,
            color=spatial.TEXT, bbox=dict(facecolor='white', alpha=.87, edgecolor='none', pad=1))

def draw_percentile_map(ax, case, values):
    """Display within-round order, not attention magnitude; no new inference."""
    spatial.image_base(ax, case)
    ranks = 100 * spatial.average_ranks(values) / (values.size - 1)
    rgba = plt.get_cmap('viridis')(colors.Normalize(0,100)(ranks.reshape(case.gh,case.gw)))
    rgba[...,3] = .75
    ax.imshow(rgba, extent=case.extent, interpolation='nearest', zorder=2)
    xs, ys = spatial.token_geometry(case)
    lines = [[(x,ys[0]),(x,ys[-1])] for x in xs]
    lines += [[(xs[0],y),(xs[-1],y)] for y in ys]
    ax.add_collection(matplotlib.collections.LineCollection(lines, colors='white',
                                                           linewidths=.25, alpha=.29, zorder=3))
    spatial.target_box(ax, case)

def render(cases, records, ids, out, dpi, three_rows=False, explain_promotion=False,
           map_style='token_grid', scale_quantile=.995, name_suffix=''):
    if map_style == 'percentile_rank' and not three_rows:
        raise ValueError('The rank-map variant is implemented only for the three-row layout.')
    n = len(ids)
    width, left, right, gx = 8.8, .49, .13, .14
    panel = (width - left - right - (n-1)*gx) / n
    top, gap, bottom, rank_gap = .72, .12, 1.06, .29
    height = top + 5*panel + 3*gap + rank_gap + bottom
    if three_rows:
        bottom = .55
        height = top + 3*panel + 2*gap + bottom
    fig = plt.figure(figsize=(width, height), facecolor='white', dpi=dpi)
    axes = [[None]*n for _ in range(3 if three_rows else 5)]
    all_axes, displays = [], []
    row_names = ['Round 1 (end)', 'Round 2 (end)', 'Difference', 'Promoted tokens', 'Round 2 rank']
    y_values = [height - top - panel - r*(panel+gap) for r in range(4)]
    y_values.append(bottom)
    if three_rows:
        row_names = ['Round 1 (end)', 'Round 2 (end)', 'Promoted tokens']
        y_values = [height-top-panel-r*(panel+gap) for r in range(3)]
    if explain_promotion:
        row_names = [label.replace('Promoted tokens', 'Promoted tokens\n(low-to-high rank)')
                     for label in row_names]
    if map_style == 'percentile_rank':
        row_names[:2] = ['Round 1 (end)\nrank', 'Round 2 (end)\nrank']
    for row, label in enumerate(row_names):
        fig.text(.15/width, (y_values[row]+panel/2)/height, label,
                 ha='center', va='center', rotation=90, fontsize=7.1,
                 color=spatial.TEXT, weight='medium')

    for col, sid in enumerate(ids):
        case, record = cases[sid], records[sid]
        x = left + col*(panel+gx)
        cx = x+panel/2
        question, answer = HEADERS[sid]
        fig.text(cx/width, (height-.045)/height, f'({chr(97+col)})',
                 ha='center', va='top', fontsize=6.4, weight='bold', color=spatial.TEXT)
        fig.text(cx/width, (height-.23)/height, question,
                 ha='center', va='top', fontsize=5.9, weight='medium',
                 color=spatial.TEXT, linespacing=1.25)
        fig.text(cx/width, (height-.54)/height, f'Ref. answer: {answer}',
                 ha='center', va='top', fontsize=5.8, color=spatial.MUTED)
        vmax = float(np.quantile(np.r_[case.first, case.second], scale_quantile))
        bound = float(np.quantile(np.abs(case.second-case.first), scale_quantile))
        norm = colors.PowerNorm(.5, vmin=0, vmax=vmax, clip=True)
        diffnorm = colors.TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)
        for row, y in enumerate(y_values):
            ax = fig.add_axes([x/width, y/height, panel/width, panel/height])
            axes[row][col] = ax
            all_axes.append(ax)
            source_row = [0,1,3][row] if three_rows else row
            if source_row < 3:
                values = [case.first, case.second, case.second-case.first][source_row]
                if map_style == 'percentile_rank':
                    draw_percentile_map(ax, case, values)
                else:
                    spatial.attention_map(ax, case, values, map_style,
                                          diffnorm if source_row == 2 else norm, signed=source_row == 2)
                style_map(ax)
            elif source_row == 3:
                spatial.promoted_map(ax, case)
                style_map(ax)
            else:
                scatter(ax, case, record, col == 0)
        # Per-case scales are explicit; the pair always shares a scale.
        bars = [
            (.60, spatial.CMAP, norm, [0, vmax], ['0', f'{vmax:.3f}']),
            (.36, spatial.DIFF_CMAP, diffnorm, [-bound, bound], [f'-{bound:.3f}', f'+{bound:.3f}']),
        ]
        if three_rows:
            bars = [(.31, spatial.CMAP, norm, [0, vmax], ['0', f'{vmax:.3f}'])]
        if map_style == 'percentile_rank':
            bars = [(.31, 'viridis', colors.Normalize(0,100), [0,100], ['0','100'])]
        for yy, cmap, scale, ticks, labels in bars:
            ca = fig.add_axes([(x+.045)/width, yy/height, (panel-.09)/width, .04/height])
            cb = fig.colorbar(ScalarMappable(norm=scale, cmap=cmap), cax=ca,
                              orientation='horizontal', ticks=ticks)
            cb.ax.set_xticklabels(labels)
            cb.ax.get_xticklabels()[0].set_ha('left')
            cb.ax.get_xticklabels()[-1].set_ha('right')
            cb.ax.tick_params(labelsize=4.6, pad=1.4, length=1.4, width=.3)
            cb.outline.set_linewidth(.3)
            cb.outline.set_edgecolor('#BCC6CD')
        displays.append({'id': sid, 'attention_vmax': vmax, 'difference_bound': bound,
                         'map_style':map_style, 'scale_quantile':scale_quantile,
                         'attention_norm': ('linear percentile rank 0-100' if map_style=='percentile_rank'
                                            else 'sqrt (PowerNorm gamma=0.5)'),
                         'display_crop': case.sample.get('display_crop'),
                         'attention_clipped_tokens': ([0,0] if map_style=='percentile_rank' else
                                                       [int((a>vmax).sum()) for a in [case.first,case.second]]),
                         'difference_clipped_tokens': int((np.abs(case.second-case.first)>bound).sum())})

    if not three_rows:
        fig.text((left+(width-left-right)/2)/width, .815/height,
                 'Round 1 rank (percentile; higher = more attention)', ha='center', fontsize=6.1, color=spatial.MUTED)
        fig.text(.15/width, .38/height, 'Diff.', ha='center', va='center', rotation=90,
                 fontsize=4.9, color=spatial.MUTED)
    fig.text(.15/width, (.33 if three_rows else .62)/height,
             'Rank (%)' if map_style=='percentile_rank' else 'Attn.',
             ha='center', va='center', rotation=90,
             fontsize=4.9, color=spatial.MUTED)
    handles = [Patch(facecolor='none', edgecolor=spatial.TARGET, lw=.65, label='Target region'),
               Patch(facecolor=spatial.PROMOTED_TARGET, label='Promoted: in target'),
               Patch(facecolor=spatial.PROMOTED_OTHER, label='Promoted: outside target'),
               Line2D([], [], marker='o', markersize=3.0, markerfacecolor='white',
                      markeredgecolor=ORANGE, markeredgewidth=.5, ls='none', label='Promoted in rank plot')]
    if three_rows:
        handles = handles[:3]
    fig.legend(handles=handles, loc='center', bbox_to_anchor=(.53, .09/height),
               ncol=len(handles), frameon=False, fontsize=5.5, handlelength=1.1,
               columnspacing=1.5, handletextpad=.45, borderaxespad=0)

    fig.canvas.draw()
    geometry = []
    for ax in all_axes:
        box = ax.get_position()
        w, h = box.width*width, box.height*height
        assert abs(w-h)<1e-5, (w, h)
        assert abs(w-panel)<1e-5
        geometry.append([w, h])
    name = f'visual_reallocation_{n}cases_paper' + ('_three_rows' if three_rows else '')
    if explain_promotion:
        name += '_explained'
    name += name_suffix
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(out/f'{name}.{ext}', dpi=dpi, facecolor='white')
    plt.close(fig)
    with Image.open(out/f'{name}.png') as im:
        im.thumbnail((2200, 2400), Image.Resampling.LANCZOS)
        im.save(out/f'{name}_preview.png')
    return {'name': name, 'cases': ids, 'size_inches': [width,height],
            'rows':row_names,
            'panel_side_inches': panel, 'axes_equal_square': True,
            'all_primary_axes_inches': geometry, 'display': displays}
