from pathlib import Path

import argparse

import hashlib

import json

import matplotlib

import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

from matplotlib.patches import Patch

from matplotlib.text import Text

import numpy as np

WIDTH = 470.268

FONT = {'title': 10.5, 'panel': 9.5, 'axis': 8.5, 'tick': 7.5,
        'legend': 7.5, 'small': 7.0}

INK, EDGE, GRID = '#3F5060', '#8798A9', '#E4EAF1'

ORANGE, BLUE, BAND = '#E5852B', '#5F8CC6', '#3167AE'

MODEL, MODULE_LINE, MODULE_TEXT = '#C84E60', '#E7C2C8', '#BD8994'

H_FILL = '#FCF3EC'

def canvas(height):
    return plt.figure(figsize=(WIDTH / 72, height / 72), dpi=200, facecolor='white')

def axes(fig, x, y, w, h):
    height = fig.get_figheight() * 72
    return fig.add_axes([x / WIDTH, y / height, w / WIDTH, h / height])

def style(ax):
    ax.tick_params(labelsize=FONT['tick'], colors=INK, length=2, width=.4, pad=2)
    for spine in ax.spines.values():
        spine.set_color(EDGE)
        spine.set_linewidth(.45)
    ax.set_axisbelow(True)

def save(fig, stem, preview_dir):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for artist in fig.findobj(Text):
        if artist.get_visible() and artist.get_text():
            box = artist.get_window_extent(renderer)
            assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= fig.bbox.width and box.y1 <= fig.bbox.height, (artist.get_text(), box.bounds)
    fig.savefig(HERE / f'{stem}.pdf', facecolor='white', metadata={'Title': stem.replace('_', ' ')})
    if preview_dir:
        fig.savefig(preview_dir / f'{stem}.png', dpi=240, facecolor='white')
    plt.close(fig)

def draw_states(data, preview_dir):
    depth = np.arange(16, 129, 16)
    values = np.array([r['delta_l2_visual'] for r in data['state_samples']])
    matrix = np.array([r['cosine_visual'] for r in data['state_samples']]).mean(0)
    assert values.shape == (32, 8) and matrix.shape == (8, 8)
    np.testing.assert_array_equal(values.mean(0), data['reference']['mean_delta_visual'])
    np.testing.assert_array_equal(matrix, data['reference']['cosine_visual'])
    fig = canvas(170)
    ax = axes(fig, 33, 40, 254, 98)
    hm = axes(fig, 327, 40, 98, 98)
    cax = axes(fig, 432, 40, 5, 98)
    for row in values:
        line, = ax.plot(depth, row, color=ORANGE, alpha=.13, lw=.36, zorder=1)
        np.testing.assert_array_equal(line.get_ydata(), row)
    line, = ax.plot(depth, values.mean(0), color=ORANGE, lw=1.0,
                    marker='o', ms=2, markeredgewidth=0, zorder=3)
    np.testing.assert_array_equal(line.get_ydata(), values.mean(0))
    ax.set(xlim=(12, 132), ylim=(44, 70), xticks=depth, yticks=[45, 50, 55, 60, 65, 70])
    ax.set_ylabel('Mean token L2 update', fontsize=FONT['axis'], labelpad=4)
    ax.grid(color=GRID, lw=.35)
    style(ax)
    lo, hi = data['cosine_limits']
    mesh = hm.pcolormesh(np.arange(9) - .5, np.arange(9) - .5, matrix,
                         cmap='viridis', vmin=lo, vmax=hi, shading='flat',
                         edgecolors='face', linewidth=.15, rasterized=False)
    hm.set(xlim=(-.5, 7.5), ylim=(-.5, 7.5), xticks=np.arange(8), yticks=np.arange(8))
    hm.set_xticklabels(depth)
    hm.set_yticklabels(depth)
    hm.set_aspect('equal', adjustable='box')
    hm.set_ylabel('Unrolled depth', fontsize=FONT['axis'], labelpad=3)
    style(hm)
    hm.tick_params(length=1.5, pad=1.5, labelsize=FONT['small'])
    plt.setp(hm.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    cb = fig.colorbar(mesh, cax=cax, ticks=[-.1, 0, .25, .5, .75, 1])
    cb.ax.set_yticklabels(['-0.1', '0', '0.25', '0.5', '0.75', '1'])
    cb.ax.tick_params(labelsize=FONT['small'], length=1.5, width=.4, pad=2)
    cb.set_label('Cosine similarity', fontsize=FONT['axis'], labelpad=2)
    cb.outline.set_edgecolor(EDGE)
    cb.outline.set_linewidth(.45)
    cb.solids.set_rasterized(False)
    cb.solids.set_edgecolor('face')
    for x, label in [(160, '(a) Visual-state updates'), (376, '(b) State similarity')]:
        fig.text(x / WIDTH, 162 / 170, label, ha='center', va='center', fontsize=FONT['panel'], weight='bold')
    for x in (160, 376):
        fig.text(x / WIDTH, 8 / 170, 'Unrolled depth', ha='center', va='center', fontsize=FONT['axis'])
    fig.legend(handles=[Line2D([0], [0], color=ORANGE, alpha=.3, lw=.5, label='Samples'),
                        Line2D([0], [0], color=ORANGE, lw=1.0, marker='o', ms=2, label='Mean (n=32)')],
               loc='center', bbox_to_anchor=(160 / WIDTH, 147 / 170), ncol=2, frameon=False,
               fontsize=FONT['legend'], handlelength=1.9, handletextpad=.5, columnspacing=1.6)
    np.testing.assert_array_equal(mesh.get_array().reshape(8, 8), matrix)
    np.testing.assert_array_equal(ax.get_position().bounds[1::2], hm.get_position().bounds[1::2])
    report = {'sample_count': 32, 'sample_lines': 32, 'mean_delta_visual': values.mean(0).tolist(),
              'cosine_visual': matrix.tolist(), 'cosine_limits': [lo, hi],
              'figure_size_pt': [WIDTH, 170], 'line_axes': list(ax.get_position().bounds),
              'heatmap_axes': list(hm.get_position().bounds), 'heatmap_square': True}
    save(fig, 'visual_hidden_state', preview_dir)
    return report

def draw_logits(data, preview_dir):
    depth = np.arange(1, 129)
    values = np.array([r['kl_final_to_intermediate'] for r in data['logit_samples']])
    assert values.shape == (32, 128) and np.isfinite(values).all()
    mean = values.mean(0)
    quartiles = np.quantile(values, [.25, .75], axis=0, method='linear')
    np.testing.assert_array_equal(mean, data['reference']['mean_kl'])
    np.testing.assert_array_equal(quartiles, data['reference']['quartiles'])
    fig = canvas(170)
    ax = axes(fig, 34, 43, WIDTH - 34.725, 87)
    for start, end in [(48.5, 64.5), (112.5, 128.5)]:
        ax.axvspan(start, end, facecolor=H_FILL, edgecolor='none', zorder=0)
    ax.fill_between(depth, quartiles[0], quartiles[1], color=BAND, alpha=.12,
                    linewidth=0, edgecolor='none', zorder=2)
    line, = ax.plot(depth, mean, color=BLUE, lw=1.1, zorder=4)
    np.testing.assert_array_equal(line.get_ydata(), mean)
    module_boundaries, model_boundary = [16.5, 32.5, 80.5, 96.5], 64.5
    annotations = []
    for boundary in module_boundaries + [model_boundary]:
        model = boundary == model_boundary
        ax.axvline(boundary, color=MODEL if model else MODULE_LINE,
                   lw=.7 if model else .5, linestyle=(0, (4, 3)), zorder=3)
        annotations.append(ax.text(boundary, 1.045, 'model loop' if model else 'module loop',
                                   transform=ax.get_xaxis_transform(), ha='center', va='bottom',
                                   color=MODEL if model else MODULE_TEXT, fontsize=FONT['small'], clip_on=False))
    ax.set(xlim=(.5, 128.5), ylim=(-.25, 15),
           xticks=[1, 16, 32, 48, 64, 80, 96, 112, 128], yticks=[0, 3, 6, 9, 12, 15])
    style(ax)
    ax.get_xticklabels()[0].set_ha('left')
    ax.get_xticklabels()[-1].set_ha('right')
    ax.grid(axis='y', color=GRID, lw=.35)
    ax.set_ylabel('KL divergence (nats)', fontsize=FONT['axis'], labelpad=4)
    ax.set_xlabel('Unrolled depth', fontsize=FONT['axis'], labelpad=3)
    center = (34 + (WIDTH - 34.725) / 2) / WIDTH
    title = fig.text(center, 157 / 170, 'Logit-lens divergence', ha='center', va='center',
                     fontsize=FONT['title'], weight='bold')
    legend = fig.legend(handles=[Patch(facecolor=BAND, alpha=.12, edgecolor='none', label='Interquartile range'),
                                Patch(facecolor=H_FILL, edgecolor='#E9D6C9', lw=.4, label='H module')],
                        loc='center', bbox_to_anchor=(center, 7 / 170), ncol=2, frameon=False,
                        fontsize=FONT['legend'], handlelength=1.9, columnspacing=1.8, handletextpad=.5)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    title_box = title.get_window_extent(renderer)
    assert all(not title_box.overlaps(t.get_window_extent(renderer)) for t in annotations)
    assert not legend.get_window_extent(renderer).overlaps(ax.xaxis.label.get_window_extent(renderer))
    report = {'sample_count': 32, 'mean_kl': mean.tolist(), 'quartiles': quartiles.tolist(),
              'kl_direction': 'KL(final || intermediate)', 'log_base': 'natural',
              'quantile_method': 'linear', 'H_intervals': [[49, 64], [113, 128]],
              'module_loop_boundaries': module_boundaries, 'model_loop_boundary': model_boundary,
              'figure_size_pt': [WIDTH, 170], 'legend_outside_axes': True,
              'title_does_not_overlap_loop_annotations': True}
    save(fig, 'logit_lens', preview_dir)
    return report
