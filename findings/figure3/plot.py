"""Render four benchmark panels for three controls of the main H2L3 model.

This plots supplied archived or freshly collected scores; it does not run inference or assert
that the older Frozen hook matches the current full-benchmark protocol.
"""
from pathlib import Path
import sys
_FINDINGS_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "common" / "paths.py").is_file())
sys.path.insert(0, str(_FINDINGS_ROOT))
from common.paths import MODEL_ROOT, DATA_ROOT, OUTPUT_ROOT, repo_path, tsv_directory
import argparse
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.text import Text
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data' / 'scores.json'
PRINT_WIDTH_PT, HEIGHT_PT = 470.268, 172
FONT = {'title': 10.5, 'panel': 9.0, 'label': 8.5, 'legend': 7.5, 'tick': 7.0}
LABELS = ['Normal', 'Block L visual updates', 'Freeze visual states']
INK, EDGE, GRID = '#3F5060', '#8798A9', '#DFE5ED'
BLUE = '#4F78BB'
COLORS = ['#A8C3E5', '#D3E3F5', '#EFF5FC']
BAR_WIDTH = .44
# Reserve only the y-label/ticks at left; the last spine meets the print edge.
PLOT_LEFT_PT, PLOT_RIGHT_PT, PANEL_GAP_PT = 27.875, PRINT_WIDTH_PT - .725, 19
PANEL_WIDTH_PT = (PLOT_RIGHT_PT - PLOT_LEFT_PT - 3 * PANEL_GAP_PT) / 4
ROW_CENTER = (PLOT_LEFT_PT + PLOT_RIGHT_PT) / (2 * PRINT_WIDTH_PT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pdf', type=repo_path, default=OUTPUT_ROOT / 'figure3/figure.pdf')
    parser.add_argument('--preview-dir', type=repo_path, default=OUTPUT_ROOT / 'figure3')
    parser.add_argument('--data', type=repo_path, default=DATA)
    args = parser.parse_args()
    data = json.loads(args.data.read_text(encoding='utf-8'))
    fresh = data.get('results_not_recomputed') is False
    assert data['condition_order'] == ['normal', 'blocked', 'frozen']
    names = data['selected_benchmarks']
    assert len(names) == 4 and len(set(names)) == 4
    all_values = [value for name in names for value in data['benchmarks'][name]['scores_percent']]
    axis_max = 80 if max(all_values) <= 80 else 100
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': FONT['label'],
                         'axes.unicode_minus': False, 'pdf.fonttype': 42,
                         'ps.fonttype': 42, 'svg.fonttype': 'none',
                         'text.color': INK, 'axes.labelcolor': INK,
                         'xtick.color': INK, 'ytick.color': INK})
    fig = plt.figure(figsize=(PRINT_WIDTH_PT / 72, HEIGHT_PT / 72), dpi=180, facecolor='white')
    fig.text(ROW_CENTER, 165 / HEIGHT_PT, 'Visual-state updates', ha='center', va='center',
             fontsize=FONT['title'], weight='bold')
    boxes = [[(PLOT_LEFT_PT + j * (PANEL_WIDTH_PT + PANEL_GAP_PT)) / PRINT_WIDTH_PT,
              42 / HEIGHT_PT, PANEL_WIDTH_PT / PRINT_WIDTH_PT, 96 / HEIGHT_PT]
             for j in range(4)]
    audit = []
    for j, (name, box) in enumerate(zip(names, boxes)):
        record = data['benchmarks'][name]
        values = np.asarray(record['scores_percent'], dtype=float)
        assert values.shape == (3,) and np.isfinite(values).all()
        assert ((0 <= values) & (values <= 100)).all()
        ax = fig.add_axes(box)
        x = np.arange(3)
        bars = ax.bar(x, values, width=BAR_WIDTH, color=COLORS,
                      edgecolor=BLUE, linewidth=.65, zorder=3)
        np.testing.assert_array_equal([bar.get_height() for bar in bars], values)
        ax.set_xlim(-.55, 2.55)
        ax.set_ylim(0, axis_max)
        ax.set_xticks([])
        ax.set_yticks(list(range(0, axis_max + 1, 20)))
        ax.tick_params(axis='y', labelsize=FONT['tick'], length=2, width=.4, pad=3)
        if j == 0:
            ax.set_ylabel('Accuracy (%)', fontsize=FONT['label'], labelpad=5)
        ax.set_title(f'({chr(97+j)}) {name}', fontsize=FONT['panel'], pad=5)
        ax.set_axisbelow(True)
        ax.grid(axis='y', color=GRID, linewidth=.35)
        for spine in ax.spines.values():
            spine.set_color(EDGE)
            spine.set_linewidth(.45)
        assert not ax.texts
        audit.append({'benchmark': name, 'scores_percent': values.tolist(),
                      'bar_count': len(bars), 'bar_width': BAR_WIDTH,
                      'y_limits': list(ax.get_ylim()), 'axes_bounds': box})
    fig.legend(handles=[Patch(facecolor=color, edgecolor=BLUE, linewidth=.65, label=label)
                        for color, label in zip(COLORS, LABELS)],
               loc='center', bbox_to_anchor=(ROW_CENTER, 17 / HEIGHT_PT), ncol=3,
               frameon=False, fontsize=FONT['legend'], handlelength=1.6,
               handleheight=.8, handletextpad=.55, columnspacing=1.6)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for artist in fig.findobj(Text):
        if artist.get_visible() and artist.get_text():
            box = artist.get_window_extent(renderer)
            assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= fig.bbox.width and box.y1 <= fig.bbox.height, (artist.get_text(), box.bounds)
    args.pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.pdf, facecolor='white', metadata={
        'Title': 'LoopVL visual-state controls across four benchmarks',
        'Subject': ('Fresh full-data control evaluation' if fresh else 'Author-supplied archived scores') + ' for the main H2L3 checkpoint'})
    if args.preview_dir:
        args.preview_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.preview_dir / 'figure.png', dpi=240, facecolor='white')
        fig.savefig(args.preview_dir / 'figure.svg', facecolor='white')
        report = {'source_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
                  'conditions': data['condition_order'], 'labels': LABELS,
                  'panels': audit, 'bar_colors': COLORS, 'font_sizes_pt': FONT,
                  'figure_size_pt': [PRINT_WIDTH_PT, HEIGHT_PT], 'font_family': 'DejaVu Sans',
                  'layout': 'one row, four panels', 'shared_legend': True,
                  'horizontal_alignment': 'y-label and last spine at manuscript edges',
                  'panel_gap_pt': PANEL_GAP_PT,
                  'visible_x_tick_labels': False,
                  'visible_result_annotations': False, 'inference_rerun': fresh,
                  'legacy_400_sample_scores_used': False}
        (args.preview_dir / 'plot_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    plt.close(fig)
    print(f'Saved {args.pdf}: 4 panels, 12 {"fresh" if fresh else "archived"} scores, shared 0-{axis_max} axis.')


if __name__ == '__main__':
    main()
