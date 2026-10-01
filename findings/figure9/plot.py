"""Draw paired L/H endpoints at the paper's actual text width.

All three L panels use normalized spatial entropy at the last physical layer
of each complete L invocation. This replaces the earlier L attention-mass
heatmaps, not a relabeling of those values. No model inference is run here.
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
from matplotlib.lines import Line2D
from matplotlib.text import Text
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
WIDTH = 470.268
FONT = {'heading': 8.0, 'panel': 7.0, 'axis': 7.0, 'tick': 6.5}
INK, EDGE, GRID = '#3F5060', '#8798A9', '#DFE5ED'
BLUE, GRAY = '#4F78BB', '#AAB5C1'


def draw(data, entropy_only, output, preview):
    assert data['sample_count'] == len(data['sample_order']) == 32
    assert len(set(data['sample_order'])) == 32
    height = 112
    fig = plt.figure(figsize=(WIDTH / 72, height / 72), dpi=160, facecolor='white')
    panels = [(name, 'Spatial entropy', data['L'][name]['pairs']) for name in ('L1', 'L2', 'L3')]
    panels += [('H', 'Spatial entropy', data['H']['response_entropy']['pairs'])]
    if not entropy_only:
        panels += [('H', 'Attention coverage', data['H']['response_coverage80']['pairs']),
                   ('H', 'Gini concentration', data['H']['sink_gini']['pairs'])]
    if entropy_only:
        xs, widths = [34, 144, 254, 369], [88] * 4
        fig.text(.515, .945, 'Spatial entropy across model cycles', ha='center',
                 va='center', fontsize=FONT['heading'], weight='bold')
    else:
        xs, widths = [24, 98, 172, 250, 324, 398], [60] * 6
        fig.text(24 / WIDTH, .955, '(a) L module', fontsize=FONT['heading'], weight='bold', va='center')
        fig.text(250 / WIDTH, .955, '(b) H module', fontsize=FONT['heading'], weight='bold', va='center')
    axes, audit = [], []
    for j, ((module, metric, values), x, width) in enumerate(zip(panels, xs, widths)):
        values = np.asarray(values, dtype=float)
        assert values.shape == (32, 2) and np.isfinite(values).all()
        assert ((0 <= values) & (values <= 1)).all()
        ax = fig.add_axes([x / WIDTH, 35 / height, width / WIDTH, 54 / height])
        axes.append(ax)
        for row in values:
            line, = ax.plot([1, 2], row, color=GRAY, alpha=.52, lw=.24, zorder=2)
            np.testing.assert_array_equal(line.get_ydata(), row)
        mean = values.mean(axis=0)
        line, = ax.plot([1, 2], mean, '-o', color=BLUE, lw=.68, ms=1.8, zorder=3)
        np.testing.assert_array_equal(line.get_ydata(), mean)
        assert len(ax.lines) == len(values) + 1
        ax.set_xlim(.8, 2.2)
        ax.set_xticks([1, 2])
        if metric == 'Spatial entropy':
            ax.set_ylim(.3, 1.02)
            ax.set_yticks([.4, .6, .8, 1.0])
        elif metric == 'Attention coverage':
            ax.set_ylim(0, .52)
            ax.set_yticks([0, .2, .4])
        else:
            ax.set_ylim(.3, .86)
            ax.set_yticks([.4, .6, .8])
        low, high = ax.get_ylim()
        assert values.min() >= low and values.max() <= high, (module, metric, values.min(), values.max())
        ax.tick_params(labelsize=FONT['tick'], colors=INK, length=1.5, width=.4, pad=1.5)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(EDGE)
            spine.set_linewidth(.4)
        ax.set_axisbelow(True)
        ax.grid(axis='y', color=GRID, linewidth=.3)
        title = module if entropy_only or j < 3 else metric
        ax.set_title(title, fontsize=FONT['panel'], pad=4)
        if j == 0:
            ax.set_ylabel('Spatial entropy', fontsize=FONT['axis'], labelpad=2)
        audit.append({'module': module, 'metric': metric, 'mean': mean.tolist(),
                      'paired_samples': len(values), 'sample_lines': len(ax.lines) - 1,
                      'displayed_sample_count': len(values),
                      'mean_sample_count': len(values),
                      'y_limits': [low, high], 'axes_bounds': list(ax.get_position().bounds)})
    fig.text(.515, 19 / height, 'Model cycle', ha='center', va='center', fontsize=FONT['axis'])
    fig.legend(handles=[Line2D([0], [0], color=GRAY, lw=.3, label='Samples'),
                        Line2D([0], [0], color=BLUE, lw=.68, marker='o', ms=1.8,
                               label='Mean')],
               loc='center', bbox_to_anchor=(.515, 6 / height), ncol=2, frameon=False,
               fontsize=FONT['tick'], handlelength=2, columnspacing=1.8, handletextpad=.5)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for artist in fig.findobj(Text):
        if artist.get_visible() and artist.get_text():
            box = artist.get_window_extent(renderer)
            assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= fig.bbox.width and box.y1 <= fig.bbox.height, (artist.get_text(), box.bounds)
    fig.savefig(output, facecolor='white', metadata={'Title': 'L/H visual attention at paired module endpoints'})
    if preview:
        fig.savefig(preview, dpi=240, facecolor='white')
    plt.close(fig)
    return {'paper_width_pt': WIDTH, 'font_sizes_pt': FONT, 'entropy_only': entropy_only,
            'sample_order': list(data['sample_order']),
            'panels': audit, 'inference_rerun': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=repo_path, default=DATA / 'lh_endpoint_data.json')
    parser.add_argument('--entropy-only', action='store_true')
    parser.add_argument('--output', type=repo_path, default=OUTPUT_ROOT / 'figure9/figure.pdf')
    parser.add_argument('--preview', type=repo_path)
    parser.add_argument('--audit', type=repo_path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.preview is None:
        args.preview = args.output.with_suffix('.png')
    if args.audit is None:
        args.audit = args.output.with_suffix('.audit.json')
    data = json.loads(args.data.read_text(encoding='utf-8'))
    assert data['sample_count'] == 32
    from matplotlib import font_manager
    font = 'Arial' if any(f.name == 'Arial' for f in font_manager.fontManager.ttflist) else 'DejaVu Sans'
    plt.rcParams.update({'font.family': font, 'font.size': FONT['axis'],
                         'axes.unicode_minus': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'text.color': INK, 'axes.labelcolor': INK})
    audit = draw(data, args.entropy_only, args.output, args.preview)
    audit['source_sha256'] = hashlib.sha256(args.data.read_bytes()).hexdigest()
    if args.audit:
        args.audit.write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
