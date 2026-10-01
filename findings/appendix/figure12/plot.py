"""Redraw the eight selected cases from saved endpoint attention.

python plot.py --style all --out rendered
"""
from pathlib import Path
import sys
_FINDINGS_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "common" / "paths.py").is_file())
sys.path.insert(0, str(_FINDINGS_ROOT))
from common.paths import MODEL_ROOT, DATA_ROOT, OUTPUT_ROOT, repo_path, tsv_directory
import argparse
import json
import sys

HERE = Path(__file__).resolve().parent
CODE = HERE / 'code' if (HERE / 'code').is_dir() else HERE.parent / 'code'
sys.path.insert(0, str(CODE))
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import paper_reallocation_plot_20260912 as spatial
import plot_reallocation_combined_20260914 as layout


def load_cases(root):
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    cases, records = {}, {}
    with np.load(root / 'attention.npz', allow_pickle=False) as arrays:
        for item in manifest['cases']:
            sid = item['id']
            image_path = root / item['sample']['image']
            if spatial.sha256(image_path) != item['image_sha256']:
                raise ValueError(f'Source image checksum differs: {sid}')
            with Image.open(image_path) as im:
                image = np.asarray(im.convert('RGB'))
            a, b, target = (arrays[sid + suffix] for suffix in ['_first', '_second', '_target'])
            if not (np.isclose(a.sum(), 1) and np.isclose(b.sum(), 1)):
                raise ValueError(f'Attention must sum to one on the full grid: {sid}')
            c = spatial.Case(sid, item['sample'], a, b, target.astype(bool),
                             *item['grid'], image, image_path, root/'attention.npz', item['source'])
            assert np.array_equal(c.promoted, arrays[sid+'_promoted']), sid
            assert c.promoted.sum() == item['promoted_count'], sid
            assert (c.promoted & c.target).sum() == item['target_promoted_count'], sid
            assert np.isclose(a[target].sum(), item['target_attention_share_first']), sid
            assert np.isclose(b[target].sum(), item['target_attention_share_second']), sid
            cases[sid] = c
            records[sid] = spatial.case_metrics(c)
            layout.HEADERS[sid] = (item['display_question'], item['display_answer'])
    return manifest, cases, records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=repo_path, default=HERE/'data')
    p.add_argument('--out', type=repo_path, default=OUTPUT_ROOT / 'appendix/figure12')
    p.add_argument('--dpi', type=int, default=300)
    p.add_argument('--style', choices=['token_grid','smooth','all'], default='token_grid')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest, cases, records = load_cases(args.data)
    fill = {x['id'] for x in manifest['cases'] if x['display_fill_panel']}
    # A display-only resize. Preserve descending y coordinates (image top first).
    original_attention, original_promoted = spatial.attention_map, spatial.promoted_map
    def set_frame(ax, case):
        if case.sid in fill:
            if not case.sample.get('display_crop'):
                x0, x1, bottom, top = case.extent
                ax.set_xlim(x0, x1)
                ax.set_ylim(bottom, top)
            ax.set_aspect('auto')
    def attention(ax, case, *values, **kwargs):
        original_attention(ax, case, *values, **kwargs)
        set_frame(ax, case)
    def promoted(ax, case):
        original_promoted(ax, case)
        set_frame(ax, case)
    spatial.attention_map, spatial.promoted_map = attention, promoted
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':6,
                         'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
    results=[]
    for style in (['token_grid','smooth'] if args.style == 'all' else [args.style]):
        result=layout.render(cases, records, list(cases), args.out, args.dpi,
                             three_rows=True, explain_promotion=True,
                             map_style=style, scale_quantile=.995, name_suffix='_'+style)
        for suffix in ['.png','.pdf','.svg','_preview.png']:
            source=args.out/(result['name']+suffix)
            source.replace(args.out/('figure_'+style+suffix))
        result['name']='figure_'+style
        results.append(result)
        print(f'Redrawn figure {manifest["figure"]}: {style}, {len(cases)} verified cases')
    (args.out/'render_audit.json').write_text(json.dumps(results,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
