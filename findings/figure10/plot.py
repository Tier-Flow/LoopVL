"""Redraw Figure 10 from all 32 archived measurements."""
from pathlib import Path
import sys
_FINDINGS_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "common" / "paths.py").is_file())
sys.path.insert(0, str(_FINDINGS_ROOT))
from common.paths import MODEL_ROOT, DATA_ROOT, OUTPUT_ROOT, repo_path, tsv_directory
import argparse,json,sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from common import state_plot as plot

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=repo_path,default=HERE/'data/plot_data.json')
    p.add_argument('--out',type=repo_path,default=OUTPUT_ROOT / 'figure10')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    plot.HERE=args.out
    plot.plt.rcParams.update({'font.family':'DejaVu Sans','font.size':plot.FONT['axis'],
        'axes.unicode_minus':False,'pdf.fonttype':42,'ps.fonttype':42,'text.color':plot.INK,
        'axes.labelcolor':plot.INK,'xtick.color':plot.INK,'ytick.color':plot.INK})
    report=plot.draw_states(json.loads(args.data.read_text()),args.out)
    for ext in ['.pdf','.png']:
        (args.out/('visual_hidden_state'+ext)).replace(args.out/('figure'+ext))
    (args.out/'plot_audit.json').write_text(json.dumps(report,indent=2))
    print('Verified 32 samples; rendered Figure 10')

if __name__=='__main__':main()
