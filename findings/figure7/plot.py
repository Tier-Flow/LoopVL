"""Reproduce Figure 7 with the shared concentration-trajectory plotter."""
from pathlib import Path
import sys
_FINDINGS_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "common" / "paths.py").is_file())
sys.path.insert(0, str(_FINDINGS_ROOT))
from common.paths import MODEL_ROOT, DATA_ROOT, OUTPUT_ROOT, repo_path, tsv_directory
import argparse,importlib.util
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('trajectory_plot',HERE.parent/'figure6/plot.py')
plot=importlib.util.module_from_spec(spec);spec.loader.exec_module(plot)
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=repo_path,default=HERE/'data/plot_data.json')
    parser.add_argument('--output-dir',type=repo_path,default=OUTPUT_ROOT / 'figure7')
    args=parser.parse_args();plot.draw(args.data,args.output_dir)
