"""Compare/re-export Figures 6/7/9 from a complete 32-sample capture.

The bundled capture predates coverage collection; H coverage is then retained
from the exact original paired snapshot with this fact recorded explicitly.
New capture.py runs compute coverage directly and need no such fallback.
"""
from pathlib import Path
import sys
_FINDINGS_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "common" / "paths.py").is_file())
sys.path.insert(0, str(_FINDINGS_ROOT))
from common.paths import MODEL_ROOT, DATA_ROOT, OUTPUT_ROOT, repo_path, tsv_directory
import argparse,hashlib,json
import numpy as np

HERE=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',type=repo_path,default=HERE/'data/capture32')
    parser.add_argument('--output',type=repo_path,default=OUTPUT_ROOT / 'figure6/export')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    ids=[f's{i:02d}' for i in range(1,33)]
    means={};hashes=[]
    for sid in ids:
        path=args.capture/f'{sid}.npz'
        with np.load(path,allow_pickle=False) as data:
            for field in ('response_entropy','combined_gini','response_coverage80'):
                if field not in data:continue
                v=np.asarray(data[field],dtype=np.float64)
                assert v.shape==(128,12) and np.isfinite(v).all()
                means.setdefault(field,[]).append(v.mean(-1))
        hashes.append({'sample':sid,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    means={key:np.stack(values) for key,values in means.items()}
    assert all(v.shape==(32,128) for v in means.values())
    audit={'sample_count':32,'head_count':12,'sources':hashes,'mean_max_abs_difference':{}}
    for number,field in [(6,'response_entropy'),(7,'combined_gini')]:
        saved=json.loads((HERE.parent/f'figure{number}/data/plot_data.json').read_text())
        audit['mean_max_abs_difference'][f'figure{number}']=float(np.abs(means[field].mean(0)-saved['mean']).max())
        saved['mean']=means[field].mean(0).tolist();saved['per_sample_rerun']=means[field].tolist()
        saved['paper_source']='recomputed from capture supplied to export_capture.py'
        (args.output/f'figure{number}_plot_data.json').write_text(json.dumps(saved,indent=2),encoding='utf-8')
    endpoint=json.loads((HERE.parent/'figure9/data/lh_endpoint_data.json').read_text())
    for name,indices in [('L1',[15,79]),('L2',[31,95]),('L3',[47,111])]:
        pairs=means['response_entropy'][:,indices]
        endpoint['L'][name].update(pairs=pairs.tolist(),means=pairs.mean(0).tolist(),
            samples_decreasing=int((pairs[:,1]<pairs[:,0]).sum()))
    for metric,field in [('response_entropy','response_entropy'),('sink_gini','combined_gini'),('response_coverage80','response_coverage80')]:
        if field not in means:
            audit['coverage_source']='original saved H paired snapshot (coverage absent from supplied NPZ)'
            continue
        pairs=means[field][:,[63,127]]
        audit['mean_max_abs_difference'][f'figure9_{metric}']=float(np.abs(pairs-np.asarray(endpoint['H'][metric]['pairs'])).max())
        endpoint['H'][metric].update(pairs=pairs.tolist(),means=pairs.mean(0).tolist())
    endpoint['source_files']=hashes
    endpoint['source_root']='capture supplied to export_capture.py'
    (args.output/'figure9_endpoint_data.json').write_text(json.dumps(endpoint,indent=2),encoding='utf-8')
    (args.output/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps({key:value for key,value in audit.items() if key!='sources'},indent=2))

if __name__=='__main__':main()
