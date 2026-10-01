"""Recompute Figures 10/11 on their exact 16 VMCBench + 16 AI2D examples."""
from pathlib import Path
import sys
_FINDINGS_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "common" / "paths.py").is_file())
sys.path.insert(0, str(_FINDINGS_ROOT))
from common.paths import MODEL_ROOT, DATA_ROOT, OUTPUT_ROOT, repo_path, tsv_directory
import argparse,json,sys
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',type=repo_path,default=MODEL_ROOT)
    p.add_argument('--data-dir',type=repo_path,default=DATA_ROOT,help='VMCBench_DEV.tsv and AI2D_TEST.tsv')
    p.add_argument('--output',type=repo_path,default=OUTPUT_ROOT / 'states')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--probe',choices=['states','logits','both'],default='both')
    p.add_argument('--limit',type=int,default=32,help='Use 32 for the complete diagnostic; smaller values are smoke tests')
    args=p.parse_args()
    args.data_dir = tsv_directory(args.data_dir)
    import numpy as np
    import torch
    from common.runtime import Runtime
    from common import state_metrics as states,logit_metrics as logits
    torch.manual_seed(20260907);torch.set_num_threads(4)
    refs={n:json.loads((REPO/f'figure{n}/data/plot_data.json').read_text()) for n in (10,11)}
    wanted=[(r['dataset'],r['position']) for r in refs[10]['state_samples']][:args.limit]
    if not 1<=args.limit<=32:p.error('--limit must be 1..32')
    states.SOURCES={'VMCBench':(args.data_dir/'VMCBench_DEV.tsv',1000),'AI2D':(args.data_dir/'AI2D_TEST.tsv',3088)}
    rows=states.load_selected_rows(wanted)
    runtime=Runtime(model_dir=args.model_dir,device=args.device,attention='sdpa')
    runtime.configure(2,3)
    args.output.mkdir(parents=True,exist_ok=True)
    result_path=args.output/'samples.json'
    if result_path.exists():raise FileExistsError('Choose a new output directory')
    results={10:[],11:[]};audit=[]
    for ordinal,(dataset,position) in enumerate(wanted):
        row=rows[(dataset,position)]
        expected=refs[10]['state_samples'][ordinal]
        assert str(row.get('index',position))==str(expected['index'])
        image=states.decode_image(row['image']);prompt=states.prompt_for(row)
        identity={'dataset':dataset,'position':position,'index':row.get('index',str(position))}
        item=dict(identity)
        if args.probe in ('states','both'):
            values=states.analyse(runtime,image,prompt)
            results[10].append({**identity,**values})
            item['max_delta_difference']=float(np.max(np.abs(np.asarray(values['delta_l2_visual'])-expected['delta_l2_visual'])))
        if args.probe in ('logits','both'):
            values=logits.logit_lens(runtime,image,prompt)
            results[11].append({**identity,**values})
            item['max_kl_difference']=float(np.max(np.abs(np.asarray(values['kl_final_to_intermediate'])-refs[11]['logit_samples'][ordinal]['kl_final_to_intermediate'])))
            item['final_logit_cosine']=values['captured_final_vs_actual_logit_cosine']
        audit.append(item)
        print(json.dumps(item),flush=True)
        result_path.write_text(json.dumps(results,indent=2))
    if args.limit==32:
        for n,key in [(10,'state_samples'),(11,'logit_samples')]:
            if not results[n]:continue
            data=refs[n];data[key]=results[n]
            data['sources']={'description':'Fresh capture from the supplied checkpoint and hash-pinned benchmark rows'}
            if n==10:
                data['reference']['mean_delta_visual']=np.mean([r['delta_l2_visual'] for r in results[n]],axis=0).tolist()
                data['reference']['cosine_visual']=np.mean([r['cosine_visual'] for r in results[n]],axis=0).tolist()
            else:
                x=np.asarray([r['kl_final_to_intermediate'] for r in results[n]])
                data['reference']['mean_kl']=x.mean(0).tolist()
                data['reference']['quartiles']=np.quantile(x,[.25,.75],axis=0,method='linear').tolist()
            (args.output/f'figure{n}_plot_data.json').write_text(json.dumps(data,indent=2))
    (args.output/'audit.json').write_text(json.dumps({'sample_count':len(wanted),'complete':args.limit==32,'comparisons':audit},indent=2))


if __name__=='__main__':main()
