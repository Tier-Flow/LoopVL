"""Capture Figures 6/7/9 statistics in one teacher-forced forward per sample.

The source of the hook is the original 32-sample query-scope experiment.
Only path arguments, audit metadata, and coverage capture were added.
"""
from pathlib import Path
import sys
_FINDINGS_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "common" / "paths.py").is_file())
sys.path.insert(0, str(_FINDINGS_ROOT))
from common.paths import MODEL_ROOT, DATA_ROOT, OUTPUT_ROOT, repo_path, tsv_directory
import argparse,hashlib,importlib.util,json,os,sys,time
import numpy as np

HERE=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-root','--model-dir',dest='model_root',type=repo_path,default=MODEL_ROOT,help='Complete LoopVL snapshot (default: model/).')
    parser.add_argument('--data-root',type=repo_path,default=HERE/'data',help='Parent of samples/manifest.json and samples/images.')
    parser.add_argument('--output',type=repo_path,default=OUTPUT_ROOT / 'figure6/capture',help='A new output directory outside the release tree is recommended.')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--limit',type=int,default=32,help='For a smoke test only; final figure uses all 32 samples.')
    args=parser.parse_args()
    os.environ['LOOPVL_MODEL_ROOT']=str(args.model_root.resolve())
    os.environ['LOOPVL_DIAGNOSTIC_ROOT']=str(args.data_root.resolve())
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    import torch
    torch.set_num_threads(4);torch.manual_seed(20260907)
    spec=importlib.util.spec_from_file_location('diagnostic_common',HERE/'code/diagnostic_common.py')
    common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
    manifest=common.load_manifest()
    samples=sorted(manifest['samples'],key=lambda x:x['id'])
    assert [s['id'] for s in samples]==[f's{i:02d}' for i in range(1,33)]
    assert 1<=args.limit<=32
    samples=samples[:args.limit]
    for sample in samples:
        image=args.data_root/sample['image']
        if not image.is_file():raise FileNotFoundError(f'Missing original diagnostic image: {image}')
    args.output.mkdir(parents=True,exist_ok=True)
    # Preserve existing captures: choose a fresh output directory for a new run.
    assert not any((args.output/f"{s['id']}.npz").exists() for s in samples), 'Existing sample output found.'
    device=torch.device(args.device)
    model,tokenizer,processor,cfg=common.load_runtime(device,eager=True)
    rec=model.recurrent_model
    assert rec.config.H_cycles==2 and rec.config.L_cycles==3
    assert len(rec.L_module.layers)==len(rec.H_module.layers)==16
    head_count=rec.L_module.layers[0].self_attn.config.num_attention_heads
    assert head_count==12
    ctx={}

    def gini(p):
        p=p.float().clamp_min(0);p=p/p.sum(-1,keepdim=True).clamp_min(1e-12)
        values=p.sort(-1).values;n=values.shape[-1]
        ranks=torch.arange(1,n+1,device=p.device,dtype=p.dtype)
        return ((2*ranks-n-1)*values).sum(-1)/n

    def hook_for(kind,physical):
        def hook(module,inputs,kwargs,output):
            key=(kind,physical); call=ctx['counts'].get(key,0);ctx['counts'][key]=call+1
            stack=(call//3)*4+call%3 if kind=='L' else call*4+3
            expanded=stack*16+physical
            weights=output[1]
            assert weights is not None and weights.shape[0]==1
            visual=ctx['indices']['visual'];n=len(visual)
            for name in ('response','combined','instruction'):
                sub=weights[0].index_select(1,ctx['indices'][name]).index_select(2,visual).float()
                # Gini pools RAW query weights first, unlike entropy/coverage.
                pooled=sub.mean(1)
                ctx['metrics'][name+'_gini'][expanded]=gini(pooled)
                ctx['metrics'][name+'_mass'][expanded]=sub.sum(-1).mean(-1)
                if name=='response':
                    p=sub/sub.sum(-1,keepdim=True).clamp_min(1e-12)
                    safe=p.clamp_min(1e-12)
                    ctx['metrics']['response_entropy'][expanded]=(-(safe*safe.log()).sum(-1)/np.log(max(2,n))).mean(-1)
                    sorted_p=p.sort(-1,descending=True).values
                    ctx['metrics']['response_coverage80'][expanded]=((sorted_p.cumsum(-1)<.8).sum(-1).add(1).float()/n).mean(-1)
            ctx['seen'].add(expanded)
        return hook

    handles=[layer.self_attn.register_forward_hook(hook_for(kind,index),with_kwargs=True)
             for kind,module in [('L',rec.L_module),('H',rec.H_module)] for index,layer in enumerate(module.layers)]
    metadata=[]
    try:
        for sample in samples:
            start=time.time()
            with torch.inference_mode():
                batch=common.prepare_teacher_forced(model,tokenizer,processor,cfg,sample,device)
                masks={key:batch['masks'][key] for key in ['response','instruction','visual']}
                masks['combined']=masks['response']|masks['instruction']
                indices={key:mask.nonzero(as_tuple=False).flatten() for key,mask in masks.items()}
                assert all(len(v)>0 for v in indices.values())
                names=[f'{scope}_{metric}' for scope in ['response','combined','instruction'] for metric in ['gini','mass']]
                names+=['response_entropy','response_coverage80']
                ctx.clear();ctx.update(indices=indices,counts={},seen=set(),metrics={key:torch.full((128,12),float('nan'),device=device) for key in names})
                result=rec(inputs_embeds=batch['inputs_embeds'],attention_mask=batch['attention_mask'],
                    token_type_ids=batch['token_type_ids'],position_ids=batch['position_ids'],
                    rope_position_ids=batch['rope_position_ids'],visual_mask=batch['visual_mask'],
                    instruction_mask=batch['instruction_mask'],use_cache=False)
                assert ctx['seen']==set(range(128))
                assert all(count==(6 if kind=='L' else 2) for (kind,_),count in ctx['counts'].items())
                arrays={key:value.cpu().numpy() for key,value in ctx['metrics'].items()}
                assert all(np.isfinite(v).all() for v in arrays.values())
                np.savez_compressed(args.output/f"{sample['id']}.npz",**arrays)
                item={'sample':sample,'head_count':12,'query_counts':{k:len(v) for k,v in indices.items()},
                    'image_sha256':hashlib.sha256((args.data_root/sample['image']).read_bytes()).hexdigest(),
                    'layers':128,'teacher_forced':True,'model_H_cycles':2,'model_L_cycles':3,'seconds':time.time()-start}
                (args.output/f"{sample['id']}.json").write_text(json.dumps(item,indent=2),encoding='utf-8')
                metadata.append(item)
                print(sample['id'],f"{item['seconds']:.2f}s",flush=True)
                del result,batch,arrays;ctx.clear()
    finally:
        for handle in handles:handle.remove()
    protocol={'sample_count':len(samples),'seed':20260907,'teacher_forced':True,
        'model_root_argument':str(args.model_root),'H_cycles':2,'L_cycles':3,'head_count':12,
        'weights_dtype':str(next(model.parameters()).dtype),'torch_version':torch.__version__,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'settings':'normal checkpoint; no anchor, gate, loop-count or decoding intervention',
        'query_scope':{'entropy':'reference-answer','coverage':'reference-answer','gini':'instruction plus reference-answer'},
        'complete':len(metadata)==32}
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')

if __name__=='__main__':main()
