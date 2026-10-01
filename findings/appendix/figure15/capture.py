"""Recapture the published teacher-forced reference-answer endpoint attention.

Uses the unchanged H2L3 checkpoint via ../common/runtime.py. Reference answers
are deliberately supplied: these maps are not free-generation attention.
Outputs go outside data/ by default to preserve the archived measurements.
"""
from pathlib import Path
import sys
_FINDINGS_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "common" / "paths.py").is_file())
sys.path.insert(0, str(_FINDINGS_ROOT))
from common.paths import MODEL_ROOT, DATA_ROOT, OUTPUT_ROOT, repo_path, tsv_directory
import argparse
import json
import copy
import shutil
import sys

HERE=Path(__file__).resolve().parent
REPO=next(p for p in HERE.parents if (p/'common/runtime.py').is_file())
sys.path.insert(0,str(REPO))


def prepare(runtime, sample, image_path):
    import torch
    from PIL import Image
    from hrm_penguin.inference import build_inference_batch
    from hrm_penguin.recurrent.positions import build_compact_positions
    model,tok,proc,cfg=runtime.model,runtime.tok,runtime.proc,runtime.cfg
    device=runtime.device
    with Image.open(image_path) as im:
        batch=build_inference_batch(tok,proc,im.convert('RGB'),sample['prompt'],cfg,device)
    vp=next(model.vision_encoder.parameters())
    pixels=batch['pixel_values'].to(device=vp.device,dtype=vp.dtype)
    vision=model.vision_encoder(pixels,batch['grid_sizes'].to(vp.device),batch['merge_sizes'].to(vp.device))
    projected=model.projector(vision)
    answer_ids=tok(sample['answer'],add_special_tokens=False)['input_ids']
    if not answer_ids:answer_ids=[tok.eos_token_id]
    answers=torch.tensor([answer_ids],dtype=torch.long,device=device)
    ids=torch.cat((batch['input_ids'],answers),dim=1)
    header=int(batch['header_length']); instruction=int(batch['instruction_length'])
    visual_count=int(batch['visual_token_counts'][0]);prefix=header+instruction+visual_count
    positions,rope,visual_mask,instruction_mask=build_compact_positions(
        torch.tensor([header],device=device),torch.tensor([instruction],device=device),
        torch.tensor([len(answer_ids)],device=device),
        visual_token_counts=torch.tensor([visual_count],device=device),
        grid_heights=batch['grid_heights'],grid_widths=batch['grid_widths'],
        grid_h=cfg.vision.grid_height,grid_w=cfg.vision.grid_width,
        rope_mode=cfg.recurrent_visual.rope_mode)
    embeds=model._replace_visual_tokens(model.language_model.get_input_embeddings()(ids),projected,visual_mask)
    types=torch.zeros_like(ids);types[:,:prefix]=1
    kwargs=dict(inputs_embeds=embeds,attention_mask=torch.ones_like(ids),token_type_ids=types,
                position_ids=positions,rope_position_ids=rope,visual_mask=visual_mask,
                instruction_mask=instruction_mask,use_cache=False)
    return kwargs,visual_mask[0].nonzero(as_tuple=False).flatten(),(types[0]==0).nonzero(as_tuple=False).flatten()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',type=repo_path,default=MODEL_ROOT)
    p.add_argument('--data',type=repo_path,default=HERE/'data')
    p.add_argument('--out',type=repo_path,default=OUTPUT_ROOT / 'appendix/figure15/capture')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--ids',default='',help='Optional comma-separated manifest IDs; default all eight')
    args=p.parse_args()
    import numpy as np
    import torch
    from common.runtime import Runtime
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'attention.npz').exists():
        raise FileExistsError('Choose a new output directory; previous captures are preserved')
    torch.manual_seed(20260907)
    torch.set_num_threads(4)
    runtime=Runtime(device=args.device,model_dir=args.model_dir,attention='eager')
    model=runtime.model;rec=model.recurrent_model
    assert rec.config.H_cycles==2 and rec.config.L_cycles==3
    assert len(rec.H_module.layers)==16
    manifest=json.loads((args.data/'manifest.json').read_text(encoding='utf-8'))
    archive=np.load(args.data/'attention.npz',allow_pickle=False)
    selected=set(args.ids.split(',')) if args.ids else None
    context={};reports=[];all_arrays={}
    def hook(module,inputs,kwargs,result):
        weights=result[1]
        if weights is None:
            raise RuntimeError('Attention weights unavailable: configure eager attention in common/runtime.py.')
        attention=weights[0].index_select(1,context['query']).index_select(2,context['visual'])
        context['maps'].append(attention.float().mean(1).cpu().numpy())
    handle=rec.H_module.layers[15].self_attn.register_forward_hook(hook,with_kwargs=True)
    try:
        for item in manifest['cases']:
            sid=item['id']
            if selected is not None and sid not in selected:continue
            with torch.inference_mode():
                kwargs,visual,query=prepare(runtime,item['sample'],args.data/item['sample']['image'])
                context.update(visual=visual,query=query,maps=[])
                result=rec(**kwargs)
            assert len(context['maps'])==2,sid
            report={'id':sid,'max_abs_error':[],'promotion_agreement':None}
            normalized=[]
            for k,raw in enumerate(context['maps']):
                # Original averaging precision varies in early synthetic exports;
                # numeric tolerance is reported rather than claimed bitwise exact.
                a=raw.astype(np.float64).mean(0);a/=a.sum();normalized.append(a)
                label=['first','second'][k]
                all_arrays[sid+'_'+label]=a;all_arrays[sid+'_raw_'+label]=raw
                report['max_abs_error'].append(float(abs(a-archive[sid+'_'+label]).max()))
            a,b=normalized;promotion=(a<=np.quantile(a,.5))&(b>=np.quantile(b,.75))
            all_arrays[sid+'_promoted']=promotion;all_arrays[sid+'_target']=archive[sid+'_target']
            report['promotion_agreement']=float(np.mean(promotion==archive[sid+'_promoted']))
            old=archive[sid+'_promoted'].astype(bool)
            report['promoted_set_jaccard']=float((promotion & old).sum()/max(1,(promotion | old).sum()))
            report['promoted_count_original']=int(old.sum())
            report['promoted_count_recaptured']=int(promotion.sum())
            reports.append(report)
            print(json.dumps(report),flush=True)
            del result,kwargs
    finally:
        handle.remove()
    np.savez_compressed(args.out/'attention.npz',**all_arrays)
    (args.out/'comparison.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
    # A separate ready-to-plot dataset; never overwrite the archived paper data.
    fresh=copy.deepcopy(manifest)
    fresh['cases']=[item for item in fresh['cases'] if item['id']+'_first' in all_arrays]
    fresh['measurement_source']='Fresh H2L3 capture; comparison.json compares to the archived paper arrays'
    output_data=args.out/'data'
    (output_data/'images').mkdir(parents=True,exist_ok=True)
    for item in fresh['cases']:
        sid=item['id'];a=all_arrays[sid+'_first'];b=all_arrays[sid+'_second']
        t=all_arrays[sid+'_target'].astype(bool);promoted=all_arrays[sid+'_promoted']
        item['target_attention_share_first']=float(a[t].sum())
        item['target_attention_share_second']=float(b[t].sum())
        item['promoted_count']=int(promoted.sum());item['target_promoted_count']=int((promoted&t).sum())
        item['source']['raw_visual_attention_total']=[
            float(all_arrays[sid+'_raw_'+label].sum(-1).mean()) for label in ('first','second')]
        item['source']['measurement_source']='Fresh capture; original comparison is in ../comparison.json'
        shutil.copy2(args.data/item['sample']['image'],output_data/item['sample']['image'])
    np.savez_compressed(output_data/'attention.npz',**all_arrays)
    (output_data/'manifest.json').write_text(json.dumps(fresh,indent=2),encoding='utf-8')


if __name__=='__main__':main()
