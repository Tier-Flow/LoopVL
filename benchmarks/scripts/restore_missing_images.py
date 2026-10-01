#!/usr/bin/env python3
"""Restore selected known omitted cache images from archived source data."""
import argparse
import ast
import base64
import csv
import hashlib
import json
import re
import sys
from io import BytesIO
from pathlib import Path
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-root',type=Path,default=ROOT.parent/'data')
    parser.add_argument('--datasets',nargs='+',choices=['ChartQA_TEST','MMStar','MMMU-Pro'],
                        default=['ChartQA_TEST','MMStar','MMMU-Pro'])
    args=parser.parse_args()
    selected=set(args.datasets)
    args.asset_root=(args.asset_root if args.asset_root.is_absolute() else ROOT.parent/args.asset_root).resolve()
    snapshot=args.asset_root/'server_snapshot_20260906'
    results=[]
    csv.field_size_limit(2**31-1)
    for name,indices in [('ChartQA_TEST',{'443','444'}),('MMStar',{'159'})]:
        if name not in selected: continue
        with (snapshot/'VLMEvalData'/f'{name}.tsv').open(encoding='utf-8') as f:
            for row in csv.DictReader(f,delimiter='\t'):
                if row['index'] not in indices: continue
                dst=snapshot/'VLMEvalData'/'images'/name/(row['index']+'.png')
                if not dst.exists():
                    raw=base64.b64decode(row['image'])
                    with Image.open(BytesIO(raw)) as image: image.verify()
                    dst.parent.mkdir(parents=True,exist_ok=True)
                    dst.write_bytes(raw)
                results.append({'path':str(dst.relative_to(args.asset_root)),'sha256':hashlib.sha256(dst.read_bytes()).hexdigest(),'source':f'VLMEvalData/{name}.tsv base64'})
    if 'MMMU-Pro' in selected:
        from datasets import load_from_disk
        dataset=load_from_disk(str(snapshot/'HFDatasets'/'MMMU-Pro'))
        # Reuse only the image materialization functions, without GPU imports.
        source=ROOT/'original_scripts'/'run_hrm_hf_dataset.py'
        tree=ast.parse(source.read_text(encoding='utf-8'))
        functions=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in ('collect_images','row_images')]
        namespace={'Image':Image,'Path':Path,'BytesIO':BytesIO,'base64':base64,'re':re}
        exec(compile(ast.Module(body=functions,type_ignores=[]),str(source),'exec'),namespace)
        for position in [118,156,333,1227]:
            dst=snapshot/'hrm_bench_results'/'v2'/'MMMU-Pro'/'images'/f'{position:06d}_1.jpg'
            if not dst.exists():
                images=namespace['row_images'](dict(dataset[position]))
                if not images: raise RuntimeError(f'No MMMU image at {position}')
                dst.parent.mkdir(parents=True,exist_ok=True)
                images[0].save(dst,quality=95)
            with Image.open(dst) as image: image.verify()
            results.append({'path':str(dst.relative_to(args.asset_root)),'sha256':hashlib.sha256(dst.read_bytes()).hexdigest(),'source':f'HFDatasets/MMMU-Pro row {position}, archived RGB JPEG quality95'})
    report_name='restored_images.json' if len(selected)==3 else 'restored_images_'+'_'.join(sorted(selected))+'.json'
    report=args.asset_root/report_name
    expected={item['path']:item['sha256'] for item in json.loads((ROOT/'provenance/restored_images.json').read_text(encoding='utf-8'))}
    count=sum({'ChartQA_TEST':2,'MMStar':1,'MMMU-Pro':4}[name] for name in selected)
    if len(results)!=count or any(item['sha256']!=expected.get(item['path']) for item in results):
        raise RuntimeError('Restored images differ from pinned hashes of the completed GPU rerun')
    report.write_text(json.dumps(results,indent=2)+'\n')
    print('RESTORED_VERIFIED',len(results),str(report))

if __name__=='__main__': main()
