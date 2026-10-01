"""Verify bundled evidence and redraw figures under outputs/findings/replot."""
from pathlib import Path
import sys
_FINDINGS_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "common" / "paths.py").is_file())
sys.path.insert(0, str(_FINDINGS_ROOT))
from common.paths import MODEL_ROOT, DATA_ROOT, OUTPUT_ROOT, repo_path, tsv_directory
import argparse,hashlib,json,subprocess,sys
import numpy as np

REPO=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plots',action='store_true')
    p.add_argument('--output',type=repo_path,default=OUTPUT_ROOT / 'replot')
    args=p.parse_args();output=args.output.resolve()
    if output==REPO or REPO in output.parents:p.error('Choose an output outside findings/, such as outputs/findings/replot')
    output.mkdir(parents=True,exist_ok=True)
    assertions=[]
    for source in REPO.rglob('*.py'):
        compile(source.read_text(encoding='utf-8-sig'),str(source),'exec')
    assertions.append('All Python sources compile')
    scores=json.loads((REPO/'figure3/data/scores.json').read_text())
    assert scores['condition_order']==['normal','blocked','frozen']
    assert all(len(v['scores_percent'])==3 for v in scores['benchmarks'].values())
    assertions.append('Figure 3 contains only main-model inference controls')
    for n in (6,7):
        data=json.loads((REPO/f'figure{n}/data/plot_data.json').read_text())
        assert data['sample_count']==32
        assert np.asarray(data['per_sample_rerun']).shape==(32,128)
        assert np.asarray(data['mean']).shape==(128,)
    endpoint=json.loads((REPO/'figure9/data/lh_endpoint_data.json').read_text())
    for group in ('L','H'):
        for value in endpoint[group].values():
            if isinstance(value,dict) and 'pairs' in value:
                x=np.asarray(value['pairs']);assert x.shape==(32,2)
                np.testing.assert_allclose(x.mean(0),value['means'],atol=1e-12)
    assertions.append('Figures 6/7/9 retain 32 examples and endpoint means')
    for n,key,shape in [(10,'delta_l2_visual',(32,8)),(11,'kl_final_to_intermediate',(32,128))]:
        data=json.loads((REPO/f'figure{n}/data/plot_data.json').read_text())
        rows=data['state_samples' if n==10 else 'logit_samples']
        assert np.asarray([r[key] for r in rows]).shape==shape
        assert {ds:sum(r['dataset']==ds for r in rows) for ds in ('VMCBench','AI2D')}=={'VMCBench':16,'AI2D':16}
    assertions.append('Figures 10/11 retain the fixed 16+16 samples')
    for folder in [REPO/'figure8']+[REPO/'appendix'/f'figure{n}' for n in range(12,17)]:
        data=json.loads((folder/'data/manifest.json').read_text(encoding='utf-8'))
        assert len(data['cases'])==8
        with np.load(folder/'data/attention.npz',allow_pickle=False) as a:
            for row in data['cases']:
                sid=row['id'];image=folder/'data'/row['sample']['image']
                assert hashlib.sha256(image.read_bytes()).hexdigest()==row['image_sha256']
                first,second=a[sid+'_first'],a[sid+'_second']
                assert first.size==second.size==np.prod(row['grid'])
                assert np.isclose(first.sum(),1) and np.isclose(second.sum(),1)
                expected=(first<=np.quantile(first,.5))&(second>=np.quantile(second,.75))
                assert np.array_equal(expected,a[sid+'_promoted'])
    assertions.append('All 48 qualitative cases pass image/grid/promotion checks')
    commands=[['figure3/verify_legacy_results.py']]
    if args.plots:
        commands += [
            ['figure3/plot.py','--pdf',str(output/'figure3/figure.pdf'),'--preview-dir',str(output/'figure3')],
            ['table4/plot.py','--output',str(output/'table4')],
            ['figure6/plot.py','--output-dir',str(output/'figure6')],
            ['figure7/plot.py','--output-dir',str(output/'figure7')],
            ['figure9/plot.py','--output',str(output/'figure9/figure.pdf')],
            ['figure9/plot.py','--entropy-only','--output',str(output/'figure9/entropy.pdf')],
            ['figure10/plot.py','--out',str(output/'figure10')],
            ['figure11/plot.py','--out',str(output/'figure11')],
        ]
        for folder in ['figure8']+[f'appendix/figure{n}' for n in range(12,17)]:
            commands.append([folder+'/plot.py','--style','all','--dpi','180','--out',str(output/folder)])
    runs=[]
    for cmd in commands:
        result=subprocess.run([sys.executable,*cmd],cwd=REPO,text=True,encoding='utf-8',errors='replace',capture_output=True)
        name='_'.join(cmd[0].split('/')).replace('.py','')+('_entropy' if '--entropy-only' in cmd else '')
        (output/(name+'.log')).write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
        runs.append({'entry':cmd[0],'arguments':cmd[1:],'exit_code':result.returncode})
        print(name, 'PASS' if result.returncode==0 else 'FAIL',flush=True)
        if result.returncode:
            print(result.stderr[-3000:]);raise RuntimeError('Validation failed: '+name)
        if cmd[0] == 'figure9/plot.py':
            plot_output=Path(cmd[cmd.index('--output')+1])
            audit=json.loads(plot_output.with_suffix('.audit.json').read_text(encoding='utf-8'))
            assert audit['sample_order']==endpoint['sample_order']
            assert len(audit['panels'])==(4 if '--entropy-only' in cmd else 6)
            for panel in audit['panels']:
                assert panel['paired_samples']==panel['sample_lines']==panel['displayed_sample_count']==panel['mean_sample_count']==32
            assertions.append('Figure 9 '+('entropy' if '--entropy-only' in cmd else 'six-panel')+' layout displays all 32 samples per panel')
    report={'assertions':assertions,'runs':runs,'all_passed':True,'model_inference_tested':False}
    (output/'validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(assertions,indent=2))


if __name__=='__main__':main()
