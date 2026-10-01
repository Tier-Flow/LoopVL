"""Replot the saved spatial-entropy curve; no inference or sample selection."""
from pathlib import Path
import sys
_FINDINGS_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "common" / "paths.py").is_file())
sys.path.insert(0, str(_FINDINGS_ROOT))
from common.paths import MODEL_ROOT, DATA_ROOT, OUTPUT_ROOT, repo_path, tsv_directory
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

def draw(data_path,output_dir):
    data=json.loads(Path(data_path).read_text(encoding='utf-8'))
    x=np.asarray(data['layers']); y=np.asarray(data['mean'])
    assert x.shape==y.shape==(128,) and np.array_equal(x,np.arange(128))
    assert np.isfinite(y).all() and ((y>=0)&(y<=1)).all()
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42})
    fig=plt.figure(figsize=(19.6,8.2),dpi=160,facecolor='white')
    ax=fig.add_axes([.105,.16,.855,.66]); ink='#3F5060'
    ax.set_xlim(-4,128);ax.set_ylim(0,1)
    ax.set_xticks([0,32,64,96,128]);ax.set_yticks(np.linspace(0,1,6))
    ax.tick_params(colors=ink,labelsize=24,length=5,width=.9,pad=8)
    ax.grid(axis='y',color='#DFE5ED',linewidth=.6)
    for spine in ax.spines.values():spine.set_color('#8798A9');spine.set_linewidth(.95)
    for b in data['module_loop_boundaries']+[data['model_loop_boundary']]:
        model=b==data['model_loop_boundary']
        ax.vlines(b,.34,.985,color='#D14959' if model else '#E8BCC5',
            linewidth=1.35 if model else 1.05,linestyles='solid' if model else (0,(4,3)))
        ax.text(b,1.035,'model loop' if model else 'module loop',ha='center',va='bottom',
            color='#D14959' if model else '#CB919C',fontsize=21,weight='bold' if model else 'normal')
    dense=np.linspace(0,127,2541); interp=PchipInterpolator(x,y)
    np.testing.assert_allclose(interp(x),y,rtol=0,atol=1e-15)
    ax.plot(dense,interp(dense),color='#4F78BB',lw=2.25)
    ax.set_xlabel('Unrolled depth (zero-based)',fontsize=27,labelpad=20,color=ink)
    ax.set_ylabel('Spatial entropy' if data['metric']=='entropy' else 'Gini',fontsize=27,labelpad=21,color=ink)
    fig.text(.53,.94,data['title'],ha='center',va='center',fontsize=30,weight='bold',color=ink)
    for suffix in ('pdf','svg','png'):fig.savefig(output_dir/f'figure.{suffix}',dpi=160,facecolor='white')
    plt.close(fig)
    audit={'sample_count':data['sample_count'],'data_points':len(y),'mean_values_63_64_127':y[[63,64,127]].tolist(),
        'shape_preserving_interpolation':True,'data_changed':False,'metric':data['metric'],
        'query_scope':data['query_scope'],'font_note':'portable DejaVu Sans; original paper uses Arial'}
    (output_dir/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2))

if __name__=='__main__':
    here=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=repo_path,default=here/'data/plot_data.json')
    parser.add_argument('--output-dir',type=repo_path,default=OUTPUT_ROOT / 'figure6')
    args=parser.parse_args();draw(args.data,args.output_dir)
