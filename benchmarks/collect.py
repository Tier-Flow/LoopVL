"""Partial/final screenshot comparison using fresh predictions only."""
import argparse
from collections import defaultdict
import importlib.util
import json
import os
from pathlib import Path
import time

def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.tmp-{os.getpid()}')
    temporary.write_text(value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    temporary.replace(path)

def collect(benchmark_root,results_root):
    path=benchmark_root/'scripts/reproduce_scores.py'
    spec=importlib.util.spec_from_file_location('original_score_reproduction',path)
    score=importlib.util.module_from_spec(spec);spec.loader.exec_module(score)
    statuses={}; values={}; audits={}
    aliases={'AI2D_TEST':'AI2D','ChartQA_TEST':'ChartQA','VMCBench_DEV':'VMCBench'}
    for name,expected in score.DATASET_COUNTS.items():
        predictions=results_root/name/'predictions.jsonl'
        statuses[name]={'expected':expected,'complete':False,'completed':0}
        if not predictions.exists():continue
        try:
            records,duplicates=score.read_records(predictions)
            successful=[row for row in records if not row.get('error')]
            statuses[name].update(completed=len(successful),error_records=len(records)-len(successful))
            if len(successful)!=expected or len(records)!=expected:continue
            rows,audit=score.score_records(records,name,preserve_original=False)
            audits[name]=audit
            if any(row.get('reference') is None for row in rows) or audit['stored_flag_mismatch_count']:
                statuses[name]['integrity_error']='Missing references or stored correctness flags do not match archived scoring code'
                continue
            per={aliases.get(name,name):score.percentage(row['correct'] for row in rows)}
            if name=='ChartQA_TEST':
                matcher=score.get_chartqa_matcher()
                per={'ChartQA':score.percentage(score.chartqa_hit(row,matcher) for row in rows)}
            elif name=='HallusionBench':per={'HallusionBench':score.hallusion_scores(rows)['leaderboard_Avg']}
            elif name=='MMEval-Pro':per={'MMEval-Pro':score.SPECIAL.mmeval_scores(rows)['Macro_Average']['genuine_accuracy_score']}
            elif name=='POPE':
                details=score.pope_scores(rows)
                assert details['pooled']['total']==9000
                assert all(details['by_category'][cat]['total']==3000 for cat in ['adversarial','popular','random'])
                per={'POPE':100*details['pooled']['f1']}
                for suffix,cat in [('A','adversarial'),('P','popular'),('R','random')]:per['POPE-'+suffix]=100*details['by_category'][cat]['f1']
            elif name=='MMK12':
                groups=defaultdict(list)
                for row in rows:groups[str(row.get('metadata',{}).get('subject','')).lower()].append(row['correct'])
                assert all(len(groups[subject])==500 for subject in ['math','biology','chemistry','physics'])
                for subject in ['math','biology','chemistry','physics']:per['MMK12-'+subject.title()]=score.percentage(groups[subject])
            values.update(per);statuses[name]['complete']=True
            statuses[name]['duplicate_lines_removed']=duplicates
        except Exception as exc:
            statuses[name]['read_or_integrity_error']=f'{type(exc).__name__}: {exc}'
    table=[]
    for category,benchmark,baseline in score.EXPECTED_ROWS:
        value=values.get(benchmark)
        table.append({'category':category,'benchmark':benchmark,'screenshot':baseline,'fresh_score':value,
                      'delta':None if value is None else value-baseline})
    report={'updated_at':time.time(),'mode':'fresh_singlefile_inference_vs_screenshot',
            'all_complete':all(row['complete'] for row in statuses.values()),
            'completed_datasets':sum(row['complete'] for row in statuses.values()),
            'datasets':statuses,'table':table,'scoring_audits':audits,
            'historical_predictions_used_as_fresh_results':False,'score_equality_required':False}
    if report['all_complete']:
        full=score.evaluate(results_root,preserve_original=False)
        report['full_integrity_passed']=score.verification_passes(full,allow_different_scores=True)
        full['mode']='rescoring_fresh_singlefile_model_predictions'
        report['full_audit']=full
        if not report['full_integrity_passed']:report['all_complete']=False
    return report

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    root=Path(__file__).resolve().parent
    parser.add_argument('--benchmark-root',type=Path,default=root)
    parser.add_argument('--results-root',type=Path,default=root.parent/'outputs/benchmarks/results')
    parser.add_argument('--output-dir',type=Path,default=root.parent/'outputs/benchmarks/comparison')
    args=parser.parse_args()
    for field in ('benchmark_root','results_root','output_dir'):
        path=getattr(args,field)
        setattr(args,field,(path if path.is_absolute() else root.parent/path).resolve())
    result=collect(args.benchmark_root,args.results_root)
    atomic(args.output_dir/'comparison.json',result)
    lines=['# LoopVL single-file rerun vs screenshot','',f"Complete benchmarks: {result['completed_datasets']}/16. Pending entries are not partial scores.",'',
           '| Category | Benchmark | Screenshot | Fresh rerun | Delta |','|---|---|---:|---:|---:|']
    for row in result['table']:
        value='待完成' if row['fresh_score'] is None else f"{row['fresh_score']:.2f}"
        delta='—' if row['delta'] is None else f"{row['delta']:+.2f}"
        lines.append(f"| {row['category']} | {row['benchmark']} | {row['screenshot']:.2f} | {value} | {delta} |")
    lines+=['','Fresh generation only. Fixed original-brief prompt policy (RealWorldQA: raw question), archived per-item output budgets, original scoring metrics. Historical prompt identity is not known per item.']
    atomic(args.output_dir/'comparison.md','\n'.join(lines)+'\n')
    if 'full_audit' in result:atomic(args.output_dir/'full_scoring_audit.json',result['full_audit'])
    print(json.dumps({'completed_datasets':result['completed_datasets'],'all_complete':result['all_complete']}))

if __name__=='__main__':main()
