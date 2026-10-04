"""Read retained numerical receipts; keep pair margins separate from their mean."""
import csv
import json
import sys
from pathlib import Path

def read(path):
    return list(csv.DictReader(path.open(),delimiter='\t'))

def analyze(root):
    manifest=json.loads((root/'COMPLETE.json').read_text())
    rows=read(root/'probes.tsv')
    weights=read(root/'weights.tsv')
    summaries=[]
    for case in sorted({r['case'] for r in rows}):
        selected=[r for r in rows if r['case']==case]
        result={'case':case,'learning_rate':manifest['learning_rate'],'source':manifest['source']['commit_sha']}
        for name in sorted({r['probe'] for r in selected}):
            first=[r for r in selected if r['probe']==name and r['step']=='1']
            final=[r for r in selected if r['probe']==name and r['step']==str(manifest['steps'])]
            if first[0]['pair']:
                margins=[]
                for pair in ('1','2','3'):
                    before={r['candidate']:float(r['score_before']) for r in first if r['pair']==pair}
                    after={r['candidate']:float(r['score_after']) for r in final if r['pair']==pair}
                    margins.append({'pair':pair,'before':before['school']-before['check'],
                                    'after':after['school']-after['check']})
                result[name]={'pairs':margins,'aggregate_change':sum(r['after']-r['before'] for r in margins)/3}
            else:
                result[name]={'loss_before':-float(first[0]['score_before']),
                              'loss_after':-float(final[0]['score_after']),
                              'logit_residual_fraction_final':float(final[0]['logit_residual_l2'])/float(final[0]['logit_actual_l2']),
                              'fd_error_fraction_final':float(final[0]['jvp_fd_error_l2'])/float(final[0]['logit_tangent_l2'])}
        case_weights=[r for r in weights if r['case']==case]
        result['mixed_term_first']=float(case_weights[0]['deltaB_deltaA_l2'])
        result['mixed_term_second']=float(case_weights[1]['deltaB_deltaA_l2'])
        result['mixed_term_final']=float(case_weights[-1]['deltaB_deltaA_l2'])
        result['max_reconstruction_error_l2']=max(float(r['reconstruction_error_l2']) for r in case_weights)
        summaries.append(result)
    return summaries

if __name__=='__main__':
    print(json.dumps([{'run':arg,'results':analyze(Path(arg))} for arg in sys.argv[1:]],indent=2))
