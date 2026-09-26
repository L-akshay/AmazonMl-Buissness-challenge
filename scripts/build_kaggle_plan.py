"""Build private full-population jobs; this script does not launch remote work.

Run the three-job plan first. Its measured capacity gate must pass before the
controller continues to full_plan.json. Both plans share the scheduler journal.
"""

import importlib.util
import json
import math
from pathlib import Path

root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('builder',root/'scripts/build_kaggle_job.py')
builder=importlib.util.module_from_spec(spec); spec.loader.exec_module(builder)
jobs=[]
def add(slug,stages,inputs=(),overrides=None):
    folder=root/'cache/kaggle_jobs'/slug
    builder.build(root,folder,slug,stages,inputs,{'threads':4,'feature_workers':3,'session_hours':9,**(overrides or {})})
    id='lakshaytechai/'+slug
    jobs.append({'id':id,'folder':str(folder),'depends':list(inputs),'timeout':36000})
    return id

prep=add('amazon-er-v3-prepare',['prepare'],overrides={'session_hours':2.5})
probe=add('amazon-er-v3-first-batches',['retrieve-train','features-train'],[prep],
          {'batch_start':0,'batch_stop':2,'session_hours':1})
index=add('amazon-er-v3-test-index',['index-test'],[prep],{'session_hours':2.5})
out=root/'cache/managed_kaggle'; out.mkdir(exist_ok=True)
initial={'max_parallel':2,'jobs':list(jobs),'followup_plan':'full_plan.json',
         'capacity_gate':{'sample_job':probe,'sample_batches':2,'partition_batches':200,
                          'time_safety_factor':2,'max_partition_hours':8,
                          'max_partition_output_gib':16,'max_peak_rss_gib':23,
                          'stages':{probe:['retrieve-train','features-train'],index:['index-test']}}}
(out/'plan.json').write_text(json.dumps(initial,indent=2))
audit=json.loads((root/'reports/data_audit.json').read_text())
counts={split:sum(audit['sources'][f'{split}_source{s}']['rows'] for s in (2,3)) for split in ('train','test')}
train_parts=[]; test_parts=[]
for split,part_list in (('train',train_parts),('test',test_parts)):
    batches=math.ceil(counts[split]/10000)
    ranges=[]
    for start in range(0,batches,200):
        first=max(start,2) if split=='train' else start
        stop=min(start+200,batches)
        ranges.append((first,stop))
        inputs=[prep] if split=='train' else [prep,index]
        part_list.append(add(f'amazon-er-v3-{split}-{start//200:02d}',
            [f'retrieve-{split}',f'features-{split}'],inputs,
            {'batch_start':first,'batch_stop':stop}))
    indices=[i for first,stop in ranges for i in range(first,stop)]
    if split=='train':
        indices=[0,1]+indices
    assert indices==list(range(batches))
train_inputs=[prep,probe,*train_parts]
test_inputs=[prep,index,*test_parts]
train_assembled=add('amazon-er-v3-train-assembled',['assemble-train','research-contributions'],train_inputs)
test_assembled=add('amazon-er-v3-test-assembled',['assemble-test'],test_inputs)
train_inputs.append(train_assembled); test_inputs.append(test_assembled)
oof=[]
for kind in ('gbdt','logistic'):
    for fold in range(4):
        oof.append(add(f'amazon-er-v3-{kind}-{fold}',[f'oof-{kind}-{fold}'],train_inputs))
validation=add('amazon-er-v3-validation',['validate'],[*train_inputs,*oof])
final=add('amazon-er-v3-final',['final'],[*train_inputs,validation])
scoring=add('amazon-er-v3-scoring',['score'],[*test_inputs,final])
export=add('amazon-er-v3-export',['export'],[prep,test_assembled,final,scoring,validation])
jobs[-1]['download']=str(root/'output/kaggle_delivery/predictions')
package=add('amazon-er-v3-package',['package'],[prep,train_assembled,validation,export])
jobs[-1].update(download=str(root/'output/kaggle_delivery/package'),package=True)
for experiment in ('country-us','country-india','ablation-address','ablation-numeric','ablation-frequency','ablation-retrieval'):
    # Independent diagnostics follow the primary deliverables in queue order.
    id=add('amazon-er-v3-'+experiment,['research-'+experiment],train_inputs)
    jobs[-1]['depends'].append(package)
(out/'full_plan.json').write_text(json.dumps({'max_parallel':2,'jobs':jobs},indent=2))
print('Prepared full plan; not activated:',out/'full_plan.json',len(jobs),'jobs')
