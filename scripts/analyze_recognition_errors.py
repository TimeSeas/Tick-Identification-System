"""Summarize a completed epoch without reading images or using the test set."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split',type=Path,required=True)
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    split=json.loads(args.split.read_text('utf-8'))
    records=json.loads(args.predictions.read_text('utf-8'))
    args.out.mkdir(parents=True,exist_ok=True)
    train_groups={c:set() for c in split['classes']}
    for row in split['train']:
        if row['species_label']:
            train_groups[row['species_label']].add(row['group_id'])
    stats=[]; confusions=Counter()
    for i,c in enumerate(split['classes']):
        truth=[r for r in records if r['species_label']==i]
        tp=sum(r['species_prediction']==i for r in truth)
        fp=sum(r['species_label']>=0 and r['species_label']!=i and r['species_prediction']==i for r in records)
        fn=len(truth)-tp
        stats.append(dict(species=c,train_groups=len(train_groups[c]),validation_views=len(truth),
            correct=tp,recall=tp/len(truth) if truth else None,
            f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None))
        for r in truth:
            if r['species_prediction']!=i:
                confusions[c,split['classes'][r['species_prediction']]]+=1
    with (args.out/'species.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(stats[0])); writer.writeheader(); writer.writerows(stats)
    misses=[]
    for r in records:
        if (r['tick_probability']>=.5)!=bool(r['tick_label']):
            misses.append(dict(r,path=split['val'][r['index']]['path']))
    result=dict(predictions=str(args.predictions),species=stats,
        confusions=[dict(truth=a,predicted=b,views=n) for (a,b),n in confusions.most_common(30)],
        binary_errors=misses,
        note='correlated validation views, not independent test samples; no class relabeling')
    (args.out/'errors.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8')
    print(json.dumps(dict(worst_species=sorted(stats,key=lambda r:r['f1'])[:8],binary_errors=len(misses)),indent=2))


if __name__=='__main__':
    main()
