"""Freeze species-only views and species-disjoint unknown ticks from local files."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import re

from build_recognition_rebuild_data import Groups,image_fingerprint,near_matches,stable_order,EXTENSIONS

ROOT=Path(__file__).resolve().parents[1]
PREVIOUS=ROOT/'output/optimization_20261015/rebuild_20261024'
SUBSETS=('train','val','calibration','holdout')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),'utf-8');temp.replace(path)


def build_unknown_assignment(labels,seed):
    ordered=sorted(set(labels),key=lambda label:stable_order(label,seed))
    if len(ordered)<4:raise ValueError('At least four unknown species required')
    count=max(1,len(ordered)//6)
    result={label:'train' for label in ordered}
    for i,subset in enumerate(('val','calibration','holdout')):
        for label in ordered[i*count:(i+1)*count]:result[label]=subset
    return result


def validate_specialist_manifest(manifest):
    classes=manifest['classes']
    if not classes or classes!=sorted(set(classes)):raise ValueError('Require sorted unique classes')
    identities={};unknown_species={}
    for subset in SUBSETS:
        if not manifest[subset]:raise ValueError('Empty subset '+subset)
        for row in manifest[subset]:
            for key in ('sha256','group_id'):
                token=key,row[key]
                if token in identities and identities[token]!=subset:raise ValueError('Cross-split '+key)
                identities[token]=subset
            label=row['species_label'];source=row['source_species']
            if label is not None:
                if label not in classes or source!=label:raise ValueError('Invalid known species label')
            else:
                if source in classes:raise ValueError('Known species cannot be relabelled as unknown')
                if source in unknown_species and unknown_species[source]!=subset:
                    raise ValueError('Unknown species crosses subsets')
                unknown_species[source]=subset
    return classes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    sources={}
    def read(path):
        sources[path.relative_to(ROOT).as_posix()]=digest(path)
        return json.loads(path.read_text('utf-8'))
    regions=read(PREVIOUS/'regions/recognition_split.json')
    challenge=read(PREVIOUS/'data/unsupported_tick_challenge_v2.json')
    audit=read(PREVIOUS/'data_audit.json')
    blocked_species={r['label'] for r in challenge['rows']}
    classes=regions['classes']
    reference=set()
    for path in sorted((ROOT/'data/tick_species').glob('split*.json'))+[ROOT/'data/insects_split.json']:
        for rows in read(path).values():reference.update(r['path'] for r in rows)
    for subset in ('train','valid','test'):
        path=ROOT/f'ticks image detection.v5i.coco/{subset}/_annotations.coco.json'
        reference.update(f'ticks image detection.v5i.coco/{subset}/'+r['file_name'] for r in read(path)['images'])
    reference.update(r['path'] for r in read(PREVIOUS/'data/unseen_binary_test.json'))
    reference.update(r['path'] for r in challenge['rows'])
    groups=Groups();identity_map={}
    for path,metas in audit['photo_id_audit']['path_provenance'].items():
        for meta in metas:
            match=re.search(r'inaturalist.org/observations/(\d+)',meta.get('observation_reference',''))
            for kind,value in (('photo',meta.get('photo_id')),('observation',match.group(1) if match else None)):
                if value:
                    token=kind,str(value)
                    if token in identity_map:groups.union(path,identity_map[token])
                    identity_map[token]=path
    reference_groups={groups.find(path) for path in reference}
    hashes=set();index=defaultdict(set);missing=[]
    for number,path in enumerate(sorted(reference),1):
        file=ROOT/path
        if not file.is_file():missing.append(dict(path=path,reason='missing'));continue
        hashes.add(digest(file))
        try:value,_,_=image_fingerprint(file)
        except (OSError,ValueError) as error:
            missing.append(dict(path=path,reason=str(error)));continue
        for offset in range(8):index[offset,(value>>(offset*8))&255].add(value)
        if number%1000==0:print(f'reference {number}/{len(reference)}',flush=True)
    candidates=[];excluded=[]
    for file in sorted((ROOT/'data/tick_species').glob('*/*')):
        if file.suffix.lower() not in EXTENSIONS or file.parent.name in set(classes)|blocked_species:continue
        path=file.relative_to(ROOT).as_posix();group=groups.find(path)
        if path in reference or group in reference_groups:
            excluded.append(dict(path=path,reason='historical_reference_path_or_source_group'));continue
        try:
            sha=digest(file);value,w,h=image_fingerprint(file)
        except (OSError,ValueError) as error:
            excluded.append(dict(path=path,reason=str(error)));continue
        if min(w,h)<64 or sha in hashes or near_matches(value,index):
            excluded.append(dict(path=path,reason='small_or_historical_SHA_dHash_match'));continue
        candidates.append(dict(path=path,sha256=sha,source_species=file.parent.name,
            species_label=None,bbox=None,coordinate_frame='exif_normalized',group=group,dhash=value,
            width=w,height=h,sample_type='unsupported_species_full',tick_label=1))
    # Build unknown connected components before choosing representatives. A duplicate
    # carrying two labels is quarantined on both sides, never assigned the first label.
    unknown_groups=Groups();first_sha={};unknown_index=defaultdict(set);first_fingerprint={}
    for row in candidates:
        path=row['path'];unknown_groups.union(path,row['group'])
        if row['sha256'] in first_sha:unknown_groups.union(path,first_sha[row['sha256']])
        first_sha[row['sha256']]=path
        for value in near_matches(row['dhash'],unknown_index):
            unknown_groups.union(path,first_fingerprint[value])
        first_fingerprint[row['dhash']]=path
        for offset in range(8):unknown_index[offset,(row['dhash']>>(offset*8))&255].add(row['dhash'])
    component_labels=defaultdict(set)
    for row in candidates:component_labels[unknown_groups.find(row['path'])].add(row['source_species'])
    unique={}
    for row in sorted(candidates,key=lambda r:(-r['width']*r['height'],r['path'])):
        component=unknown_groups.find(row['path'])
        if len(component_labels[component])>1:
            excluded.append(dict(path=row['path'],reason='unknown_component_species_conflict'));continue
        if component in unique:
            excluded.append(dict(path=row['path'],reason='unknown_group_representative'));continue
        row=dict(row);row['group_id']=hashlib.sha256(component.encode()).hexdigest()[:20]
        row.pop('group');row.pop('dhash');unique[component]=row
    assignment=build_unknown_assignment([r['source_species'] for r in unique.values()],args.seed)
    result=dict(classes=classes,metadata=dict(sources=sources,seed=args.seed,
        old_challenge_species_excluded=sorted(blocked_species),unknown_assignment=assignment,
        missing_reference_count=len(missing),known_view_preprocessing='exif_context015_center_or_full448',
        limitations=['old species task weights saw some grouped known val sources',
            'folder labels and teacher boxes are proxies, not new human adjudication',
            'historical missing images and incomplete provenance prevent full independence proof',
            'unknown holdout species unseen by this experiment; old80 only retrospective']))
    for subset in SUBSETS:
        result[subset]=[dict(row,source_species=row['species_label']) for row in regions.get(subset,[])
            if row['species_label'] is not None]
        result[subset].extend(row for row in unique.values() if assignment[row['source_species']]==subset)
    validate_specialist_manifest(result)
    for subset in SUBSETS:
        for row in result[subset]:
            if digest(ROOT/row['path'])!=row['sha256']:raise ValueError('Changed frozen image '+row['path'])
    save(args.out/'manifest.json',result)
    save(args.out/'excluded.json',dict(excluded=excluded,missing_reference_images=missing))
    review=[]
    old_report=read(PREVIOUS/'detector_species_balanced_run01/predictions.json')
    for row in old_report:
        ds=row['new']['detections']
        if not ds or any(d['candidate_slug']!=row['label'] or d['status']!='confirmed' for d in ds):
            review.append(dict(path=row['path'],label=row['label'],role='review_only_previous_val',detections=ds))
    browser=read(PREVIOUS/'browser_species_20261026_run01/report.json')
    review.extend(dict(path=r['path'],label=r['folder_label'],role='review_only_browser',detections=r['detections'])
        for r in browser['rows'] if any(d['candidate_slug']!=r['folder_label'] for d in r['detections']))
    save(args.out/'review_queue.json',review)
    counts={s:dict(known=sum(r['species_label'] is not None for r in result[s]),
        unknown=sum(r['species_label'] is None for r in result[s]),
        unknown_species=len({r['source_species'] for r in result[s] if r['species_label'] is None})) for s in SUBSETS}
    save(args.out/'summary.json',dict(counts=counts,manifest_sha256=digest(args.out/'manifest.json'),
        missing_references=len(missing),excluded=len(excluded),review_only_cases=len(review),builder_sha256=digest(__file__)))
    print(json.dumps(dict(counts=counts,excluded=len(excluded),missing_references=len(missing)),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
