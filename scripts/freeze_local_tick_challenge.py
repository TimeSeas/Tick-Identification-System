"""Freeze positive-only, unsupported-species coverage cases without model scores.

Uses existing local files only. Screens known SHA/photo/observation and dHash;
incomplete provenance and missing historical paths prevent an independence claim.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

from build_recognition_rebuild_data import EXTENSIONS,Groups,image_fingerprint,near_matches,stable_order

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--limit',type=int,default=80)
    args=p.parse_args()
    if args.limit<1 or args.out.exists():
        raise ValueError('Require a new manifest and positive limit')
    audit_path=ROOT/'output/optimization_20261015/rebuild_20261024/data_audit.json'
    audit=json.loads(audit_path.read_text('utf-8'))
    reference=set(); species=set(); identities={}
    sources={str(audit_path.relative_to(ROOT)):digest(audit_path)}
    split_sources=sorted((ROOT/'data/tick_species').glob('split*.json'))+[ROOT/'data/insects_split.json']
    for path in split_sources:
        name=path.relative_to(ROOT).as_posix(); sources[name]=digest(path)
        for rows in json.loads(path.read_text('utf-8')).values():
            for row in rows:
                reference.add(row['path'])
                if name=='data/tick_species/split_clean.json':
                    species.add(row['label'])
    for subset in ('train','valid','test'):
        name=f'ticks image detection.v5i.coco/{subset}/_annotations.coco.json'
        path=ROOT/name; sources[name]=digest(path)
        for row in json.loads(path.read_text('utf-8'))['images']:
            reference.add(f'ticks image detection.v5i.coco/{subset}/'+row['file_name'])
    challenge_source=ROOT/'output/optimization_20261015/rebuild_20261024/data/unseen_binary_test.json'
    for row in json.loads(challenge_source.read_text('utf-8')):
        reference.add(row['path'])
    sources[str(challenge_source.relative_to(ROOT))]=digest(challenge_source)
    groups=Groups()
    for path,metas in audit['photo_id_audit']['path_provenance'].items():
        for meta in metas:
            match=re.search(r'inaturalist.org/observations/(\d+)',meta.get('observation_reference',''))
            for kind,value in (('photo',meta.get('photo_id')),('observation',match.group(1) if match else None)):
                if value:
                    key=(kind,str(value))
                    if key in identities:
                        groups.union(path,identities[key])
                    identities[key]=path
    reference_groups={groups.find(path) for path in reference}
    hashes=set(); index=defaultdict(set); missing=[]
    for number,relative in enumerate(sorted(reference)):
        path=ROOT/relative
        if not path.is_file():
            missing.append(relative); continue
        hashes.add(digest(path))
        try:
            value,_,_=image_fingerprint(path)
        except (OSError,ValueError) as error:
            missing.append(dict(path=relative,reason=str(error)))
            continue
        for offset in range(8):
            index[(offset,(value>>(offset*8))&255)].add(value)
        if (number+1)%1000==0:
            print(f'screened reference {number+1}/{len(reference)}',flush=True)
    rows=[]; excluded=[]; selected_groups=set(); represented=set()
    candidates=[p for p in (ROOT/'data/tick_species').glob('*/*')
        if p.suffix.lower() in EXTENSIONS and p.parent.name not in species]
    candidates.sort(key=lambda path:stable_order(path.relative_to(ROOT).as_posix(),42))
    for path in candidates:
        relative=path.relative_to(ROOT).as_posix(); label=path.parent.name; group=groups.find(relative)
        if label in represented or group in selected_groups:
            continue
        reason=None
        if relative in reference or group in reference_groups:
            reason='known_path_or_photo_observation_group'
        else:
            sha=digest(path)
            try:
                value,width,height=image_fingerprint(path)
            except (OSError,ValueError) as error:
                excluded.append(dict(path=relative,reason='decode_failed',detail=str(error)))
                continue
            if min(width,height)<64:
                reason='dimension_below_64'
            elif sha in hashes or near_matches(value,index):
                reason='known_or_selected_SHA_or_dHash_distance_le_4'
        if reason:
            excluded.append(dict(path=relative,reason=reason)); continue
        rows.append(dict(path=relative,label=label,binary_label=1,species_supported=False,
            sha256=sha,dhash=f'{value:016x}',width=width,height=height,
            group_id=hashlib.sha256(group.encode()).hexdigest()[:20],
            provenance='existing_folder_species_label_not_independently_adjudicated'))
        represented.add(label); selected_groups.add(group); hashes.add(sha)
        for offset in range(8):
            index[(offset,(value>>(offset*8))&255)].add(value)
        if len(rows)>=args.limit:
            break
    result=dict(selection='seed42 stable path order, at most one per unsupported species and known source group',
        purpose='positive-only tick presence coverage; no species accuracy, localization or false-positive truth',
        manifest_builder_sha256=digest(Path(__file__)),sources=sources,rows=rows,excluded=excluded,
        missing_historical_paths=missing,
        limitations=['known metadata and 64-bit dHash screening cannot prove full independence',
            'missing historical paths cannot be fingerprinted','folder species labels not independently checked',
            'unsupported species must not be scored as one of the 28 deployed species',
            'freeze before predictions; evaluate after model/policy selection, do not tune on these results'])
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8')
    print(json.dumps(dict(cases=len(rows),missing=len(missing),sha256=digest(args.out))))


if __name__=='__main__':
    main()
