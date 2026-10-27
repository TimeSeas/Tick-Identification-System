"""Exercise an explicitly isolated local API; never use the user's history DB."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import time
import urllib.request
import uuid
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]


def validate_smoke_database(database):
    if not Path(database).resolve().is_relative_to((ROOT/'output/optimization_20261015').resolve()):
        raise ValueError('Smoke database must be under the ignored experiment directory')


def decode_result_image(value):
    if not value.startswith('data:image/jpeg;base64,'):
        raise ValueError('Expected the service JPEG data URI')
    decoded=base64.b64decode(value.split(',',1)[1],validate=True)
    with Image.open(io.BytesIO(decoded)) as image:
        image.verify()
    return decoded


def request(base,path,content=None,mime=None):
    headers={'Content-Type':mime} if mime else {}
    req=urllib.request.Request(base+path,data=content,headers=headers)
    with urllib.request.urlopen(req,timeout=180) as response:
        return json.load(response)


def upload(base,data,mime):
    boundary='codex'+uuid.uuid4().hex
    content=(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="smoke.jpg"\r\n'
             f'Content-Type: {mime}\r\n\r\n').encode()+data+f'\r\n--{boundary}--\r\n'.encode()
    return request(base,'/api/v1/analyze',content,f'multipart/form-data; boundary={boundary}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',default='http://127.0.0.1:8016')
    p.add_argument('--database',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--cases',type=Path,required=True)
    p.add_argument('--expected-detector',type=Path,default=ROOT/'output/det_v3/mask_rcnn_ticks_v3.pth')
    p.add_argument('--device-label',choices=('cpu','cuda','unspecified'),default='unspecified')
    args=p.parse_args()
    database=args.database.resolve()
    validate_smoke_database(database)
    if args.out.exists():
        raise ValueError('Preserve prior smoke evidence')
    with sqlite3.connect(database) as conn:
        if conn.execute('select count(*) from analysis_records').fetchone()[0]!=0:
            raise ValueError('Require an empty isolated database')
    health=request(args.base,'/healthz'); assert health['code']==200
    before=request(args.base,'/api/v1/history')['data']['total']; assert before==0
    wrong_type=upload(args.base,b'not an image','text/plain'); assert wrong_type['code']==400
    broken_image=upload(args.base,b'not an image','image/jpeg'); assert broken_image['code']==400
    too_large=upload(args.base,b'0'*(10*1024*1024+1),'image/jpeg'); assert too_large['code']==400
    assert request(args.base,'/api/v1/history')['data']['total']==0
    cases=json.loads(args.cases.read_text('utf-8')); row=cases['cases'][0]['row']
    data=(ROOT/row['path']).read_bytes()
    assert hashlib.sha256(data).hexdigest()==row['sha256']
    started=time.perf_counter(); result=upload(args.base,data,'image/jpeg'); elapsed=time.perf_counter()-started
    assert result['code']==200
    analyzed=result['data']; assert analyzed['model_version']['profile_sha256']
    assert analyzed['model_version']['detector_model_sha256']==hashlib.sha256(args.expected_detector.read_bytes()).hexdigest()
    assert analyzed['model_version']['species_model_sha256'] and analyzed['model_version']['insect_model_sha256']
    assert analyzed['result_image_base64'] and analyzed['request_id']
    history=request(args.base,'/api/v1/history')['data']; assert history['total']==1
    detail=request(args.base,f"/api/v1/history/{history['items'][0]['id']}")['data']
    assert detail['request_id']==analyzed['request_id'] and detail['model_version']==analyzed['model_version']
    with sqlite3.connect(database) as conn:
        assert conn.execute('select count(*) from analysis_records').fetchone()[0]==1
    args.out.mkdir(parents=True)
    (args.out/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8')
    (args.out/'result.jpg').write_bytes(decode_result_image(analyzed['result_image_base64']))
    report=dict(health=health,invalid_mime=wrong_type,invalid_image=broken_image,
        too_large=too_large,input_path=row['path'],input_sha256=row['sha256'],
        request_seconds=elapsed,history_count=history['total'],database=str(database),
        detections=len(analyzed['detections']),model_version=analyzed['model_version'],
        device_label=args.device_label,
        note='Actual HTTP single-image request; not a controlled speed comparison or accuracy evaluation')
    (args.out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf-8')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':
    main()
