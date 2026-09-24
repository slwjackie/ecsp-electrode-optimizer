"""Staging-only packaging: publish nothing unless every frozen byte matches."""
from __future__ import annotations
import base64, hashlib, importlib, json, lzma, os, platform, sys, tempfile, zipfile, zlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
parts=sorted((ROOT/'bundle_upload').glob('payload_*.b64'))
expected_names=[f'payload_{i:02}.b64' for i in range(15)]
if [p.name for p in parts]!=expected_names:
    raise RuntimeError('Missing or extra payload parts')
text=''.join(''.join(p.read_text().split()) for p in parts)
if len(text)!=179432 or hashlib.sha256(text.encode()).hexdigest()!='2130fc7881cbffcb82cfb9c81abd5dcf956ed136bebc5153fb29010f140616c8':
    raise RuntimeError('Transport checksum mismatch')
payload=json.loads(lzma.decompress(base64.b64decode(text,validate=True)))
expected_archive='c0aad302a4e1a7a6faa233f59f2575b7592be13956624d3d6bcd3c921941356f'
assert payload['archive_sha256']==expected_archive
assert len(payload['references'])==len(payload['metadata_text'])==143
assert len(payload['members'])==429
expected={m['filename']:m['sha256'] for m in payload['members']}
assert len(expected)==429
contents={}
with tempfile.TemporaryDirectory(prefix='ecsp-frozen-replay-') as temp:
    tmp=Path(temp)
    (tmp/'geometry.py').write_text(payload['geometry_source'])
    (tmp/'replay.py').write_text(payload['replay_source'])
    sys.path.insert(0,temp)
    replay=importlib.import_module('replay')
    for index,(sid,original_text) in enumerate(payload['metadata_text'].items(),1):
        meta=json.loads(original_text)
        output=replay.replay(payload['references'][sid],meta)
        output['metadata.json']=original_text.encode()
        for filename,data in output.items():
            key=f'library/{sid}/{filename}'
            actual=hashlib.sha256(data).hexdigest()
            if actual!=expected.get(key):
                raise RuntimeError(f'Frozen file changed: {key}: {actual} != {expected.get(key)}')
            contents[key]=data
        print(f'{index}/143 {sid}: all three original files identical',flush=True)
assert set(contents)==set(expected)
destination=ROOT/'data/image_design/frozen143_geometry.zip'
destination.parent.mkdir(parents=True,exist_ok=True)
staging=destination.with_suffix('.zip.tmp')
with zipfile.ZipFile(staging,'w') as archive:
    archive.comment=base64.b64decode(payload['archive_comment'])
    for member in payload['members']:
        info=zipfile.ZipInfo(member['filename'],tuple(member['date_time']))
        for key in ('compress_type','create_system','create_version','extract_version',
                    'reserved','flag_bits','volume','internal_attr','external_attr'):
            setattr(info,key,member[key])
        info.extra=base64.b64decode(member['extra']);info.comment=base64.b64decode(member['comment'])
        archive.writestr(info,contents[info.filename],compresslevel=payload['zip_compression_level'])
data=staging.read_bytes();actual=hashlib.sha256(data).hexdigest()
if len(data)!=13217873 or actual!=expected_archive:
    raise RuntimeError(f'Archive bytes changed; WILL NOT PUBLISH: {len(data)} {actual}')
git_blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
assert git_blob=='62379ad971344ce19974450383fa5afceec68a05'
os.replace(staging,destination)
import numpy,scipy,shapely,skimage,cv2
record=dict(archive_sha256=actual,archive_bytes=len(data),git_blob_sha=git_blob,
    geometry_count=143,byte_identical_member_count=429,all_original_files_unchanged=True,
    pde_executed=False,python=sys.version,zlib=zlib.ZLIB_RUNTIME_VERSION,
    numpy=numpy.__version__,scipy=scipy.__version__,shapely=shapely.__version__,
    skimage=skimage.__version__,opencv=cv2.__version__)
(ROOT/'bundle_upload/verified.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2),flush=True)
