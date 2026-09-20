#!/usr/bin/env python3
"""Verify every v8.0.0 source file against the embedded original SHA256 manifest."""
from __future__ import annotations
import argparse,difflib,hashlib,json,zipfile
from pathlib import Path

def sha(b):return hashlib.sha256(b).hexdigest()
def main():
    root=Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline-zip',type=Path)
    p.add_argument('--output-dir',type=Path,default=root/'docs/v8_1_validation/preservation')
    args=p.parse_args();ref=json.loads((root/'docs/V800_BASELINE_MANIFEST.json').read_text())
    allowed=json.loads((root/'docs/V810_EXPECTED_CHANGES.json').read_text())
    baseline_bytes={}
    if args.baseline_zip:
        with zipfile.ZipFile(args.baseline_zip) as archive:
            for name in archive.namelist():
                if name.endswith('/'):continue
                rel=name.split('/',1)[1];baseline_bytes[rel]=archive.read(name)
        if {k:sha(v) for k,v in baseline_bytes.items()}!=ref['files']:
            raise SystemExit('Provided ZIP does not match the release input manifest')
    same=[];changed=[];missing=[];unexpected=[]
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for rel,before in sorted(ref['files'].items()):
        path=root/rel
        if not path.is_file():missing.append(rel);continue
        after=sha(path.read_bytes())
        if after==before:same.append(rel)
        else:
            row={'path':rel,'sha256_before':before,'sha256_after':after,'reason':allowed.get(rel)};changed.append(row)
            if rel not in allowed:unexpected.append(rel)
            if rel in baseline_bytes:
                try:
                    a=baseline_bytes[rel].decode().splitlines(True);b=path.read_text().splitlines(True)
                    diff=''.join(difflib.unified_diff(a,b,fromfile='v8.0.0/'+rel,tofile='v8.1.0/'+rel))
                    dest=args.output_dir/'diffs'/(rel+'.patch');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(diff)
                except UnicodeError:pass
    report={'status':'passed' if not missing and not unexpected else 'failed',
            'baseline_zip_sha256':ref['archive_sha256'],'baseline_file_count':len(ref['files']),
            'byte_identical_count':len(same),'modified_count':len(changed),'deleted_count':len(missing),
            'unexpected_modified_count':len(unexpected),'modified':changed,'deleted':missing,
            'unexpected_modified':unexpected,'byte_identical_files':same,
            'meaning':'Hash preservation checks file identity; functional/numerical equivalence is tested separately.'}
    (args.output_dir/'audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    lines=['# v8.0.0 파일 보존 감사','',f"기준 파일 {report['baseline_file_count']}개 / 동일 {len(same)}개 / 수정 {len(changed)}개 / 삭제 {len(missing)}개",'',
           '| 파일 | 변경 사유 |','|---|---|']
    lines += [f"| `{x['path']}` | {x['reason']} |" for x in changed]
    lines+=['','해시는 파일 보존을 검증합니다. 기능·수치 동등성은 회귀시험과 parity 결과를 별도로 확인해야 합니다.']
    (args.output_dir/'audit.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('byte_identical_files','modified')},indent=2));return int(report['status']!='passed')
if __name__=='__main__':raise SystemExit(main())
