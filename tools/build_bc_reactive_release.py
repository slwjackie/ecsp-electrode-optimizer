#!/usr/bin/env python3
"""Package v8.4 from an exactly bound, passing CPU test source tree.

Historical builders/manifests remain in the payload and are not authoritative
for v8.4. Runtime caches/native binaries are never distributed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile

ROOT=Path(__file__).resolve().parents[1]
NAME='ECSP_v8_4_0_BCReactive_Condensed'
VERSION='8.4.0-BCReactive-Condensed-Experimental'
MANIFEST='PACKAGE_SHA256_MANIFEST_V8_4.txt'
ALIAS='PACKAGE_SHA256_MANIFEST.txt'
EXCLUDE={'.git','.pytest_cache','.ruff_cache','__pycache__','.native_build','build','runs'}
REGENERATE={MANIFEST,ALIAS,'PACKAGE_MANIFEST.txt','PACKAGE_BUILD_METADATA_V8_4_0.json'}
SOURCE_SUFFIX={'.py','.cpp','.cu','.cuh','.h','.hpp','.cc','.c','.sh','.yaml','.yml','.toml'}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def included(relative):
    return (not any(p in EXCLUDE or p.startswith(('tmp_','.tmp_')) for p in relative.parts)
            and relative.suffix not in {'.pyc','.pyo'} and relative.name!='.DS_Store')


def source_binding(root):
    result={}
    for f in sorted(root.rglob('*')):
        r=f.relative_to(root)
        if not included(r):continue
        if f.is_symlink():raise ValueError(f'Unsupported symlink: {r}')
        if (f.is_file() and (r.parts[0] in {'python','cpp','tools','config'} and f.suffix in SOURCE_SUFFIX
                            or str(r) in {'VERSION.json','pytest.ini','RUN_NSGA2_ONLY.sh'})):
            result[str(r)]=sha(f)
    return result


def junit_counts(path):
    tree=ET.parse(path)
    cases=tree.findall('.//testcase')
    failures=sum(c.find('failure') is not None or c.find('error') is not None for c in cases)
    skipped=sum(c.find('skipped') is not None for c in cases)
    if not cases or failures:raise ValueError(f'Missing/failing JUnit evidence: {path}')
    return {'tests':len(cases),'passed':len(cases)-skipped,'skipped':skipped,'failures':failures}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT)
    p.add_argument('--record-source',type=Path)
    p.add_argument('--validation-dir',type=Path)
    p.add_argument('--output-root',type=Path)
    a=p.parse_args();source=a.source.resolve()
    if a.record_source:
        a.record_source.parent.mkdir(parents=True,exist_ok=True)
        a.record_source.write_text(json.dumps(source_binding(source),indent=2)+'\n')
        return
    if not a.validation_dir or not a.output_root:p.error('--validation-dir and --output-root are required')
    evidence=a.validation_dir.resolve()
    expected=json.loads((evidence/'validated_source_sha256.json').read_text())
    if expected!=source_binding(source):raise ValueError('Source changed after the recorded test binding')
    counts={k:junit_counts(evidence/(k+'.xml')) for k in ('full_suite','bc_reactive_suite')}
    if counts['full_suite']['tests']<500 or counts['bc_reactive_suite']['passed']<43:
        raise ValueError('Evidence does not cover the required full and new integration suites')
    version=json.loads((source/'VERSION.json').read_text())
    if version['version']!=VERSION or version['release_directory_name']!=NAME:
        raise ValueError('Wrong release version/root')
    output=a.output_root.resolve();output.mkdir(parents=True,exist_ok=True)
    archive=output/(NAME+'.zip');checksum=output/(NAME+'.zip.sha256.txt')
    verification=output/(NAME+'_ARCHIVE_VERIFICATION.json')
    if any(f.exists() for f in (archive,checksum,verification)):
        raise ValueError('Refusing to overwrite an existing immutable release')
    sys.path.insert(0,str(source/'python'))
    from verify_release_manifest import verify
    with tempfile.TemporaryDirectory(prefix='ecsp-v840-build-',dir=output) as d:
        temp=Path(d);root=temp/NAME;root.mkdir()
        for f in sorted(source.rglob('*')):
            r=f.relative_to(source)
            if not included(r) or str(r) in REGENERATE:continue
            if f.is_symlink():raise ValueError(f'Symlink in payload: {r}')
            if f.is_file():
                dst=root/r;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dst)
        target_evidence=root/'docs/validation_v8_4_0'
        target_evidence.mkdir(parents=True,exist_ok=True)
        for f in evidence.rglob('*'):
            if f.is_symlink():raise ValueError('Symlink in validation evidence')
            if f.is_file():
                dst=target_evidence/f.relative_to(evidence);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dst)
        metadata={'release':VERSION,'test_counts':counts,'bound_source_file_count':len(expected),
                  'experimental_validation':False,'cuda_hardware_tested':False,
                  'new_reactive_backend':'CPU FP64','historical_manifests':'retained_not_authoritative',
                  'authoritative_manifest':MANIFEST}
        (root/'PACKAGE_BUILD_METADATA_V8_4_0.json').write_text(json.dumps(metadata,indent=2)+'\n')
        inventory=sorted({str(f.relative_to(root)) for f in root.rglob('*') if f.is_file()}
                          |{MANIFEST,ALIAS,'PACKAGE_MANIFEST.txt'})
        (root/'PACKAGE_MANIFEST.txt').write_text('\n'.join(inventory)+'\n')
        lines=[f'{sha(f)}  {f.relative_to(root).as_posix()}' for f in sorted(root.rglob('*'))
               if f.is_file() and f.name not in {MANIFEST,ALIAS}]
        payload='\n'.join(lines)+'\n'
        (root/MANIFEST).write_text(payload);(root/ALIAS).write_text(payload)
        before=verify(root,root/MANIFEST)
        if before['status']!='passed':raise ValueError(before)
        zpath=temp/(NAME+'.zip')
        with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
            for f in sorted(root.rglob('*')):
                if f.is_file():z.write(f,(Path(NAME)/f.relative_to(root)).as_posix())
        with zipfile.ZipFile(zpath) as z:
            bad=z.testzip()
            if bad:raise ValueError('ZIP CRC failure: '+bad)
            z.extractall(temp/'fresh')
        fresh=verify(temp/'fresh'/NAME,temp/'fresh'/NAME/MANIFEST)
        if fresh['status']!='passed':raise ValueError(fresh)
        if source_binding(temp/'fresh'/NAME)!=expected:raise ValueError('Fresh archive source differs from tested source')
        report={**fresh,'zip_crc':'passed','fresh_extract':'passed','test_counts':counts,
                'source_binding_matches_tested_source':True,'archive_sha256':sha(zpath)}
        os.replace(zpath,archive)
        checksum.write_text(f'{report["archive_sha256"]}  {archive.name}\n')
        verification.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'archive':str(archive),**report},indent=2))

if __name__=='__main__':main()
