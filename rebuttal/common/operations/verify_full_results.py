"""Safely extract and verify every captured file from the full results archive."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time


def main():
    folder=Path(sys.argv[1]);manifest=json.loads((folder/'manifest.json').read_text())
    success=json.loads((folder/'SUCCESS.json').read_text())
    if hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest()!=success['manifest_sha256']:
        raise RuntimeError('Downloaded manifest hash mismatch')
    if shutil.disk_usage(folder).free < manifest['captured_bytes']+8*1024**3:
        raise RuntimeError('Insufficient local reserve for extraction')
    target=folder/'data';target.mkdir(exist_ok=False)
    expected={row['path']:row for row in manifest['files']};seen=set();total=0;manifest_seen=False
    digest=hashlib.sha256()
    with (folder/'results.tar.gz').open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
    if digest.hexdigest()!=success['archive_sha256']:
        raise RuntimeError('Downloaded archive hash mismatch')
    with tarfile.open(folder/'results.tar.gz','r|gz') as archive:
        for member in archive:
            parts=Path(member.name)
            if not member.isfile() or parts.is_absolute() or '..' in parts.parts:
                raise RuntimeError('Unsafe archive member: '+member.name)
            source=archive.extractfile(member)
            if member.name=='SNAPSHOT_MANIFEST.json':
                if hashlib.sha256(source.read()).hexdigest()!=success['manifest_sha256']:
                    raise RuntimeError('Embedded manifest mismatch')
                manifest_seen=True
                continue
            if member.name not in expected or member.name in seen:
                raise RuntimeError('Unexpected/duplicate archive member: '+member.name)
            row=expected[member.name];path=target/member.name;path.parent.mkdir(parents=True,exist_ok=True)
            sha=hashlib.sha256();size=0
            with path.open('xb') as output:
                for block in iter(lambda:source.read(1024*1024),b''):
                    output.write(block);sha.update(block);size+=len(block)
            if size!=row['bytes'] or sha.hexdigest()!=row['sha256']:
                raise RuntimeError('Extracted file integrity mismatch: '+member.name)
            path.chmod(member.mode & 0o777)
            os.utime(path,(member.mtime,member.mtime))
            seen.add(member.name);total+=size
            if len(seen)%100==0 or len(seen)==len(expected):
                print(json.dumps(dict(verified_files=len(seen),total_files=len(expected),verified_bytes=total)),flush=True)
    if seen!=set(expected) or not manifest_seen:raise RuntimeError('Archive missing declared files or embedded manifest')
    receipt=dict(at=time.time(),result='ok',verified_files=len(seen),verified_bytes=total,
        archive_sha256=success['archive_sha256'],manifest_sha256=success['manifest_sha256'],
        source=manifest['source'],destination=str(target.resolve()),
        consistency=manifest['consistency'],capture_started_at=manifest['started_at'],capture_finished_at=manifest['finished_at'])
    (folder/'LOCAL_VERIFIED.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':main()
