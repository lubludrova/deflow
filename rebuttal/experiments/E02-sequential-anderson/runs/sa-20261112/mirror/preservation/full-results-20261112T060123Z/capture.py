"""Preserve a per-file, SHA-256 verified snapshot without pausing training."""
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time
import zipfile

FOLDERS = ('results', 'evaluations', 'audits', 'preflight', 'validation',
           'ops', 'package', 'DEFlow', 'reviews')
SKIP_DIRS = {'__pycache__', '.pytest_cache', '.git', '.ssh'}
SKIP_FILES = {'.env', 'id_rsa', 'id_ed25519'}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


class PrefixReader:
    def __init__(self, stream, size):
        self.stream, self.remaining = stream, size
        self.sha = hashlib.sha256()

    def read(self, size=-1):
        count = self.remaining if size < 0 else min(size, self.remaining)
        data = self.stream.read(count)
        if not data and self.remaining:
            raise RuntimeError('Source truncated during prefix capture')
        self.remaining -= len(data)
        self.sha.update(data)
        return data


def inventory(root):
    skipped = []
    files = []
    for path in root.iterdir():
        if path.is_file():
            if path.is_symlink() or path.name.endswith(('.tmp','.lock')) or path.name in SKIP_FILES:
                skipped.append(str(path.relative_to(root)))
            else:files.append(path)
    for folder in FOLDERS:
        for base, dirs, names in os.walk(root/folder, followlinks=False):
            dirs[:] = sorted(name for name in dirs if name not in SKIP_DIRS
                             and not (Path(base)/name).is_symlink())
            for name in sorted(names):
                path = Path(base)/name
                if path.is_symlink() or name.endswith(('.tmp','.lock')) or name in SKIP_FILES:
                    skipped.append(str(path.relative_to(root)))
                    continue
                if path.is_file():
                    files.append(path)
    return [(path, path.stat()) for path in sorted(files)], skipped


def capture(root, destination, binding):
    root, destination = Path(root), Path(destination)
    files, skipped = inventory(root)
    total = sum(stat.st_size for _, stat in files)
    if shutil.disk_usage(root).free < total + 30*1024**3:
        raise RuntimeError('Insufficient remote disk reserve for preservation')
    manifest = dict(binding, started_at=time.time(), source=str(root),
        scope=list(FOLDERS)+['root regular files'], selected_files=len(files),
        initial_selected_bytes=total, files=[], skipped=skipped,
        excluded_environment='venv, caches, git internals, locks and temporary files',
        consistency='Independent file captures, not a globally atomic snapshot. Append-only logs/events are bounded byte prefixes. New files after inventory are not part of this snapshot.')
    partial = destination/'results.tar.gz.partial'
    copied = 0
    with tarfile.open(partial, 'w:gz', compresslevel=1, format=tarfile.PAX_FORMAT) as archive:
        for index, (path, original) in enumerate(files):
            relative = str(path.relative_to(root))
            append = path.suffix in ('.jsonl','.log') or 'tfevents' in path.name
            observed = time.time()
            record = dict(path=relative, observed_at=observed, append_prefix=append,
                inventory_size=original.st_size, inventory_mtime_ns=original.st_mtime_ns)
            with path.open('rb') as stream:
                before = os.fstat(stream.fileno())
                info = tarfile.TarInfo(relative)
                info.mode, info.mtime = original.st_mode & 0o777, original.st_mtime
                if append:
                    size = original.st_size
                    if before.st_size < size:
                        raise RuntimeError('Append-only source became shorter: '+relative)
                    reader = PrefixReader(stream,size)
                    info.size=size
                    archive.addfile(info,reader)
                    record.update(bytes=size,sha256=reader.sha.hexdigest())
                else:
                    data=stream.read()
                    after=os.fstat(stream.fileno())
                    # Atomic replacements keep an already-open inode stable. Retry in-place writers.
                    for _ in range(4):
                        if before.st_size==after.st_size and before.st_mtime_ns==after.st_mtime_ns:
                            break
                        time.sleep(.2)
                        with path.open('rb') as retry:
                            before=os.fstat(retry.fileno());data=retry.read();after=os.fstat(retry.fileno())
                    record['changed_during_read']=before.st_size!=after.st_size or before.st_mtime_ns!=after.st_mtime_ns
                    if record['changed_during_read']:
                        raise RuntimeError('Could not capture stable mutable file: '+relative)
                    if path.suffix=='.json':
                        try:json.loads(data);record['json_valid']=True
                        except (ValueError,UnicodeDecodeError):record['json_valid']=False
                    if path.suffix=='.pt':
                        try:
                            with zipfile.ZipFile(io.BytesIO(data)) as package:
                                record['torch_zip_crc_valid']=package.testzip() is None
                        except zipfile.BadZipFile:
                            record['torch_zip_crc_valid']=False
                    info.size=len(data)
                    archive.addfile(info,io.BytesIO(data))
                    record.update(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
            manifest['files'].append(record)
            copied+=record['bytes']
            if index%50==0 or index+1==len(files):
                progress=dict(at=time.time(),files=index+1,total_files=len(files),bytes=copied,
                              initial_total_bytes=total,path=relative)
                (destination/'progress.json').write_text(json.dumps(progress)+'\n')
                print(json.dumps(progress),flush=True)
        manifest.update(finished_at=time.time(),captured_bytes=copied)
        encoded=(json.dumps(manifest,indent=2,allow_nan=False)+'\n').encode()
        info=tarfile.TarInfo('SNAPSHOT_MANIFEST.json');info.size=len(encoded)
        archive.addfile(info,io.BytesIO(encoded))
    final=destination/'results.tar.gz'
    os.replace(partial,final)
    (destination/'manifest.json').write_bytes(encoded)
    receipt=dict(binding,finished_at=time.time(),files=len(files),captured_bytes=copied,
        archive_bytes=final.stat().st_size,archive_sha256=digest(final),
        manifest_sha256=hashlib.sha256(encoded).hexdigest(),
        json_parse_failures=[r['path'] for r in manifest['files'] if r.get('json_valid') is False],
        torch_zip_failures=[r['path'] for r in manifest['files'] if r.get('torch_zip_crc_valid') is False])
    (destination/'SUCCESS.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def main():
    root,destination=map(Path,sys.argv[1:3])
    sys.path.insert(0,str(root/'ops'))
    from common import contract_check,identity
    contract=contract_check()
    os.nice(10)
    binding=dict(campaign_id=contract['campaign_id'],contract_sha256=os.environ['DEFLOW_CONTRACT_SHA256'],
        capture_source_sha256=digest(__file__))
    (destination/'START.json').write_text(json.dumps(dict(binding,at=time.time(),identity=identity(os.getpid())),indent=2)+'\n')
    try:print(json.dumps(capture(root,destination,binding)),flush=True)
    except BaseException:
        import traceback
        (destination/'FAILED.json').write_text(json.dumps(dict(at=time.time(),traceback=traceback.format_exc()),indent=2)+'\n')
        raise


if __name__=='__main__':main()
