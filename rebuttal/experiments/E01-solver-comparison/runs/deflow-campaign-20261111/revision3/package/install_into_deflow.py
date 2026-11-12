"""Install the new package without overwriting existing repository files."""
from pathlib import Path
import argparse
import hashlib
import shutil

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--repo',type=Path,required=True)
p.add_argument('--move-original-screenshot',type=Path)
a=p.parse_args()
source=Path(__file__).resolve().parent
folder=a.repo.resolve()/'rebuttal'
destination=folder/'solver_tools'
picture=folder/'REVIEWS_ALL_2026-11-10.png'
if destination.exists() or picture.exists():
    raise SystemExit('Destination exists. Refusing overwrite.')
folder.mkdir(parents=True,exist_ok=True)
shutil.copytree(source,destination,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
shutil.copy2(source/'rebuttal'/picture.name,picture)
if a.move_original_screenshot:
    original=a.move_original_screenshot
    if hashlib.sha256(original.read_bytes()).digest()!=hashlib.sha256(picture.read_bytes()).digest():
        raise SystemExit('Screenshot mismatch. Original retained.')
    original.unlink()
print(destination)
print(picture)
