"""Restore the checked-in model bundle after preserving local versions in a ZIP."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import subprocess
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]
NAMES=('model.pt','metadata.json','manifest.json')

def restore():
    blobs={name:subprocess.run(['git','show','HEAD:artifacts/'+name],cwd=ROOT,check=True,stdout=subprocess.PIPE).stdout for name in NAMES}
    manifest=json.loads(blobs['manifest.json'])
    for name in NAMES[:2]:
        content=blobs[name].replace(b'\r\n',b'\n') if name.endswith('.json') else blobs[name]
        if hashlib.sha256(content).hexdigest()!=manifest[name]:
            raise ValueError('The committed model bundle does not match its manifest; nothing was restored.')
    backups=ROOT/'.local_backups';backups.mkdir(exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backup=backups/('artifacts-'+stamp+'.zip')
    with zipfile.ZipFile(backup,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name in NAMES:
            path=ROOT/'artifacts'/name
            if path.exists():
                archive.write(path,'artifacts/'+name)
    with zipfile.ZipFile(backup) as archive:
        if archive.testzip() is not None:
            raise ValueError('Backup verification failed; nothing was restored.')
    (ROOT/'artifacts').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as directory:
        staged=Path(directory)
        for name,content in blobs.items():
            (staged/name).write_bytes(content)
        for name in NAMES:
            (staged/name).replace(ROOT/'artifacts'/name)
    print('Preserved local artifacts:',backup)
    print('Restored the complete bundle from the current local Git commit. Restart Streamlit.')

if __name__=='__main__':
    restore()
