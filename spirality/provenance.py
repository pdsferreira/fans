"""Small, explicit provenance helpers; hashes do not imply statistical validity."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path


def file_digest(path: Path) -> str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def code_fingerprint(root: Path) -> str:
    h=hashlib.sha256()
    for folder in ("spirality","scripts"):
        for p in sorted((Path(root)/folder).glob("*.py")):
            h.update(str(p.relative_to(root)).encode()); h.update(p.read_bytes())
    return h.hexdigest()


def environment_versions() -> dict:
    versions={"python":platform.python_version(),"platform":platform.platform()}
    for name in ("numpy","pandas","scipy","matplotlib","astropy","healpy"):
        try: versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name]="not installed"
    return versions


def input_metadata(paths, *, hash_contents: bool=False):
    rows=[]
    for p in map(Path,paths):
        st=p.stat(); row={"path":str(p.resolve()),"size_bytes":st.st_size,"mtime_ns":st.st_mtime_ns}
        if hash_contents: row["sha256"]=file_digest(p)
        rows.append(row)
    return rows
