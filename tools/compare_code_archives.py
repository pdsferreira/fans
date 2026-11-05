#!/usr/bin/env python3
"""Compare two source ZIPs by bytes and top-level AST (standard library only)."""
from __future__ import annotations
import argparse
import ast
import difflib
import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath


def sources(path):
    result={}
    with zipfile.ZipFile(path) as z:
        entries=[i for i in z.infolist() if not i.is_dir() and '__pycache__' not in i.filename and '__MACOSX' not in i.filename]
        roots={PurePosixPath(i.filename).parts[0] for i in entries}
        strip=len(roots)==1 and all(len(PurePosixPath(i.filename).parts)>1 for i in entries)
        for i in entries:
            parts=PurePosixPath(i.filename).parts
            name='/'.join(parts[1:] if strip else parts)
            if name in result:raise ValueError(f'Duplicate archive member {name}')
            result[name]=z.read(i)
    return result


def functions(data):
    tree=ast.parse(data.decode('utf-8'))
    return {n.name:(ast.dump(n,include_attributes=False),n.lineno,n.end_lineno) for n in tree.body
            if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))}


def compare(first,second,outdir):
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True);a=sources(first);b=sources(second);rows=[]
    for name in sorted(set(a)|set(b)):
        status='added' if name not in a else 'removed' if name not in b else 'identical' if a[name]==b[name] else 'changed'
        row={'file':name,'status':status,'old_sha256':hashlib.sha256(a[name]).hexdigest() if name in a else None,
             'new_sha256':hashlib.sha256(b[name]).hexdigest() if name in b else None}
        if status=='changed' and name.endswith('.py'):
            fa,fb=functions(a[name]),functions(b[name])
            row['changed_functions']=[{'name':k,'old_lines':fa[k][1:],'new_lines':fb[k][1:]} for k in sorted(fa.keys()&fb.keys()) if fa[k][0]!=fb[k][0]]
            row['added_functions']=sorted(fb.keys()-fa.keys());row['removed_functions']=sorted(fa.keys()-fb.keys())
            text=''.join(difflib.unified_diff(a[name].decode().splitlines(True),b[name].decode().splitlines(True),fromfile='old/'+name,tofile='new/'+name))
            (out/(name.replace('/','__')+'.diff')).write_text(text)
        rows.append(row)
    report={'old_archive':str(Path(first).resolve()),'new_archive':str(Path(second).resolve()),
            'old_archive_sha256':hashlib.sha256(Path(first).read_bytes()).hexdigest(),
            'new_archive_sha256':hashlib.sha256(Path(second).read_bytes()).hexdigest(),
            'note':'AST comparisons remove comments/formatting, not docstrings. This is not an observational rerun.', 'files':rows}
    (out/'source_comparison.json').write_text(json.dumps(report,indent=2))
    import csv
    with (out/'source_comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['file','status','old_sha256','new_sha256']);w.writeheader()
        w.writerows({k:r[k] for k in w.fieldnames} for r in rows)
    print(f"{sum(r['status']=='changed' and r['file'].endswith('.py') for r in rows)} changed Python files; output: {out}")

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('old_zip');p.add_argument('new_zip');p.add_argument('--outdir',required=True)
    a=p.parse_args();compare(a.old_zip,a.new_zip,a.outdir)
