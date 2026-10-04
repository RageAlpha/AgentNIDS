#!/usr/bin/env python3
"""Prepare an isolated legacy-layout workspace from the organized research sources.

No scientific source is modified in the review tree. Absolute source-root literals
and active config directory fields are rebased only in the generated workspace.
Saved protocol JSON is preserved; preparing a directory does not validate resume.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'configs/package_layout.json'
EXAMPLE_RUN = 'outputs/post_p7_diagnostics/ton_local_action_correction_v1/run_20260929T105356Z_w5yzUa'
EVIDENCE_RUN = 'outputs/post_p7_diagnostics/reliability_local_evidence_v1/run_20260929T074336Z_8Aw2gr'
EXAMPLE_INPUTS = [EVIDENCE_RUN + '/calibration/' + name for name in (
    'frozen_rules.json', 'ToN_reference.npz', 'ToN_selection_evidence.npz', 'sample_roles.csv')]

def read_manifest():
    return json.loads(MANIFEST.read_text(encoding='utf-8'))

def safe_join(base, relative):
    path = (base / relative).resolve()
    if not path.is_relative_to(base.resolve()):
        raise ValueError('Path escapes its root: ' + relative)
    return path

def history_root(manifest, argument=None):
    return Path(argument).expanduser().resolve() if argument else ROOT.parent / manifest['history_directory']

def rebase(text, manifest, work):
    for old in sorted(set(manifest['legacy_roots']), key=len, reverse=True):
        # Forward slashes keep generated Python literals valid on Windows and POSIX.
        text = re.sub(re.escape(old) + r'(?=/|[\x27\x22\s`])', lambda _: work.as_posix(), text)
    return text

def copy_file(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)

def prepare(args, manifest, example=False):
    experiments={x['id']:x for x in manifest['experiments']}
    exp_id = 'example' if example else args.experiment
    if not example and exp_id != 'main' and exp_id not in experiments:
        raise ValueError('Unknown experiment: ' + exp_id)
    target_run = EXAMPLE_RUN if example else experiments[exp_id]['original_run'] if exp_id != 'main' else None
    work = Path(args.workspace).expanduser().resolve() if args.workspace else ROOT/'runs'/(exp_id+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    hist=history_root(manifest,getattr(args,'history',None)).resolve()
    if work.exists(): raise ValueError('Workspace must not exist; use a fresh path: '+str(work))
    if work.is_relative_to(ROOT.resolve()) and not work.is_relative_to((ROOT/'runs').resolve()):
        raise ValueError('Within the review package, workspaces must be under runs/')
    if work.is_relative_to(hist) or ROOT.resolve().is_relative_to(work):
        raise ValueError('Workspace overlaps the archive or review sources')
    with_history = bool(getattr(args,'with_history',False))
    if with_history and exp_id=='main':
        raise ValueError('Main rebuild starts fresh; do not seed it with old completed outputs')
    if with_history and not hist.is_dir(): raise FileNotFoundError('History package not found: '+str(hist))
    by_original={x['original']:x for x in manifest['files']}
    for name in EXAMPLE_INPUTS if example else []:
        item=by_original[name]
        if not item['current'] or not safe_join(ROOT,item['current']).is_file():
            raise FileNotFoundError('Required example input missing: '+name)
    # Plan every copy first. No historical file is copied into the target run.
    plan={}
    for item in manifest['files']:
        rel=item['original']; role=item['role']; current=item['current']
        target_file=bool(target_run and (rel==target_run or rel.startswith(target_run+'/')))
        include=role in ('source','configuration','documentation')
        if role=='input':
            if example: include=rel in EXAMPLE_INPUTS
            elif exp_id=='main': include=rel.startswith('outputs/round1_data_prep_v1/')
            else: include=not target_file
        if include and current:
            plan[rel]=safe_join(ROOT,current)
        elif with_history and not target_file and current is None:
            source=safe_join(hist,rel)
            if source.is_file(): plan[rel]=source
    # Mapped files already have an original execution destination, including renamed files.
    mapped_sources = {safe_join(ROOT, item['current']).resolve() for item in manifest['files']
                      if item['current'] and item['role'] == 'source'}
    # Include new modules added by the professor under the existing source trees.
    source_trees = [('src','src'),('tests','tests'),('experiments','outputs/post_p7_diagnostics'),
                    ('tools/data_preparation','outputs/round1_data_prep_v1'),('tools/legacy/scripts','scripts')]
    for current_base,original_base in source_trees:
        tree=ROOT/current_base
        if tree.exists():
            for path in tree.rglob('*'):
                if path.is_file() and path.suffix in ('.py','.sh') and '__pycache__' not in path.parts:
                    if path.resolve() in mapped_sources: continue
                    if not path.resolve().is_relative_to(ROOT): raise ValueError('Source symlink leaves package: '+str(path))
                    plan[original_base+'/'+path.relative_to(tree).as_posix()]=path
    work.mkdir(parents=True)
    for rel,source in plan.items(): copy_file(source,safe_join(work,rel))
    changed=[]
    for relative in plan:
        if Path(relative).suffix not in ('.py','.sh'):continue
        path=safe_join(work,relative)
        before=path.read_text(encoding='utf-8-sig');after=rebase(before,manifest,work)
        if after!=before:
            path.write_text(after,encoding='utf-8');changed.append(relative)
    data=Path(args.data_root).expanduser().resolve() if getattr(args,'data_root',None) else work/'datasets'
    config_path=work/'configs/execution_v2.json'
    config=json.loads(config_path.read_text(encoding='utf-8'))
    config.update(project_root=work.as_posix(),data_root=data.as_posix(),prepared_run=(work/'outputs/round1_data_prep_v1').as_posix(),python=sys.executable)
    config['outputs']={k:(work/'outputs'/Path(v.replace('\\','/')).name).as_posix() for k,v in config['outputs'].items()}
    config_path.write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    prep=work/'outputs/round1_data_prep_v1/round1_config.json'
    if prep.exists():
        c=json.loads(prep.read_text(encoding='utf-8'));c['data_root']=data.as_posix();c['execution_environment']['python']=sys.executable
        prep.write_text(json.dumps(c,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (work/'configs/run_paths.json').write_text(json.dumps(dict(project_root=work.as_posix(),data_root=data.as_posix(),prepared_run=config['prepared_run'],output_root=(work/'outputs').as_posix(),python=sys.executable),indent=2)+'\n',encoding='utf-8')
    if target_run:
        for folder in ('src','config','calibration','reports','logs','tables','models','checks','evaluation'):
            (work/target_run/folder).mkdir(parents=True,exist_ok=True)
    record=dict(experiment=exp_id,workspace=str(work),target_run=target_run,copied_files=len(plan),copied_original_paths=sorted(plan),source_path_rebases=changed,
      history_used=with_history,history_directory=str(hist) if with_history else None,
      limitations='Prepared layout only; large assets, target preregistration and protocol compatibility require review. Original target outputs were not copied. Historical protocol contents were not rewritten.')
    (work/'WORKSPACE_PREPARATION.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    return work,record

def example(args,manifest):
    work,record=prepare(args,manifest,example=True)
    script=work/EXAMPLE_RUN/'src/run_calibration.py'
    command=[sys.executable,str(script),'--source-run',str(work/EVIDENCE_RUN),'--run-dir',str(work/EXAMPLE_RUN)]
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8',NIDS_PYTHON=sys.executable)
    env['PYTHONPATH']=str(work/'src')
    log=work/EXAMPLE_RUN/'logs/execution.log'
    with log.open('w',encoding='utf-8') as stream:
        proc=subprocess.run(command,cwd=work,env=env,stdout=stream,stderr=subprocess.STDOUT)
    validation={'command':command,'returncode':proc.returncode,'log':str(log),'kind':'real calibration on reused 200R/200C labels; not an outer evaluation','status':'FAIL'}
    if proc.returncode==0:
        result=json.loads((work/EXAMPLE_RUN/'calibration/selected_rules.json').read_text(encoding='utf-8'))
        rule=result['methods']['E1-local'];validation['selected_rule']=rule
        validation['status']='PASS' if (rule['B']['accepted'],rule['B']['errors'],rule['A']['accepted'],rule['A']['errors'])==(52,2,13,0) else 'REFERENCE_MISMATCH'
    (work/'EXAMPLE_VALIDATION.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(workspace=str(work),status=validation['status'],log=str(log)),ensure_ascii=False,indent=2))
    return 0 if validation['status']=='PASS' else 1

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('list',help='Show research runs and source/input counts')
    p=sub.add_parser('prepare',help='Create a new workspace; does not execute an experiment')
    p.add_argument('--experiment',required=True,help='main or E01..E29')
    p.add_argument('--workspace');p.add_argument('--data-root');p.add_argument('--with-history',action='store_true',help='Include archived metadata of OTHER runs for diagnostics; omitted large assets remain missing')
    p.add_argument('--history',help='Path to the separate historical package')
    p=sub.add_parser('example',help='Run the real ToN calibration example using retained compact inputs')
    p.add_argument('--workspace')
    args=parser.parse_args();manifest=read_manifest()
    if args.command=='list':
        for e in manifest['experiments']:
            print(f"{e['id']}  {e['title']}  ({len(e['source_files'])} source/commands, {len(e['retained_inputs'])} retained inputs)")
        return 0
    if args.command=='example':return example(args,manifest)
    work,record=prepare(args,manifest)
    print(json.dumps(dict(workspace=str(work),target_run=record['target_run'],copied_files=record['copied_files'],status='PREPARED_NOT_EXECUTED',limitations=record['limitations']),ensure_ascii=False,indent=2))
    return 0

if __name__=='__main__':
    try:sys.exit(main())
    except (ValueError,FileNotFoundError) as error:
        print('ERROR: '+str(error),file=sys.stderr);sys.exit(2)
