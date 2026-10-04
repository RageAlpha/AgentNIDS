#!/usr/bin/env python3
"""Rebase the professor run copy after moving it. Changes paths only."""
import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = '/home/karri/projects/NIDSProject'
PATHS = ROOT / 'configs/run_paths.json'


def replace_root(text: str, old: str, new: str) -> str:
    # A boundary avoids changing e.g. NIDSProject_Professor_Run as an original root.
    return re.sub(re.escape(old) + r'(?=/|[\'"\s`])', lambda _: new, text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, help='directory holding the four original NetFlow v3 CSVs')
    parser.add_argument('--python', default=os.environ.get('NIDS_PYTHON', 'python3'), help='Python command recorded in active configs')
    parser.add_argument('--prepared-run', type=Path, help='fresh data-preparation output directory after rebuilding')
    parser.add_argument('--output-root', type=Path, help='directory for newly generated P0-P7 artifacts; post-P7 scripts still use project/outputs')
    args = parser.parse_args()
    previous = json.loads(PATHS.read_text()) if PATHS.exists() else {}
    previous_root = Path(previous.get('project_root', str(ROOT)))
    if args.data_root:
        data_root = args.data_root.expanduser().resolve()
    else:
        previous_data = Path(previous.get('data_root', ROOT / 'datasets')).expanduser()
        try:
            data_root = ROOT / previous_data.relative_to(previous_root)
        except ValueError:
            data_root = previous_data.resolve()
    if args.prepared_run:
        prepared_run = args.prepared_run.expanduser().resolve()
    else:
        previous_prepared = Path(previous.get('prepared_run', ROOT / 'outputs/round1_data_prep_v1'))
        try:
            prepared_run = ROOT / previous_prepared.relative_to(previous_root)
        except ValueError:
            prepared_run = previous_prepared
    if args.output_root:
        output_root = args.output_root.expanduser().resolve()
    else:
        previous_output = Path(previous.get('output_root', ROOT / 'outputs'))
        try:
            output_root = ROOT / previous_output.relative_to(previous_root)
        except ValueError:
            output_root = previous_output.resolve()
    old_roots = {ORIGINAL, previous.get('project_root', ORIGINAL)} - {str(ROOT)}
    changed = []
    for path in ROOT.rglob('*.py'):
        if path == Path(__file__).resolve() or 'archive_project_code' in path.parts:
            continue
        old_text = path.read_text(encoding='utf-8')
        new_text = old_text
        for old in sorted(old_roots, key=len, reverse=True):
            new_text = replace_root(new_text, old, str(ROOT))
        if new_text != old_text:
            path.write_text(new_text, encoding='utf-8')
            changed.append(str(path.relative_to(ROOT)))
    config_path = ROOT / 'configs/execution_v2.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    config.update(project_root=str(ROOT), data_root=str(data_root),
                  prepared_run=str(prepared_run), python=args.python)
    config['outputs'] = {key: str(output_root / Path(value).name) for key, value in config['outputs'].items()}
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    prep_path = ROOT / 'outputs/round1_data_prep_v1/round1_config.json'
    prep = json.loads(prep_path.read_text(encoding='utf-8'))
    prep['data_root'] = str(data_root)
    prep['execution_environment']['python'] = args.python
    prep_path.write_text(json.dumps(prep, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    PATHS.write_text(json.dumps(dict(project_root=str(ROOT), data_root=str(data_root),
                                    output_root=str(output_root), prepared_run=str(prepared_run), python=args.python), indent=2) + '\n')
    print(json.dumps(dict(project_root=str(ROOT), data_root=str(data_root),
                          output_root=str(output_root), prepared_run=str(prepared_run), python=args.python,
                          python_files_rebased=len(changed), changed_python=changed), indent=2))


if __name__ == '__main__':
    main()
