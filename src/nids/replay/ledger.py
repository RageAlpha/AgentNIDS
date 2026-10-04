from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .predictions import PREDICTION_DTYPE, write_json


def window_columns(windows: list[dict], rank: np.ndarray) -> dict:
    starts = np.asarray([w['rank_start'] for w in windows])
    ends = np.asarray([w['rank_end_exclusive'] for w in windows])
    indices = np.searchsorted(starts, rank, side='right')-1
    complete = (indices >= 0) & (rank < ends[np.maximum(indices,0)])
    selected = np.maximum(indices,0)
    window_id = pa.array([w['window_id'] for w in windows]).take(pa.array(selected, mask=~complete))
    previous_id = pa.array([w['previous_window_id'] for w in windows], type=pa.string()).take(pa.array(selected, mask=~complete))
    return dict(window_id=window_id, previous_window_id=previous_id,
                shift=pa.array(np.asarray([w['shift'] for w in windows])[selected],mask=~complete),
                initialization=pa.array(np.asarray([w['initialization'] for w in windows])[selected] & complete),
                complete_window=pa.array(complete), tail=pa.array(~complete))


def build_domain_ledger(config: dict, prepared, domain: str) -> dict:
    started = time.perf_counter()
    project = Path(config['project_root'])
    base = project / config['outputs']['round1'] / 'P3'
    windows = pq.read_table(project / config['outputs']['shift'] / '10k' / f'{domain}_windows.parquet').to_pylist()
    windows = [w for w in windows if w['scheme']=='development']
    caches = {mid: np.load(project / config['outputs']['base_predictions'] / domain / f'{mid}.npy',mmap_mode='r') for mid in config['models']['schedule']}
    order = prepared.sorted_rows(domain)
    available = prepared.sorted_end_ms(domain)
    partitions = []
    for name in ['R_fit','R_cal','R_select']:
        partition = prepared.partition(domain,'development',name)
        selected = sorted([w for w in windows if w['partition']==name],key=lambda w:w['rank_start'])
        output = base / 'development_shared' / f'domain={domain}' / f'partition={name}' / 'records.parquet'
        output.parent.mkdir(parents=True,exist_ok=True)
        total_rows, tails, initialized = 0,0,0
        writer = None
        for a in range(partition.rank_start,partition.rank_end_exclusive,config['batches']['inference_rows']):
            b = min(a+config['batches']['inference_rows'],partition.rank_end_exclusive)
            rank = np.arange(a,b,dtype=np.int64)
            original = np.asarray(order[a:b])
            columns = dict(record_id=pa.array([f'{domain}:{row}' for row in original]),rank=pa.array(rank),
                           original_row=pa.array(original), available_ms=pa.array(available[a:b]))
            columns.update(window_columns(selected,rank))
            for mid,cache in caches.items():
                for field in PREDICTION_DTYPE.names:
                    columns[f'{mid}__{field}'] = pa.array(cache[field][a:b])
            table = pa.table(columns)
            if writer is None:
                writer = pq.ParquetWriter(output,table.schema,compression='zstd',use_dictionary=['window_id','previous_window_id'])
            writer.write_table(table,row_group_size=config['batches']['inference_rows'])
            total_rows += b-a
            tails += int(np.count_nonzero(columns['tail'].to_numpy(zero_copy_only=False)))
            initialized += int(np.count_nonzero(columns['initialization'].to_numpy(zero_copy_only=False)))
        writer.close()
        folds = [fold for fold,roles in config['folds'].items() if domain in roles['development']]
        fold_paths = []
        for fold in folds:
            dest = base / 'development' / f'fold={fold}' / f'domain={domain}' / f'partition={name}' / 'records.parquet'
            dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():
                dest.unlink()  # Replace this generated link when rebuilding its owning domain.
            os.link(output,dest)
            fold_paths.append(str(dest.relative_to(project)))
        partitions.append(dict(domain=domain,partition=name,rows=total_rows,complete_windows=len(selected),
                               complete_window_rows=total_rows-tails,tail_rows=tails,initialization_rows=initialized,
                               file=str(output.relative_to(project)),bytes=output.stat().st_size,fold_files=fold_paths))
        print(f'{domain}/{name}: ledger {total_rows:,} rows, {len(selected):,} complete windows, tail={tails}',flush=True)
    result = dict(domain=domain,partitions=partitions,seconds=time.perf_counter()-started,
                  labels='absent; joined by domain/original_row in later fitting or evaluation',
                  storage='identical development rows shared by hard link; fold/domain/partition in Hive paths')
    write_json(base / f'{domain}_ledger_manifest.json',result)
    return result


def validate_domain_ledger(config: dict, prepared, domain: str) -> dict:
    started = time.perf_counter()
    project = Path(config['project_root'])
    base = project / config['outputs']['round1'] / 'P3'
    manifest = json.loads((base / f'{domain}_ledger_manifest.json').read_text())
    windows = pq.read_table(project / config['outputs']['shift'] / '10k' / f'{domain}_windows.parquet').to_pylist()
    windows = {w['window_id']:w for w in windows if w['scheme']=='development'}
    caches = {mid: np.load(project / config['outputs']['base_predictions'] / domain / f'{mid}.npy',mmap_mode='r') for mid in config['models']['schedule']}
    order, ends = prepared.sorted_rows(domain),prepared.sorted_end_ms(domain)
    checked = 0
    for row in manifest['partitions']:
        p = prepared.partition(domain,'development',row['partition'])
        file = project / row['file']
        cursor = p.rank_start
        for batch in pq.ParquetFile(file).iter_batches(batch_size=config['batches']['inference_rows']):
            table = pa.Table.from_batches([batch])
            n = len(table)
            rank = table['rank'].to_numpy()
            assert np.array_equal(rank,np.arange(cursor,cursor+n))
            assert np.array_equal(table['original_row'].to_numpy(),order[cursor:cursor+n])
            assert table['record_id'].to_pylist()==[f'{domain}:{r}' for r in order[cursor:cursor+n]]
            assert np.array_equal(table['available_ms'].to_numpy(),ends[cursor:cursor+n])
            for mid,cache in caches.items():
                for field in PREDICTION_DTYPE.names:
                    assert np.array_equal(table[f'{mid}__{field}'].to_numpy(),cache[field][cursor:cursor+n])
            ids = table['window_id'].combine_chunks().dictionary_encode()
            codes = ids.indices.to_numpy(zero_copy_only=False)
            saved_shift = table['shift'].to_numpy()
            saved_init = table['initialization'].to_numpy()
            saved_previous = np.asarray(table['previous_window_id'].to_pylist(),dtype=object)
            for code,window_id in enumerate(ids.dictionary.to_pylist()):
                selected = codes == code
                w = windows[window_id]
                assert np.all(rank[selected]>=w['rank_start']) and np.all(rank[selected]<w['rank_end_exclusive'])
                assert w['partition']==row['partition']
                assert np.all(saved_shift[selected]==w['shift'])
                assert np.all(saved_init[selected]==w['initialization'])
                assert all(v==w['previous_window_id'] for v in saved_previous[selected])
            tail = table['tail'].to_numpy()
            last_complete_end = max(w['rank_end_exclusive'] for w in windows.values() if w['partition']==row['partition'])
            assert np.array_equal(tail,rank>=last_complete_end)
            assert np.array_equal(tail,table['window_id'].is_null().to_numpy())
            assert np.array_equal(tail,table['shift'].is_null().to_numpy())
            assert np.array_equal(~tail,table['complete_window'].to_numpy())
            cursor += n
            checked += n
        assert cursor == p.rank_end_exclusive
        expected_folds = {f for f,roles in config['folds'].items() if domain in roles['development']}
        assert len(row['fold_files'])==len(expected_folds)
        for path in row['fold_files']:
            fold = next(part.split('=',1)[1] for part in Path(path).parts if part.startswith('fold='))
            assert fold in expected_folds and domain != config['folds'][fold]['heldout']
            assert os.path.samefile(file, project/path)
    result = dict(domain=domain,status='PASS',physical_rows_scanned=checked,fold_rows_checked=checked*2,
                  record_identity='all_rows',prediction_join='all_rows_all_30_fields',window_join='all_rows',
                  fold_role='frozen_config',seconds=time.perf_counter()-started)
    write_json(base/f'{domain}_ledger_validation.json',result)
    print(f'{domain}: ledger validation PASS, {checked:,} physical rows',flush=True)
    return result
