from __future__ import annotations

import gc
import importlib.metadata
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cupy as cp
import numpy as np
import pyarrow.parquet as pq
import torch

from .models.base_models import MODEL_SPECS, load_model
from .replay.predictions import PREDICTION_DTYPE, gpu_raw_probability, gpu_evidence, load_preprocessor, predict_domain, write_json
from .replay.shift import compute_domain_shift
from .replay.ledger import build_domain_ledger, validate_domain_ledger


def upstream_inventory(config: dict) -> list[dict]:
    project = Path(config['project_root'])
    paths = [project/'configs/execution_v2.json',project/config['feature_views']]
    for root in [project/config['outputs']['feature_store'],project/config['outputs']['source'],Path(config['prepared_run'])]:
        paths.extend(p for p in root.rglob('*') if p.is_file())
    return [dict(path=str(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in sorted(paths)]


def validate_predictions(config: dict, prepared, domain: str) -> dict:
    started = time.perf_counter()
    project = Path(config['project_root'])
    root = project/config['outputs']['base_predictions']/domain
    source = project/config['outputs']['source']
    values = np.load(project/config['outputs']['feature_store']/f'{domain}_X.npy',mmap_mode='r')
    preprocessor = load_preprocessor(config)
    manifest = json.loads((root/'manifest.json').read_text())
    assert len(values)==manifest['rows']==len(prepared.sorted_rows(domain))
    # Fixed rank locations cover start, end and each prepared partition boundary.
    positions = {0,len(values)-1,len(values)//2}
    for p in prepared.splits:
        if p['domain']==domain:
            positions.update([int(p['rank_start']),int(p['rank_end_exclusive'])-1])
    positions = sorted(positions)
    reports = []
    for mid in config['models']['schedule']:
        cache = np.load(root/f'{mid}.npy',mmap_mode='r')
        assert len(cache)==len(values) and cache.dtype==PREDICTION_DTYPE
        calibrator = json.loads((source/'calibrators'/f'{mid}.json').read_text())
        competence = json.loads((source/'competence'/f'{mid}.json').read_text())['table']
        h = cp.asarray([v['h'] for v in sorted(competence,key=lambda v:v['predicted_class'])])
        max_error = np.zeros(4)
        for a in range(0,len(cache),config['batches']['inference_rows']):
            b = min(a+config['batches']['inference_rows'],len(cache))
            chunk = cache[a:b]
            raw = cp.asarray(chunk['raw_probability'])
            expected = gpu_evidence(raw,calibrator,h,config['competence']['attack_threshold'])
            for k,field in enumerate(['probability','predicted_class','u','h']):
                saved = cp.asarray(chunk[field])
                assert bool(cp.isfinite(saved).all()) and float(saved.min())>=0 and float(saved.max())<=1
                error = float(cp.max(cp.abs(saved.astype(cp.float64)-expected[k])))
                assert error <= 2e-14
                max_error[k] = max(max_error[k],error)
            assert bool(cp.isfinite(raw).all()) and float(raw.min())>=0 and float(raw.max())<=1
        spec = MODEL_SPECS[mid]
        training = json.loads((source/'models'/mid/'training_manifest.json').read_text())
        model = load_model(spec,source/'models'/mid/training['artifact'],'cuda:0')
        batch_starts = sorted({rank//config['batches']['inference_rows']*config['batches']['inference_rows'] for rank in positions})
        difference = 0.
        for batch_start in batch_starts:
            batch_end = min(batch_start+config['batches']['inference_rows'],len(values))
            x = cp.ascontiguousarray(preprocessor.transform(values[batch_start:batch_end],spec.standardized)[:,training['feature_indices']])
            actual = cp.asnumpy(gpu_raw_probability(model,spec,x))
            difference = max(difference,float(np.max(np.abs(actual-cache['raw_probability'][batch_start:batch_end]))))
            assert np.array_equal(actual,cache['raw_probability'][batch_start:batch_end])
        reports.append(dict(model_id=mid,rows=len(cache),ranges_and_evidence='PASS',
                            evidence_max_error=max_error.tolist(),identity_reprediction_ranks=positions,
                            identity_reprediction_batch_starts=batch_starts,
                            identity_reprediction_max_error=difference))
        del model,cache,x
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        torch.cuda.empty_cache()
    result = dict(domain=domain,status='PASS',rows=len(values),models=reports,seconds=time.perf_counter()-started)
    write_json(root/'validation.json',result)
    print(f'{domain}: prediction validation PASS, {len(values):,} rows x 6',flush=True)
    return result


def validate_shift(config: dict, prepared, domain: str) -> dict:
    project = Path(config['project_root'])
    root = project/config['outputs']['shift']/'10k'
    rows = pq.read_table(root/f'{domain}_windows.parquet').to_pylist()
    source = {r['window_id']:r for r in prepared.windows if r['domain']==domain and int(r['window_size'])==10000}
    lookup = {r['window_id']:r for r in rows}
    assert set(source)==set(lookup)
    ties = 0
    for row in rows:
        r = source[row['window_id']]
        assert row['previous_window_id']==(r['predecessor_window_id'] or None)
        assert row['rank_start']==int(r['rank_start']) and row['rank_end_exclusive']==int(r['rank_end_exclusive'])
        assert 0 <= row['jsd'] <= 1 and 0 <= row['shift'] <= 1
        assert len(row['marginal_jsd'])==49 and np.isfinite(row['marginal_jsd']).all()
        assert abs(row['jsd']-np.mean(row['marginal_jsd'])) <= 1e-15
        previous_id = row['previous_window_id']
        if previous_id:
            previous = lookup[previous_id]
            assert previous['scheme']==row['scheme'] and previous['stream_id']==row['stream_id']
            assert previous['last_available_ms']<=row['first_available_ms']
            assert previous['rank_end_exclusive']<=row['rank_start']
            assert row['shift']==previous['jsd'] and not row['initialization']
            ties += previous['last_available_ms']==row['first_available_ms']
        else:
            assert row['shift']==0 and row['initialization']
    result = dict(domain=domain,status='PASS',windows=len(rows),predecessor_checks='all_windows',
                  equal_availability_boundaries=ties,same_time_rule='available_at_same_timestamp_and_earlier_stable_rank')
    write_json(root/f'{domain}_validation.json',result)
    return result


def run_p3(config: dict, prepared) -> Path:
    project = Path(config['project_root'])
    output = project/config['outputs']['round1']/'P3'
    output.mkdir(parents=True,exist_ok=True)
    started = time.perf_counter()
    before = upstream_inventory(config)
    write_json(output/'upstream_inventory.json',dict(files=before))
    shutil.copy2(project/'configs/execution_v2.json',output/'execution_v2.json')
    runtime = dict(started_at_utc=datetime.now(timezone.utc).isoformat(),python=sys.executable,
                   gpu=torch.cuda.get_device_name(0),gpu_count=torch.cuda.device_count(),
                   torch_cuda=torch.version.cuda,cupy_runtime=cp.cuda.runtime.runtimeGetVersion(),
                   cupy_driver=cp.cuda.runtime.driverGetVersion(),
                   nvidia_smi=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],text=True).strip(),
                   versions={p:importlib.metadata.version(p) for p in ['numpy','torch','cupy-cuda12x','cuml-cu12','xgboost','pyarrow']})
    assert runtime['gpu_count']==1 and runtime['gpu']==config['device']['model_actual']
    write_json(output/'runtime.json',runtime)
    predictions,shifts,ledgers = [],[],[]
    for domain in config['datasets']:
        predictions.append(predict_domain(config,domain))
        validate_predictions(config,prepared,domain)
        shifts.append(compute_domain_shift(config,prepared,domain))
        validate_shift(config,prepared,domain)
        if domain!='UNSW':
            ledgers.append(build_domain_ledger(config,prepared,domain))
            validate_domain_ledger(config,prepared,domain)
        write_json(output/'progress.json',dict(completed_domains=[r['domain'] for r in predictions],seconds=time.perf_counter()-started))
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        torch.cuda.empty_cache()
    assert before==upstream_inventory(config)
    results = dict(status='COMPLETE',completed_at_utc=datetime.now(timezone.utc).isoformat(),
                   seconds=time.perf_counter()-started,predictions=predictions,shifts=shifts,ledgers=ledgers,
                   upstream_files_unchanged=len(before),scope='P3_prediction_cache_and_development_replay')
    write_json(output/'P3_manifest.json',results)
    write_json(output/'P3_validation.json',dict(status='PASS',upstream_unchanged=True,
        prediction_rows=sum(r['rows'] for r in predictions),prediction_model_rows=6*sum(r['rows'] for r in predictions),
        window_identities=sum(r['manifest_windows'] for r in shifts),unique_window_intervals=sum(r['unique_data_windows'] for r in shifts),
        physical_development_rows=sum(p['rows'] for r in ledgers for p in r['partitions']),
        logical_fold_development_rows=2*sum(p['rows'] for r in ledgers for p in r['partitions']),
        checks=['all_saved_prediction_ranges_and_evidence','fixed_rank_GPU_reprediction',
                'all_manifest_window_identities_and_predecessor_completion','all_ledger_rows_and_30_prediction_fields',
                'frozen_fold_development_roles','GPU_input_output_and_CUDA_events','upstream_size_and_mtime']))
    return write_report(config,results,runtime)


def write_report(config: dict, results: dict, runtime: dict) -> Path:
    output = Path(config['project_root'])/config['outputs']['round1']/'P3'
    lines = ['# P3 Forecast Caching and Development Recycling Report','',f" State: COMPLETETime to finish:{results['completed_at_utc']}。",'',
             '## records and windows','',
             '| range | cache record number (per model) | 6 model prediction number | 10k window identity | actual calculation range | initialization window | across partition front drive |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for p,s in zip(results['predictions'],results['shifts']):
        lines.append(f"| {p['domain']} | {p['rows']:,} | {p['rows']*6:,} | {s['manifest_windows']:,} | {s['unique_data_windows']:,} | {s['initialization_windows']} | {s['cross_partition_windows']} |")
    lines += ['', 'The projected cache covers all four domains. Six models, Platt, h and P1 statistics are directly loaded; All models in each batch are limited in quantity. The caching fields are raw_probability, probability, predicted_class, u, h. The NPY line r corresponds to availability rank r; The corresponding original line is determined only by the sorted_original_rows.npy, as determined by manifest.', '',
              'JSD preserves source domain, develops and excludes all 10k manifest identity roles. Calculate a non-labels linear graph of the same domain with the rank interval, with each character taking shift as their own manifest pre-drive. The numerical boundary, the classification vocabulary, OTHER and the filler are all from P1; 37 numeric columns and 12 categorical columns JSD both use GPU float64.', '',
              '## development records and tail','',
              '| range | partition | record | complete windows | complete windows record | tail | Initial record |',
              '|---|---|---:|---:|---:|---:|---:|']
    for ledger in results['ledgers']:
        for r in ledger['partitions']:
            lines.append(f"| {r['domain']} | {r['partition']} | {r['rows']:,} | {r['complete_windows']:,} | {r['complete_window_rows']:,} | {r['tail_rows']:,} | {r['initialization_rows']:,} |")
    lines += ['', '### Triple logical record number', '', '| Fold | development domain | Record number | complete windows Number | Initial window recording | tail recording |', '|---|---|---:|---:|---:|---:|']
    for fold, roles in config['folds'].items():
        parts = [p for ledger in results['ledgers'] if ledger['domain'] in roles['development'] for p in ledger['partitions']]
        lines.append(f"| {fold} | {'/'.join(roles['development'])} | {sum(p['rows'] for p in parts):,} | {sum(p['complete_windows'] for p in parts):,} | {sum(p['initialization_rows'] for p in parts):,} | {sum(p['tail_rows'] for p in parts):,} |")
    source_rows = next(r['rows'] for r in results['predictions'] if r['domain']=='UNSW')
    source_windows = next(r['unique_data_windows'] for r in results['shifts'] if r['domain']=='UNSW')
    lines += ['', f' source domain tail combined {source_rows-source_windows*10000:,} records, located in S_source_test; It is stored in six models of predictive caching.']
    lines += ['', 'tail maintains record_id, rank, original row indices, available time and six model predictions, tail=true, complete_window=false; window_id , previous_window_id , shift is null , initialization=false . tail has no fictitious window identity, and subsequent complete windows analysis filters out complete_window=true. source domain tail is stored in predictive caching.', '',
              'Initiation only occurs when the first complete windows and shift=0 are relayed back to each character. In other windows, shift is equivalent to manifest, which specifies the real JSD front-wheel drive, and continues to use the same current as the partition front-wheel drive without the labels front-wheel drive. At the same time, the brake border is processed according to the freeze steady-line sequence, the front-end rank is strictly earlier than the current window and the available time is no later than the current decision.', '',
              'F1 uses CIC/BoT, F2 uses CIC/ToN and F3 uses BoT/ToN; Each domain includes R_fit, R_cal, and R_select. The files are organized according to the fold/domain/partition path of Hive, and the same development records shared disk storage using hard links. record_id= domain name : original CSV zero starting line. The window_id, previous_window_id, shift, initialization and tail markings and all six model fields are retained in the table. labels is alternately associated with identity by subsequent modules.', '',
              '## equipment, time-consuming and persistent','',f"GPU：{runtime['nvidia_smi']}；Python：{runtime['python']}。",'',
              f" It’s based on version:{json.dumps(runtime['versions'],ensure_ascii=False)}。",'',
              f" P3 Finally effective runs the wall clock:{results['seconds']:.3f} Seconds (including drop-down discs, value-by-value checks and fixed position re-predictions)",'',
              '| domain | prediction and write (s) | JSD(s) | development write (s) | prediction phase sampling peak (MiB) | JSD sampling peak (MiB) |',
              '|---|---:|---:|---:|---:|---:|']
    ledger_times = {r['domain']:r['seconds'] for r in results['ledgers']}
    for p,s in zip(results['predictions'],results['shifts']):
        lines.append(f"| {p['domain']} | {p['seconds']:.3f} | {s['seconds']:.3f} | {ledger_times.get(p['domain'],0):.3f} | {max(m['gpu_memory']['peak_used_mib'] for m in p['models']):.1f} | {s['gpu_memory']['peak_used_mib']:.1f} |")
    lines += ['', 'Total occupancy of total-free devices with a storage capacity of CUDA memGetInfo, with each 100ms sample containing the background and storage memory pool on CUDA; It’s the peak of the sample. Each model manifest has an alternate CUDA reasoning event time and input / output residence device evidence. RF uses cuML/nvforest’s device input interface, XGBoost booster for cuda:0 and receives / back to CuPy, MLP parameters and output to CUDA Tensor; Platt, entropy, h queries, JSD, and cache-numeric verification use GPU; file and identity operations use CPU.', '',
              '## verification and products','',
              'Validate all caches in batches for finite raw/calibrated probabilities,predicted classes,u,h and[0,1] bounds;recompute calibration,entropy/competence. Re-predict fixed start,middle,end/partition-boundary ranks on GPU and compare caches. Check all window identities,predecessor IDs,times,ranks/shift sources. Reread all development Parquet rows and check record_id,rank,original row,availability time,30 prediction fields,windows,shift/fold roles.', '',
              f" Recycled {results['upstream_files_unchanged']} The upstream file size is consistent with the time before and after the nanosecond modification. freeze configuration is duplicated.", '',
              '- outputs/prediction_cache_v2/DOMAIN/: 6 model caches, by model device / range / time consuming manifest validation.json',
              '- outputs/shift_v2/10k/: per-domain window JSD, 49 marginal JSD, true front drive shift, initialization and calibration',
              '- outputs/round1_v2/P3/development/fold=F*/domain=*/partition=*/records.parquet: threefold development and recycling',
              '- outputs/round1_v2/P3/: total manifest, P3_validation.json, runtime.json, upstream registration, region-by-district recycling verification and reporting', '',
              'The code is implemented: replay/predictions.py, replay/shift.py, replay/ledger.py, p3.py, run.py; Unified input --stage replay-p3, running the scripts/run_p3.sh script, official log logs/P3.log.', '',
              'For the first time, the CIC acceptance of the fixed-position record was subtracted into small batches, with the final MLP–All49 probability of occurrence of 1.907×10⁻⁶ at the GPU float32 batch shape difference. The total probability of the original batch at positioning is completely consistent with the cached value-by-value. p3.py is converted to replace the fixed rank in the original batch with a precise, value-by-value comparison, and then re-run P3; Keep freeze parameters at the original batch.', '', 'Unresolved issues: GPU float32 may cause small differences in the shape of the reasoned batch; P3 fixed caching for subsequent recycling use, the original batch shape repositioning has been accurately verified.', '', 'The P3 data is being prepared at this stage. The P4 receiver has preserved the development records of fit CRE and the calculation of G0/G1; External impact assessments are carried out as planned at P7.', '']
    report = output/'stage_report.md'
    report.write_text('\n'.join(lines),encoding='utf-8')
    return report
