"""Round 1 data preparation only; existing NumPy + Python standard library.

Stable identity: (domain, original filename, zero-based CSV data row).
All rank intervals are half-open. Input CSV files are never written.
"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import itertools
import json
import platform
import resource
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

START, END = "FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS"
EXCLUDED = {START, END, "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "Label", "Attack"}
VERSION = "round1-data-prep-v1.0"
GROUP = 10000
SIZES = (5000, 10000, 50000)
FOLDS = {"F1": {"development": ["CIC", "BoT"], "heldout": "ToN"},
         "F2": {"development": ["CIC", "ToN"], "heldout": "BoT"},
         "F3": {"development": ["BoT", "ToN"], "heldout": "CIC"}}


def now():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_header(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        original = next(csv.reader(f))
    clean = [x.strip() for x in original]
    if len(set(clean)) != len(clean):
        raise ValueError(f"Duplicate column names: {path}")
    return original, clean


def split_ranges(n, source):
    # Source proportions use all indivisible groups including the final short group.
    # Development proportions use eligible full 10k groups; short tail goes to R_select.
    units = (n + GROUP - 1) // GROUP if source else n // GROUP
    cumulative = (60, 75, 90) if source else (50, 75)
    cuts = [0] + [min(n, ((units * p + 50) // 100) * GROUP) for p in cumulative] + [n]
    names = ("S_train", "S_prob_cal", "S_competence", "S_source_test") if source else ("R_fit", "R_cal", "R_select")
    return [(name, cuts[i], cuts[i + 1]) for i, name in enumerate(names)]


def make_windows(domain, scheme, ranges, order, ends, size):
    windows, tails, blocks = [], [], []
    previous = None
    for partition, lo, hi in ranges:
        count = (hi - lo) // size
        part_windows = []
        for i in range(count):
            a, b = lo + i * size, lo + (i + 1) * size
            w = dict(domain=domain, scheme=scheme, stream_id=f"{domain}:whole_domain",
                     partition=partition, window_size=size,
                     window_id=f"{domain}:{scheme}:{size}:{partition}:{i}",
                     rank_start=a, rank_end_exclusive=b,
                     first_original_row=int(order[a]), last_original_row=int(order[b - 1]),
                     first_available_ms=int(ends[a]), last_available_ms=int(ends[b - 1]),
                     predecessor_window_id=previous["window_id"] if previous else "",
                     predecessor_partition=previous["partition"] if previous else "",
                     predecessor_last_available_ms=previous["last_available_ms"] if previous else "",
                     cross_partition_context=bool(previous and previous["partition"] != partition),
                     initialization=previous is None,
                     shift_initial_value=0.0 if previous is None else "",
                     shift_source="source_reference" if previous is None else "previous_complete_unlabeled_window")
            if previous:
                assert previous["rank_end_exclusive"] <= a
                assert previous["last_available_ms"] <= w["first_available_ms"]
            windows.append(w)
            part_windows.append(w)
            previous = w
        tails.append(dict(domain=domain, scheme=scheme, partition=partition, window_size=size,
                          rank_start=lo + count * size, rank_end_exclusive=hi,
                          excluded_rows=(hi - lo) % size, complete_windows=count))
        for j in range(0, count, 10):
            sub = part_windows[j:j + 10]
            blocks.append(dict(domain=domain, scheme=scheme, stream_id=f"{domain}:whole_domain",
                               partition=partition, window_size=size,
                               block_id=f"{domain}:{scheme}:{size}:{partition}:block{j // 10}",
                               first_window_id=sub[0]["window_id"], last_window_id=sub[-1]["window_id"],
                               window_count=len(sub), rank_start=sub[0]["rank_start"],
                               rank_end_exclusive=sub[-1]["rank_end_exclusive"], shorter_final_block=len(sub) < 10))
    return windows, tails, blocks


def synthetic_tests():
    # Long flow begins first but is only available last; stable ties remain ordered.
    starts = np.array([1, 2, 3, 4, 5, 6, 7])
    end = np.array([100, 10, 10, 11, 12, 12, 13])
    order = np.argsort(end, kind="stable")
    assert order.tolist() == [1, 2, 3, 4, 5, 6, 0]
    assert np.all(end >= starts)
    w, t, b = make_windows("test", "development", [("R_fit", 0, 3), ("R_cal", 3, 7)], order, end[order], 2)
    assert t[0]["excluded_rows"] == 1 and w[1]["cross_partition_context"]
    assert w[1]["predecessor_window_id"] == w[0]["window_id"]
    assert len(b) == 2 and w[0]["initialization"] and not w[1]["initialization"]
    tied, _, _ = make_windows("tie", "development", [("a", 0, 2), ("b", 2, 4)], np.arange(4), np.array([10]*4), 2)
    assert tied[1]["predecessor_last_available_ms"] == tied[1]["first_available_ms"]
    for n in (0, 1, 9999, 10000, 10001, 39999, 40000, 235001):
        for source in (True, False):
            ranges = split_ranges(n, source)
            assert ranges[0][1] == 0 and ranges[-1][2] == n
            assert all(a <= z for _, a, z in ranges)
            assert all(ranges[i][2] == ranges[i+1][1] for i in range(len(ranges)-1))
            assert all(a % GROUP == 0 for _, a, _ in ranges if a < n)
    return {"long_flow_availability": "PASS", "stable_ties_and_equal_time_predecessor": "PASS",
            "partition_tails_and_context": "PASS", "rounding_empty_small_and_short_groups": "PASS",
            "runtime_future_feature_mutation": "PENDING_MODEL_PIPELINE",
            "runtime_heldout_label_mutation": "PENDING_MODEL_PIPELINE", "paired_CRE_fit": "PENDING_MODEL_PIPELINE"}


def scan(path, output, chunk_rows):
    before = path.stat()
    _, header = read_header(path)
    positions = [header.index(x) for x in (START, END, "Label", "Attack")]
    chunks_end, chunks_label = [], []
    families = Counter()
    invalid = Counter()
    rows = decreases = 0
    last_end = None
    start_min, start_max = None, None
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        while True:
            selected = []
            for row in itertools.islice(reader, chunk_rows):
                if len(row) != len(header):
                    raise ValueError(f"{path.name} data row {rows + len(selected)}: wrong column count")
                s, e, label, attack = (row[p].strip() for p in positions)
                try:
                    s, e, label = int(s), int(e), int(label)
                except ValueError as exc:
                    raise ValueError(f"Invalid timestamp/label at {path.name}:{rows + len(selected)}") from exc
                invalid["nonpositive_timestamp"] += int(s <= 0 or e <= 0)
                invalid["end_before_start"] += int(e < s)
                invalid["nonbinary_label"] += int(label not in (0, 1))
                invalid["label_attack_mismatch"] += int((label == 0) != (attack.lower() == "benign"))
                families[attack] += 1
                selected.append((s, e, label))
            if not selected:
                break
            a = np.asarray(selected, dtype=np.int64)
            ends = a[:, 1].copy()
            decreases += int(np.count_nonzero(ends[1:] < ends[:-1]))
            decreases += int(last_end is not None and ends[0] < last_end)
            last_end = int(ends[-1])
            mn, mx = int(a[:, 0].min()), int(a[:, 0].max())
            start_min = mn if start_min is None else min(start_min, mn)
            start_max = mx if start_max is None else max(start_max, mx)
            chunks_end.append(ends)
            chunks_label.append(a[:, 2].astype(np.int8))
            rows += len(a)
            if rows % (chunk_rows * 10) == 0:
                print(f"{now()} {path.name}: scanned {rows:,} rows", flush=True)
    after = path.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "Input changed"
    audit = dict(file=str(path), bytes=before.st_size, mtime_ns=before.st_mtime_ns,
                 rows=rows, columns=header, raw_adjacent_end_decreases=decreases,
                 start_min_ms=start_min, start_max_ms=start_max, invalid_counts=dict(invalid),
                 attack_family_counts=dict(families), input_stat_unchanged=True)
    save_json(output / f"{path.stem}_scan.json", audit)
    if not rows or any(invalid.values()):
        raise ValueError(f"Data quality blocks split generation: {audit['invalid_counts']}")
    return np.concatenate(chunks_end), np.concatenate(chunks_label), audit


def process(domain, path, output, chunk_rows, features, dictionary):
    _, header = read_header(path)
    assert len(header) == 55 and set(header) - EXCLUDED == features
    assert set(header) - {"Label", "Attack"} == dictionary
    assert not any("capture" in x.lower() or "session" in x.lower() for x in header)
    ends, labels, audit = scan(path, output, chunk_rows)
    print(f"{now()} {domain}: stable end-time sort", flush=True)
    order = np.argsort(ends, kind="stable")
    ends, labels = ends[order], labels[order]
    n = len(order)
    # Exact permutation validation, not just count/sum: each row occurs once.
    seen = np.zeros(n, dtype=np.bool_)
    seen[order] = True
    assert bool(seen.all()) and len(order) == n
    del seen
    assert bool(np.all(ends[1:] >= ends[:-1]))
    tied = ends[1:] == ends[:-1]
    assert bool(np.all(order[1:][tied] > order[:-1][tied]))
    np.save(output / f"{domain}_sorted_original_rows.npy", order, allow_pickle=False)
    np.save(output / f"{domain}_sorted_end_ms.npy", ends, allow_pickle=False)
    schemes = {"source" if domain == "UNSW" else "development": split_ranges(n, domain == "UNSW")}
    if domain != "UNSW":
        schemes["heldout"] = [("T_heldout", 0, n)]
    splits, groups, windows, tails, blocks, intersections = [], [], [], [], [], []
    for scheme, ranges in schemes.items():
        covered = np.zeros(n, dtype=np.uint8)
        for part, lo, hi in ranges:
            covered[lo:hi] += 1
            attack = int(labels[lo:hi].sum())
            splits.append(dict(domain=domain, filename=path.name, scheme=scheme, partition=part,
                               stream_id=f"{domain}:whole_domain", rank_start=lo, rank_end_exclusive=hi,
                               rows=hi-lo, benign=hi-lo-attack, attack=attack, fraction=(hi-lo)/n,
                               first_available_ms=int(ends[lo]) if lo < hi else "",
                               last_available_ms=int(ends[hi-1]) if lo < hi else "",
                               group_start=lo//GROUP, group_end_exclusive=(hi+GROUP-1)//GROUP,
                               sorted_row_index=f"{domain}_sorted_original_rows.npy"))
            for a in range(lo, hi, GROUP):
                b = min(a+GROUP, hi)
                assert a % GROUP == 0 and (b % GROUP == 0 or b == n)
                groups.append(dict(domain=domain, scheme=scheme, partition=part,
                                   group_id=f"{domain}:chronological10k:{a//GROUP}",
                                   rank_start=a, rank_end_exclusive=b, rows=b-a,
                                   first_available_ms=int(ends[a]), last_available_ms=int(ends[b-1])))
        assert bool(np.all(covered == 1))
        del covered
        for i, (p, a, b) in enumerate(ranges):
            for q, c, d in ranges[i+1:]:
                overlap = max(0, min(b, d) - max(a, c))
                assert overlap == 0
                intersections.append(dict(domain=domain, scheme=scheme, partition_a=p, partition_b=q, shared_records=overlap))
        for size in SIZES:
            w, t, b = make_windows(domain, scheme, ranges, order, ends, size)
            assert sum(size for _ in w) + sum(x["excluded_rows"] for x in t) == n
            assert sum(x["window_count"] for x in b) == len(w)
            windows.extend(w)
            tails.extend(t)
            blocks.extend(b)
    audit.update(domain=domain, end_min_ms=int(ends[0]), end_max_ms=int(ends[-1]),
                 equal_end_adjacent_pairs=int(tied.sum()), max_availability_gap_ms=int(np.diff(ends).max()) if n > 1 else 0,
                 checks={key: "PASS" for key in ["schema_dictionary", "feature_boundary_49", "exact_record_permutation",
                     "end_time_order_and_stable_ties", "partition_exact_cover", "partition_intersections_zero",
                     "indivisible_10k_groups", "predecessor_completion", "window_tail_conservation", "bootstrap_coverage"]})
    return audit, splits, groups, windows, tails, blocks, intersections


def report(output, run, audits, splits, windows, tails, blocks):
    lines = ["# Round 1 Data preparation report", "", f" State of play:{run['status']}; The experimental configuration is still DRAFT_NOT_FROZEN.",
             f" It runs:{run['started_at_utc']} From now on, {run['finished_at_utc']}。", "",
             f" The actual scanning {sum(a['rows'] for a in audits):,} the records; It’s not a trained model, it’s not looking at the results of external models.", "",
             "## implementation and allocation rules", "",
             "Use existing agentnids It’s... NumPy (b) With the standard library, only the time and binary labels are kept; The original CSV file is read only.",
             "Record identity as ( data set , Original filename , Zero starting data line ). A stable ranking of ) according to (FLOW_END_MILLISECONDS, prime number; The NPY index supports mmap.",
             "No real capture/session column or additional metadata with each domain as a relay stream; Continuous 10,000 clauses consist of indelible time groups, the last short group being retained.",
             "source domain by the cumulative 60%/75%/90% of the whole set of integers, taking the most recent integer ((precisely half upwards); The goal of development is to integrate the cumulative 50%/75% of the total number of 10k groups into R_select.",
             "partition is distributed continuously according to temporal order and does not change according to category. 5k/10k/50k is repeatedly grouped into partition and the window is re-opened.",
             "The end of each section is less than one window to retain the identity of the member of the division, excluding a full window analysis of the size of the corresponding window.",
             "The front drive takes only the same scheme and the previous complete windows which has been completed in the same stream; Allowing the adjacent partition to be crossed without labels in the context above, the tail is excluded.",
             "The same end time is available for the same batch and the time allowed for the front-wheel finish is equal to the time available for the current first batch; The batch is handled according to the stable line number and is not declared to be strictly smaller.",
             "Bootstrap per partition per 10 per continuous complete windows component, retaining the last short block, does not actually execute resampling.", "",
             "## partition category and number", "",
             "| Data set | Use of | partition | Record number | Benign | Attack | Actual ratio |",
             "|---|---|---|---:|---:|---:|---:|"]
    for s in splits:
        lines.append(f"| {s['domain']} | {s['scheme']} | {s['partition']} | {s['rows']:,} | {s['benign']:,} | {s['attack']:,} | {s['fraction']:.6%} |")
    lines += ["", "For each partition millisecond time range see split_manifest.csv; Primitive time quantity, attack subcategories count and maximum time interval in the domain see data_audit.json.", "",
              "## complete windows / tail excludes / resampling units", "",
              "| data set | use | partition | window flow number | complete windows | exclusion record | blocks number | window midrange ( seconds ) |",
              "|---|---|---|---:|---:|---:|---:|---:|"]
    for t in tails:
        key = tuple(t[k] for k in ("domain", "scheme", "partition", "window_size"))
        ws = [w for w in windows if tuple(w[k] for k in ("domain", "scheme", "partition", "window_size")) == key]
        bs = [b for b in blocks if tuple(b[k] for k in ("domain", "scheme", "partition", "window_size")) == key]
        span = float(np.median([(w['last_available_ms']-w['first_available_ms'])/1000 for w in ws])) if ws else None
        lines.append(f"| {' | '.join(map(str, key))} | {t['complete_windows']} | {t['excluded_rows']} | {len(bs)} | {span} |")
    missing = [f"{s['domain']}/{s['partition']}" for s in splits if not s['benign'] or not s['attack']]
    insufficient = [f"{t['domain']}/{t['scheme']}/{t['partition']}/{t['window_size']}" for t in tails if t['complete_windows'] < 191]
    lines += ["", "## Acceptance and Limitation", "",
              "Adopted by: / time / labels consistency; The ranking index is ranked with precision on the original line; For each use, a partition is recorded; partition is zero; The group shall not be divided; The windows and the tail coverage are permanent; the front-wheel drive is complete and available; The resampling block covers complete windows and does not exceed partition.",
              "By: synthesizing long-flow, simultaneous cross-border, short-tail, partition front-drive, blank data and group testing. The features table maintains the original 17/17/15 subgroup, with 49 items excluding absolute time IP/ / labels / identity.",
              f" There is a lack of the necessary partition category:{', '.join(missing) or 'none'}。",
              "Combination of 20 units of resampling that are insufficient (thresholds restricted only to relevant partition assessments):" + (", ".join(insufficient) or "none") + "。",
              "There is no evidence of capture/session isolation, and the time group and any consecutive window block do not guarantee independent statistics; Large time intervals are not fictitious as independent capture. Unsimulated / output and latency transmission.",
              "development and heldout manifest are alternate uses of the same target domain in different folds, and their mutual reuse is a design requirement; In the same way, you can only choose one of the uses specified by fold_manifest.json, which cannot be combined with training.",
              "Category statistics for this stage are for preparation of audit only; future features disrupts, leaves labels disruption and CRE to fit when the actual model is not validated, and static front-drive checks are not accepted as such.",
              "freeze is not configured; parameters training, follow-up sensitivity, threshold justification, previous outer results exposure and other freeze items still need to be registered. The state of the missing training library is run_config.json, which is not dependent on installation.", "",
              "## output and running back", "", "Running command:", "", "```bash", run['command'], "```", "",
              "split_manifest.csv gives partition a half-off rank interval; group_manifest.csv gives the group identity; window_manifest.csv gives complete windows, forerunner and initialization; tail_manifest.csv gives the exclusion interval; bootstrap_manifest.csv gives the resampling block.",
              "Rebuild the member line with DOMAIN_sorted_original_rows.npy[rank_start:rank_end_exclusive] and extract the required features from the original CSV; The original row number at the end of the window cannot be used as a continuous interval between the original files.",
              "The code, configuration and original features manifest copies are kept running; It uses the normal version number, time, file size, time and row records, and no hashes.", "",
              "## Next small-scale entry (not trained)", "",
              "First run prepare_data.py --self-test and then read / as a small-scale relay input using F1 development domain in fold_manifest.json and 10k complete windows in window_manifest.csv in this directory.",
              "The future training script is not yet available; First, complete and register training configuration and dependency, and then implement S_train preprocessing/Model, source domain calibration and development relay, performing real disruptive acceptance. This data should not be considered as a formal training or external unblinding permit to be successfully prepared.", ""]
    (output / "data_preparation_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("datasets"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--chunk-rows", type=int, default=250000)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    tests = synthetic_tests()
    if args.self_test:
        print(json.dumps(tests, indent=2))
        return
    if args.output is None or args.chunk_rows <= 0:
        parser.error("--output and positive --chunk-rows required")
    output, root = args.output.resolve(), args.data_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    setup = Path(__file__).resolve().parent
    config = json.loads((setup / "round1_config.json").read_text(encoding="utf-8-sig"))
    with (setup / "feature_view_manifest_v1.csv").open(encoding="utf-8-sig", newline="") as f:
        views = list(csv.DictReader(f))
    features = {r['feature'] for r in views}
    assert len(views) == len(features) == 49 and not features.intersection(EXCLUDED)
    assert Counter((r['view'], r['detector']) for r in views) == {('Traffic', 'RF'): 17, ('Protocol', 'XGBoost'): 17, ('Behavior', 'MLP'): 15}
    with (root / config['feature_dictionary']).open(encoding="utf-8-sig", newline="") as f:
        dictionary = {r['Feature'].strip() for r in csv.DictReader(f)}
    versions = {}
    for p in ('numpy', 'pyarrow', 'pandas', 'scipy', 'scikit-learn', 'xgboost'):
        try:
            versions[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            versions[p] = "NOT_INSTALLED"
    import shlex
    run = dict(version=VERSION, status="RUNNING", started_at_utc=now(), python=sys.version,
               executable=sys.executable, platform=platform.platform(), dependencies=versions,
               data_root=str(root), output=str(output), chunk_rows=args.chunk_rows,
               command=shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]),
               disk_free_bytes=shutil.disk_usage(output).free,
               memory_info=Path('/proc/meminfo').read_text(),
               metadata_inventory=[dict(path=str(p), bytes=p.stat().st_size) for p in sorted(root.rglob('*')) if p.is_file()],
               group_mode="nonoverlapping_chronological_10000_flow_groups_no_session_claim",
               source_rounding="nearest_half_up_cumulative_all_groups_60_75_90_percent",
               development_rounding="nearest_half_up_cumulative_full_groups_50_75_percent_tail_to_R_select",
               window_sizes=list(SIZES), bootstrap_block_windows=10, minimum_bootstrap_units=20,
               feature_view_counts=dict(Counter(r['view'] for r in views)),
               experiment_config=config, synthetic_checks=tests)
    save_json(output / 'run_config.json', run)
    for name in ('round1_config.json', 'feature_view_manifest_v1.csv', 'prepare_data.py'):
        shutil.copyfile(setup / name, output / name)
    shutil.copyfile(root / config['feature_dictionary'], output / config['feature_dictionary'])
    save_json(output / 'fold_manifest.json', {fold: {**v, 'source_domain': 'UNSW',
              'source_scheme': 'source', 'development_scheme': 'development', 'heldout_scheme': 'heldout',
              'heldout_fit_records': 0, 'outer_results_inspected': False} for fold, v in FOLDS.items()})
    audits, splits, groups, windows, tails, blocks, intersections = [], [], [], [], [], [], []
    try:
        for domain, filename in config['datasets'].items():
            result = process(domain, root / filename, output, args.chunk_rows, features, dictionary)
            audits.append(result[0])
            for target, rows in zip((splits, groups, windows, tails, blocks, intersections), result[1:]):
                target.extend(rows)
            save_json(output / 'data_audit.json', audits)
            print(f"{now()} {domain}: {result[0]['rows']:,} rows; all data checks PASS", flush=True)
        for name, rows in [('split', splits), ('group', groups), ('window', windows), ('tail', tails), ('bootstrap', blocks), ('partition_intersection', intersections)]:
            save_csv(output / f'{name}_manifest.csv', rows)
        save_json(output / 'validation.json', dict(synthetic=tests, domains={a['domain']: a['checks'] for a in audits},
                  fold_domain_disjoint=all(v['heldout'] not in v['development'] and 'UNSW' not in v['development'] for v in FOLDS.values())))
        run['status'] = 'DATA_PREPARATION_COMPLETE_NOT_EXPERIMENT_FROZEN'
    except Exception as exc:
        run.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        run['finished_at_utc'] = now()
        run['peak_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        save_json(output / 'run_config.json', run)
        report(output, run, audits, splits, windows, tails, blocks)
    print(f"Report: {output / 'data_preparation_report.md'}", flush=True)


if __name__ == '__main__':
    main()
