#!/usr/bin/env python3
"""Scan native N-1 scaling on Linux; run from any directory after a release build."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

REPO = Path(__file__).resolve().parents[2]
CASES = {
    'case14': (14, 201), 'case30': (30, 101), 'case57': (57, 101),
    'case89pegase': (89, 51), 'case300': (300, 21),
    'case1354pegase': (1354, 1), 'case2869pegase': (2869, 1), 'case9241': (9241, 1),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def command_output(command):
    result = subprocess.run(command, cwd=REPO, capture_output=True, text=True)
    return {'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}


def power_snapshot():
    paths = [Path('/sys/firmware/acpi/platform_profile')]
    for cpu in Path('/sys/devices/system/cpu').glob('cpu[0-9]*'):
        paths += [cpu / 'cpufreq' / name for name in
                  ('scaling_driver', 'scaling_governor', 'energy_performance_preference',
                   'scaling_min_freq', 'scaling_max_freq', 'scaling_cur_freq')]
    values = {}
    for path in paths:
        try:
            values[str(path)] = path.read_text().strip()
        except OSError:
            pass
    return values


def prepare_cases(names, case_dir):
    """Reuse cached ZIPs; export missing small cases from local pandapower."""
    paths = {}
    missing = []
    for name in names:
        path = REPO / 'cases/pegase9241/data.zip' if name == 'case9241' else case_dir / f'{name}.zip'
        paths[name] = path
        if not path.is_file():
            if name == 'case9241':
                raise FileNotFoundError(path)
            missing.append(name)
    if not missing:
        return paths, None
    import zipfile
    import pandapower as pp
    import pandapower.networks as networks
    case_dir.mkdir(parents=True, exist_ok=True)
    for name in missing:
        print(f'EXPORT {name}', flush=True)
        net = getattr(networks, name)()
        if float(net.sn_mva) != 100.0:
            raise ValueError(f'{name}: CSV reader expects this sweep\'s 100 MVA system base')
        path = paths[name]
        partial = path.with_suffix('.partial')
        try:
            with zipfile.ZipFile(partial, 'w', compression=zipfile.ZIP_DEFLATED) as output:
                for table in ('bus', 'line', 'trafo', 'shunt', 'gen', 'ext_grid', 'load', 'sgen', 'switch'):
                    frame = net[table].copy()
                    for column in ('bus', 'from_bus', 'to_bus', 'hv_bus', 'lv_bus', 'parallel',
                                   'element', 'step', 'max_step'):
                        if column in frame.columns:
                            frame[column] = frame[column].astype('Int64')
                    if table == 'trafo' and 'tap_phase_shifter' not in frame.columns:
                        kinds = set(frame.get('tap_changer_type', []).dropna()) if 'tap_changer_type' in frame else set()
                        if not kinds.issubset({'Ratio'}):
                            raise ValueError(f'{name}: unsupported legacy tap conversion: {kinds}')
                        frame['tap_phase_shifter'] = False
                    output.writestr(f'{table}.csv', frame.to_csv(index=True, index_label='index'))
            partial.replace(path)
            save(path.with_suffix('.source.json'), {'case': name, 'pandapower_version': pp.__version__,
                 'sn_mva': float(net.sn_mva), 'sha256': digest(path)})
        finally:
            partial.unlink(missing_ok=True)
    return paths, pp.__version__


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpus', default='0-7', help='same CPU affinity for both worker counts (default: 0-7)')
    parser.add_argument('--out', type=Path, default=Path('/tmp/n1-scale-4'))
    parser.add_argument('--case-dir', type=Path, help='cached CSV ZIPs; default: OUT/cases')
    parser.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES))
    parser.add_argument('--max-candidates', type=int, help='select the first N active branches before topology screening; default: all')
    parser.add_argument('--large-repeats', type=int, default=1, help='whole-batch repeats for >=1000 buses')
    target = Path(os.environ.get('CARGO_TARGET_DIR', str(REPO / 'target')))
    if not target.is_absolute():
        target = REPO / target
    parser.add_argument('--binary', type=Path, default=target / 'release/bench_n1_scale')
    args = parser.parse_args()
    if args.max_candidates is not None and args.max_candidates < 0:
        parser.error('--max-candidates must be nonnegative')
    if args.large_repeats < 1:
        parser.error('--large-repeats must be positive')
    binary = args.binary.resolve()
    if not binary.is_file():
        parser.error('build first: cargo build --release --features experimental_n1 --bin bench_n1_scale')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'metadata.json').exists():
        parser.error(f'{out} already contains a scan; choose a new --out directory')
    paths, pp_version = prepare_cases(args.cases, (args.case_dir or out / 'cases').resolve())
    metadata = {
        'binary': str(binary), 'binary_sha256': digest(binary), 'cpu_affinity': args.cpus,
        'candidate_limit': args.max_candidates,
        'candidate_selection': 'catalog_prefix_before_topology_screening' if args.max_candidates is not None else 'all_active_branches',
        'completed': False, 'workers': [1, 4], 'batch_size': 8, 'initialization': 'BaseAc',
        'reporting': 'StatusOnly', 'constraint_rules': 0, 'max_iter': 30, 'tolerance': 1e-10,
        'source_head': command_output(['git', 'rev-parse', 'HEAD']),
        'source_status': command_output(['git', 'status', '--short']),
        'cpu_model': command_output(['lscpu']),
        'cpu_topology': command_output(['lscpu', '-e=CPU,CORE,SOCKET']),
        'power_before': power_snapshot(),
        'power_snapshot_scope': 'before/after snapshots; scaling_cur_freq is instantaneous, not a busy-frequency average',
        'export_pandapower_version': pp_version,
        'inputs': {name: {'path': str(path), 'sha256': digest(path)} for name, path in paths.items()},
        'commands': [],
    }
    save(out / 'metadata.json', metadata)
    rows = []
    print(f'PF workers=1,4; allowed CPUs={args.cpus}; batch=8; candidate_limit={args.max_candidates}; output={out}', flush=True)
    print('case              PF/island  reps       1w ms       4w ms   speedup   efficiency', flush=True)
    for name in args.cases:
        buses, repeats = CASES[name]
        if buses >= 1000:
            repeats = args.large_repeats
        results = {}
        for workers in (1, 4):
            result_path = out / f'{name}-workers{workers}.json'
            command = ['taskset', '-c', args.cpus, str(binary), name, str(paths[name]),
                       str(workers), str(repeats), str(result_path), '8']
            if args.max_candidates is not None:
                command.append(str(args.max_candidates))
            metadata['commands'].append(command)
            save(out / 'metadata.json', metadata)
            print(f'RUN {name} workers={workers} repeats={repeats}', flush=True)
            started = time.monotonic()
            with (out / f'{name}-workers{workers}.log').open('w') as log:
                result = subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'{name}, workers={workers} failed; read {log.name}')
            result = json.loads(result_path.read_text())
            selection_fields = ('available_candidates', 'candidate_limit', 'candidate_selection')
            if (not all(key in result for key in selection_fields)
                    or result['candidate_limit'] != args.max_candidates):
                raise RuntimeError('benchmark ignored --max-candidates; rebuild bench_n1_scale')
            result['driver_process_wall_seconds'] = time.monotonic() - started
            save(result_path, result)
            results[workers] = result
        a, b = results[1], results[4]
        keys = ('available_candidates', 'candidate_limit', 'candidate_selection', 'buses', 'candidates', 'numerically_solved', 'passed', 'violated', 'numerical_failed',
                'islanded', 'rejected_scenario_statuses', 'island_outage_indices',
                'attempted_ac_iterations_samples', 'actual_cpu_binding')
        if any(a[key] != b[key] for key in keys):
            raise RuntimeError(f'{name}: worker counts changed decisions/iterations/affinity; inspect JSON files')
        speedup = a['run_wall_seconds_median'] / b['run_wall_seconds_median']
        row = {'case': name, 'buses': buses, 'candidates': a['candidates'],
               'available_candidates': a['available_candidates'], 'candidate_limit': args.max_candidates,
               'numerically_solved': a['numerically_solved'], 'islanded': a['islanded'],
               'failed': a['numerical_failed'], 'repeats': repeats,
               'one_worker_ms': a['run_wall_seconds_median'] * 1000,
               'four_workers_ms': b['run_wall_seconds_median'] * 1000,
               'speedup': speedup, 'efficiency_percent': speedup / 4 * 100,
               'decisions_and_iterations_match': True,
               'prepare_one_ms': a['shared_preparation_seconds'] * 1000,
               'prepare_four_ms': b['shared_preparation_seconds'] * 1000,
               'first_end_to_end_one_ms': a['first_call_end_to_end_seconds'] * 1000,
               'first_end_to_end_four_ms': b['first_call_end_to_end_seconds'] * 1000}
        rows.append(row)
        save(out / 'summary.json', rows)
        with (out / 'summary.csv').open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
        print(f"{name:17} {row['numerically_solved']:5}/{row['islanded']:<5} {repeats:4} "
              f"{row['one_worker_ms']:11.3f} {row['four_workers_ms']:11.3f} "
              f"{speedup:8.3f}x {row['efficiency_percent']:9.1f}%", flush=True)
    metadata['power_after'] = power_snapshot()
    metadata['completed'] = True
    save(out / 'metadata.json', metadata)
    print(f'COMPLETE: {out}/summary.csv', flush=True)


if __name__ == '__main__':
    main()
