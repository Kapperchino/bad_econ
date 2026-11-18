import argparse
import json
from pathlib import Path
import re
import subprocess
import time


def executables(manifest):
    with manifest.open(encoding='utf-8-sig') as stream:
        return {
            record['executable']: record['target']['name']
            for record in map(json.loads, stream)
            if record.get('reason') == 'compiler-artifact'
            and record.get('profile', {}).get('test')
            and record.get('executable')
        }


def test_count(executable):
    listing = subprocess.run([executable, '--list'], capture_output=True, text=True, check=True)
    return sum(line.endswith(': test') for line in listing.stdout.splitlines())


def measure(executable, target, count, output):
    started = previous = time.perf_counter()
    cases = []
    print(f'Starting {target}: {count} tests', flush=True)
    with (output / f'{target}.log').open('w', encoding='utf-8') as log, subprocess.Popen(
        [executable, '--test-threads=1', '--color', 'never'],
        stdout=subprocess.PIPE, stderr=log, text=True, encoding='utf-8', errors='replace'
    ) as process:
        for line in process.stdout:
            log.write(line)
            log.flush()
            now = time.perf_counter()
            result = re.match(r'test (.+?) \.\.\. (ok|FAILED|ignored)', line)
            if result:
                case = dict(name=result[1], status=result[2], approximate_seconds=now-previous)
                cases.append(case)
                previous = now
                print(f'{target}::{case["name"]}: {case["status"]} {case["approximate_seconds"]:.3f}s', flush=True)
            elif re.match(r'running \d+ tests?', line):
                previous = now
    return dict(target=target, executable=executable, tests=count,
                exit_code=process.returncode or int(len(cases) != count),
                seconds=time.perf_counter()-started, cases=cases)


parser = argparse.ArgumentParser()
parser.add_argument('manifest', type=Path)
parser.add_argument('output', type=Path)
parser.add_argument('--target', action='append', default=[])
parser.add_argument('--max-seconds', type=int, default=120)
args = parser.parse_args()
if args.max_seconds <= 0:
    parser.error('--max-seconds must be positive')
args.output.mkdir(parents=True, exist_ok=True)
artifacts = executables(args.manifest)
selected = set(args.target) or set(artifacts.values())
unknown = selected.difference(artifacts.values())
if unknown:
    parser.error(f'unknown test targets: {sorted(unknown)}')
results = []
for executable, target in artifacts.items():
    if target in selected and (count := test_count(executable)):
        results.append(measure(executable, target, count, args.output))
        (args.output / 'summary.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
if not results:
    parser.error('the manifest contains no selected executable tests')
seconds = sum(r['seconds'] for r in results)
failures = sum(r['exit_code'] != 0 for r in results)
print(json.dumps(dict(targets=len(results), tests=sum(r['tests'] for r in results), seconds=seconds, failures=failures, max_seconds=args.max_seconds, within_budget=seconds < args.max_seconds)), flush=True)
raise SystemExit(int(failures != 0 or seconds >= args.max_seconds))
