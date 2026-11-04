import argparse
import json
from pathlib import Path
import re
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument('manifest', type=Path)
parser.add_argument('output', type=Path)
parser.add_argument('--target', action='append', default=[])
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
artifacts = {}
for line in args.manifest.read_text(encoding='utf-8-sig').splitlines():
    record = json.loads(line)
    if record.get('reason') == 'compiler-artifact' and record.get('profile', {}).get('test') and record.get('executable'):
        artifacts[record['executable']] = record['target']['name']
unknown = set(args.target) - set(artifacts.values())
if unknown:
    parser.error(f'unknown test targets: {sorted(unknown)}')
results = []
for executable, target in artifacts.items():
    selected = not args.target or target in args.target
    if selected:
        listing = subprocess.run([executable, '--list'], capture_output=True, text=True, check=True)
        count = len(re.findall(r': test$', listing.stdout, re.MULTILINE))
        if count:
            started = time.perf_counter()
            previous = started
            cases = []
            print(f'Starting {target}: {count} tests', flush=True)
            with (args.output / f'{target}.log').open('w', encoding='utf-8') as log:
                process = subprocess.Popen([executable, '--test-threads=1', '--color', 'never'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    now = time.perf_counter()
                    if re.match(r'running \d+ tests?', line):
                        previous = now
                    result = re.match(r'test (.+?) \.\.\. (ok|FAILED|ignored)', line)
                    if result:
                        case = dict(name=result[1], status=result[2], approximate_seconds=now-previous)
                        cases.append(case)
                        previous = now
                        print(f'{target}::{case["name"]}: {case["status"]} {case["approximate_seconds"]:.3f}s', flush=True)
                code = process.wait()
            outcome = code or int(len(cases) != count)
            results.append(dict(target=target, executable=executable, tests=count, exit_code=outcome, seconds=time.perf_counter()-started, cases=cases))
            (args.output / 'summary.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
if not results:
    parser.error('the manifest contains no selected executable tests')
print(json.dumps(dict(targets=len(results), tests=sum(r['tests'] for r in results), seconds=sum(r['seconds'] for r in results), failures=sum(r['exit_code'] != 0 for r in results))), flush=True)
raise SystemExit(int(any(r['exit_code'] != 0 for r in results)))
