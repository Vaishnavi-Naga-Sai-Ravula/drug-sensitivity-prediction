"""Measure a running local research API; this is not a production load guarantee."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import platform
import time
import httpx
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--url', default='http://127.0.0.1:8505')
parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'outputs/runtime_benchmark.json')
args = parser.parse_args()
with httpx.Client(base_url=args.url, timeout=30, trust_env=False) as client:
    ready = client.get('/health/ready'); ready.raise_for_status()
    # A known identifier pair is loaded from the repository, not assumed.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pathdsp.data import build_dataset
    bundle = build_dataset('GDSC', include_examples=False)
    pair = {'drug':bundle.drug_ids[0], 'cell':bundle.cell_ids[0]}
    def request(count):
        started = time.perf_counter()
        response = client.post('/v1/predict', json={'dataset':'GDSC','pairs':[pair] * count})
        response.raise_for_status()
        assert len(response.json()['results']) == count
        return (time.perf_counter()-started)*1000
    request(1)
    single = [request(1) for _ in range(30)]
    batch = [request(256) for _ in range(5)]
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=8) as executor:
        concurrent = list(executor.map(request, [1]*64))
    duration = time.perf_counter()-start
    invalid = client.post('/v1/predict', json={'pairs':[{'drug':'invalid','cell':'invalid'}]})
    assert invalid.status_code == 422
    def summary(values):
        return {'requests':len(values), 'p50_ms':float(np.percentile(values,50)), 'p95_ms':float(np.percentile(values,95)), 'max_ms':float(max(values))}
    report = {'scope':'Local HTTP benchmark, one warmed API process, OMP_NUM_THREADS=2. Synthetic repeat requests, not a clinical or internet-scale readiness claim.',
              'python':platform.python_version(), 'platform':platform.system(), 'model_version':ready.json()['model_version'],
              'single_pair':summary(single), 'batch_256_pairs':summary(batch),
              'concurrent_8_clients':dict(summary(concurrent), successful_requests=64, requests_per_second=64/duration),
              'invalid_identifiers_status':invalid.status_code}
    path=args.output
    if path.exists() and path.resolve() != (Path(__file__).resolve().parents[1]/'outputs/runtime_benchmark.json').resolve():
        raise ValueError('Refusing to overwrite a recorded benchmark')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
