"""Download a reproducible, one-file-per-case research cohort from GDC.

The API and available files may change; the manifest records exact IDs. Downloaded
expression data stays local and must not be treated as synthetic or independent.
"""
import argparse
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API = 'https://api.gdc.cancer.gov'
FILTERS = {'op': 'and', 'content': [
    {'op': '=', 'content': {'field': 'cases.project.project_id', 'value': 'TCGA-LUAD'}},
    {'op': '=', 'content': {'field': 'data_type', 'value': 'Gene Expression Quantification'}},
    {'op': '=', 'content': {'field': 'analysis.workflow_type', 'value': 'STAR - Counts'}},
    {'op': '=', 'content': {'field': 'access', 'value': 'open'}},
]}


def stage_label(value):
    match = re.fullmatch(r'stage\s+(iv|iii|ii|i)(?:[abc])?', (value or '').strip(), re.I)
    return {'i': 0, 'ii': 1, 'iii': 2, 'iv': 3}.get(match.group(1).lower()) if match else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('data/gdc'))
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    response = requests.get(f'{API}/files', params={
        'filters': json.dumps(FILTERS), 'format': 'JSON', 'size': 2000,
        'fields': 'file_id,file_size,md5sum,cases.case_id,cases.diagnoses.ajcc_pathologic_stage,cases.samples.sample_type,cases.samples.submitter_id,associated_entities.entity_submitter_id',
    }, timeout=60)
    response.raise_for_status()
    payload = response.json()
    if payload.get('warnings'):
        raise ValueError(f"GDC query warnings: {payload['warnings']}")
    hits = payload['data']['hits']
    if len(hits) != payload['data']['pagination']['total']:
        raise ValueError('GDC response is incomplete; paginate before proceeding.')
    # Exactly one case, one unambiguous stage, and only primary tumor metadata.
    cases = {}
    for hit in sorted(hits, key=lambda item: item['file_id']):
        if len(hit.get('cases', [])) != 1:
            continue
        case = hit['cases'][0]
        entities = hit.get('associated_entities', [])
        matched_samples = [sample for sample in case.get('samples', [])
                           if sample.get('submitter_id') and any(
                               (entity.get('entity_submitter_id') or '').startswith(sample['submitter_id'] + '-')
                               for entity in entities)]
        stages = {stage_label(d.get('ajcc_pathologic_stage')) for d in case.get('diagnoses', [])}
        if len(matched_samples) != 1 or matched_samples[0].get('sample_type') != 'Primary Tumor' or None in stages or len(stages) != 1:
            continue
        case_id = case['case_id']
        cases.setdefault(case_id, {'case_id': case_id, 'file_id': hit['file_id'], 'stage': stages.pop(),
                                   'size': hit['file_size'], 'md5': hit['md5sum']})
    manifest = sorted(cases.values(), key=lambda item: item['case_id'])
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Selected {len(manifest)} unique cases from {len(hits)} files.', flush=True)
    def fetch(row):
        path = args.output / (row['file_id'] + '.tsv')
        def matches(content):
            return len(content) == row['size'] and hashlib.md5(content).hexdigest() == row['md5']
        if path.exists() and matches(path.read_bytes()):
            return
        retry = Retry(total=4, connect=4, read=4, backoff_factor=1,
                      status_forcelist=[429, 500, 502, 503, 504], allowed_methods=['GET'])
        with requests.Session() as session:
            session.mount('https://', HTTPAdapter(max_retries=retry))
            result = session.get(f"{API}/data/{row['file_id']}", timeout=(20, 120))
            result.raise_for_status()
        if not matches(result.content):
            raise ValueError(f"Size or MD5 mismatch for {row['file_id']}")
        if 'gene_id' not in result.text[:2000] or 'fpkm_unstranded' not in result.text[:2000]:
            raise ValueError(f"Unexpected content for {row['file_id']}")
        temporary = path.with_suffix('.part')
        temporary.write_bytes(result.content)
        temporary.replace(path)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for number, _ in enumerate(pool.map(fetch, manifest), 1):
            if number % 25 == 0 or number == len(manifest):
                print(f'Downloaded {number}/{len(manifest)}', flush=True)

if __name__ == '__main__':
    main()
