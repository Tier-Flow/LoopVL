"""Download SHA-256-pinned findings inputs from the public GitHub Release."""
from pathlib import Path
import sys
_FINDINGS_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "common" / "paths.py").is_file())
sys.path.insert(0, str(_FINDINGS_ROOT))
from common.paths import DATA_ROOT, repo_path
import argparse
import concurrent.futures
import json

REPO = _FINDINGS_ROOT.parent
sys.path.insert(0, str(REPO))
from scripts.release_assets import download_asset, install_file, is_valid


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--asset',choices=['benchmarks'],default='benchmarks')
    p.add_argument('--output',type=repo_path,default=None)
    p.add_argument('--only',nargs='*',help='Optional exact local file names (benchmarks only)')
    p.add_argument('--cache-root',type=Path,default=REPO.parent/'.loopvl_download_cache')
    p.add_argument('--release-base-url',help='Override the pinned GitHub Release URL (for mirrors/testing)')
    args=p.parse_args()
    repo_root=Path(__file__).resolve().parent
    dest=repo_path(args.output) if args.output else DATA_ROOT
    cache_root=(args.cache_root if args.cache_root.is_absolute() else REPO/args.cache_root).resolve()
    if dest==repo_root or repo_root in dest.parents:
        p.error('Keep downloads outside findings/; use model/ or data/')
    manifest=json.loads((repo_root/'common'/'benchmarks_manifest.json').read_text())
    release_path=(repo_root/'common'/manifest['release_assets']).resolve()
    if REPO not in release_path.parents:raise ValueError('Release manifest must stay inside the repository')
    release=json.loads(release_path.read_text())
    rows=manifest['files']
    if args.only:
        unknown=set(args.only)-{r['local'] for r in rows}
        if unknown:p.error('Unknown file selection: '+str(unknown))
        rows=[r for r in rows if r['local'] in args.only]
    def fetch(row):
        local = Path(row['local'])
        if local.suffix=='.tsv':
            local = Path('VLMEvalData') / local
        target=(dest/local).resolve()
        if dest not in target.parents:raise ValueError('Unsafe manifest path')
        spec={'bytes':row['bytes'],'sha256':row['sha256']}
        if release['assets'].get(row['asset'])!=spec:
            raise ValueError('Findings and Release manifests disagree: '+row['asset'])
        if is_valid(target,spec):
            return row['local']+' (verified cached)'
        cached=download_asset(release,row['asset'],cache_root/release['tag'],
                              base_url=args.release_base_url)
        install_file(cached,target,spec,dest)
        return row['local']
    dest.mkdir(parents=True,exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(fetch,rows):print(result,flush=True)
    print(f'Verified {len(rows)} files; destination: {dest}')


if __name__=='__main__':main()
