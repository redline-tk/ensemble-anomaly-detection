from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'requests', '-q'])
    import requests

try:
    import numpy as np
    from scipy.io import loadmat
    VERIFY_AVAILABLE = True
except ImportError:
    VERIFY_AVAILABLE = False

DATASETS = [
    ('annthyroid',  7200,   7.4),
    ('arrhythmia',   452,  14.6),
    ('breastw',      683,  34.5),
    ('cardio',      1831,   9.6),
    ('glass',        214,   4.2),
    ('ionosphere',   351,  35.9),
    ('letter',      1600,   6.3),
    ('lympho',       148,   4.1),
    ('mammography',  11183,  2.3),
    ('musk',         3062,   3.2),
    ('optdigits',    5216,   3.0),
    ('pendigits',    6870,   2.3),
    ('pima',          768,  34.9),
    ('satellite',    6435,  31.6),
    ('satimage-2',   5803,   1.2),
    ('mnist',     49097,   7.2),
    ('speech',       3686,   1.7),
    ('thyroid',      3772,   2.5),
    ('vowels',      1456,   3.4),
    ('vertebral',     240,  12.5),
    ('wbc',           378,   5.6),
]

GITHUB_BASE = (
    "https://raw.githubusercontent.com/Minqi824/ADBench/"
    "main/adbench/datasets/Classical"
)

ODDS_BASE = "https://odds.cs.stonybrook.edu/data"

def download_file(url: str, dest: Path, retries: int = 3, timeout: int = 60) -> bool:
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=timeout, stream=True)
            if resp.status_code == 200:
                dest.parent.mkdir(parents=True, exist_ok=True)
                with open(dest, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
            elif resp.status_code == 404:
                return False
            else:
                if attempt < retries:
                    time.sleep(2 ** attempt)
        except requests.exceptions.RequestException:
            if attempt == retries:
                return False
            time.sleep(2 ** attempt)
    return False

def verify_dataset(path: Path) -> dict:
    data = loadmat(str(path))
    if 'X' not in data or 'y' not in data:
        raise ValueError(f"Missing 'X' or 'y' keys in {path.name}")
    X = data['X'].astype(float)
    y = data['y'].ravel().astype(int)
    if len(X) != len(y):
        raise ValueError(f"X/y length mismatch in {path.name}")
    return {
        'n_samples':    len(X),
        'n_features':   X.shape[1],
        'n_anomalies':  int(y.sum()),
        'anomaly_rate': float(y.mean()),
        'size_kb':      path.stat().st_size // 1024,
    }

def download_all(output_dir: Path, verify: bool = False, force: bool = False):
    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(DATASETS)
    downloaded = 0
    skipped = 0
    failed = []

    print(f"\n{'='*62}")
    print(f" Downloading {total} ODDS benchmark datasets")
    print(f" Output directory: {output_dir.resolve()}")
    print(f"{'='*62}\n")

    for i, (name, expected_n, expected_anom_pct) in enumerate(DATASETS, 1):
        dest = output_dir / f'{name}.mat'
        prefix = f"  [{i:02d}/{total}] {name:<15}"

        if dest.exists() and not force:
            size_kb = dest.stat().st_size // 1024
            print(f"{prefix} SKIP  (already exists, {size_kb} KB)")
            skipped += 1
            continue

        url_primary = f"{GITHUB_BASE}/{name}.mat"
        success = download_file(url_primary, dest)

        if not success:
            url_fallback = f"{ODDS_BASE}/{name}.mat"
            success = download_file(url_fallback, dest)

        if success:
            size_kb = dest.stat().st_size // 1024
            if verify and VERIFY_AVAILABLE:
                try:
                    info = verify_dataset(dest)
                    rate_str = f"{info['anomaly_rate']:.1%}"
                    print(f"{prefix} OK    "
                          f"({size_kb} KB | n={info['n_samples']:>6} | "
                          f"d={info['n_features']:>3} | anom={rate_str})")
                except Exception as e:
                    print(f"{prefix} WARN  downloaded but verify failed: {e}")
            else:
                print(f"{prefix} OK    ({size_kb} KB)")
            downloaded += 1
        else:
            print(f"{prefix} FAIL  (not found at primary or fallback URL)")
            failed.append(name)

    print(f"\n{'='*62}")
    print(f" Summary")
    print(f"{'='*62}")
    print(f"  Downloaded : {downloaded}")
    print(f"  Skipped    : {skipped} (already present)")
    print(f"  Failed     : {len(failed)}")
    
    if failed:
        print(f"\n  Failed datasets: {', '.join(failed)}")
        print(f"\n  For failed datasets, download manually from:")
        print(f"    https://odds.cs.stonybrook.edu/")
        print(f"  and place the .mat files in: {output_dir.resolve()}")

    n_present = sum(1 for name, _, _ in DATASETS
                    if (output_dir / f'{name}.mat').exists())
    print(f"\n  Total present: {n_present}/{total} datasets")
    print(f"  Ready to run: {'YES' if n_present >= 10 else 'NO (need >=10)'}")
    print(f"{'='*62}\n")

    return len(failed) == 0

def print_dataset_table(output_dir: Path):
    if not VERIFY_AVAILABLE:
        print("scipy not available -- cannot read .mat files for verification")
        return

    rows = []
    for name, _, _ in DATASETS:
        path = output_dir / f'{name}.mat'
        if path.exists():
            try:
                info = verify_dataset(path)
                rows.append((name, info['n_samples'], info['n_features'],
                              info['n_anomalies'], info['anomaly_rate']))
            except Exception as e:
                rows.append((name, '?', '?', '?', f'ERROR: {e}'))

    if not rows:
        print("No datasets found in", output_dir)
        return

    print(f"\n{'Dataset':<16} {'n':>7} {'d':>4} {'Anomalies':>10} {'Anom. %':>8}")
    print("-" * 50)
    for name, n, d, na, rate in rows:
        if isinstance(rate, float):
            print(f"{name:<16} {n:>7} {d:>4} {na:>10} {rate*100:>7.1f}%")
        else:
            print(f"{name:<16} {str(n):>7} {str(d):>4} {str(na):>10} {rate:>8}")
    print(f"\nTotal: {len(rows)} datasets\n")

def main():
    parser = argparse.ArgumentParser(
        description='Download ODDS benchmark datasets for anomaly detection evaluation')
    parser.add_argument('--output_dir', type=str, default='./odds_data',
                        help='Directory to save .mat files (default: ./odds_data)')
    parser.add_argument('--verify',  action='store_true',
                        help='Verify each file after download (load and check X/y)')
    parser.add_argument('--force',   action='store_true',
                        help='Re-download even if files already exist')
    parser.add_argument('--table',   action='store_true',
                        help='Print dataset summary table (requires scipy)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.table:
        print_dataset_table(output_dir)
        return

    success = download_all(output_dir, verify=args.verify, force=args.force)

    if args.verify or True:
        print_dataset_table(output_dir)

    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()