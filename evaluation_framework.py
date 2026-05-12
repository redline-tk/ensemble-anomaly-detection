from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import wilcoxon
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings('ignore')

try:
    import scikit_posthocs as sp
    POSTHOCS_AVAILABLE = True
except ImportError:
    POSTHOCS_AVAILABLE = False
    print("Warning: scikit_posthocs not found. Install with: pip install scikit-posthocs")

try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

from pyod.models.iforest import IForest
from pyod.models.inne import INNE
from pyod.models.lof import LOF
from pyod.models.knn import KNN
from pyod.models.hbos import HBOS
from pyod.models.mcd import MCD
from pyod.models.copod import COPOD
from pyod.models.pca import PCA as PyOD_PCA
from pyod.models.vae import VAE
from pyod.models.deep_svdd import DeepSVDD
from pyod.models.lunar import LUNAR

import torch as _torch

def _detect_gpus() -> dict:
    if not _torch.cuda.is_available():
        return {'available': False, 'count': 0,
                'vae_device': 'cpu', 'svdd_device': 'cpu', 'lunar_device': 'cpu'}
    n = _torch.cuda.device_count()
    names = [_torch.cuda.get_device_properties(i).name for i in range(n)]
    vram  = [_torch.cuda.get_device_properties(i).total_memory / 1024**3 for i in range(n)]
    vae_dev   = 'cuda:0'
    svdd_dev  = 'cuda:0'
    lunar_dev = 'cuda:1' if n > 1 else 'cuda:0'
    return {
        'available': True, 'count': n,
        'names': names, 'vram_gb': vram,
        'vae_device': vae_dev,
        'svdd_device': svdd_dev,
        'lunar_device': lunar_dev,
    }

GPU_CONFIG = _detect_gpus()

def _gpu_summary() -> str:
    g = GPU_CONFIG
    if not g['available']:
        return "No GPU — running all detectors on CPU"
    lines = [f"{g['count']} GPU(s) detected:"]
    for i, (name, vram) in enumerate(zip(g['names'], g['vram_gb'])):
        role = []
        if f'cuda:{i}' == g['vae_device']:   role.append('VAE')
        if f'cuda:{i}' == g['svdd_device']:  role.append('DeepSVDD')
        if f'cuda:{i}' == g['lunar_device']: role.append('LUNAR')
        lines.append(f"  GPU {i}: {name}  ({vram:.0f} GB)  → {', '.join(role) or 'standby'}")
    return '\n'.join(lines)

DETECTOR_PARADIGMS = {
    'IsolationForest': 'Tree isolation',
    'INNE':            'Tree isolation',
    'LOF':             'Density/distance',
    'KNN':             'Density/distance',
    'HBOS':            'Statistical',
    'MCD':             'Statistical',
    'COPOD':           'Probabilistic',
    'PCA':             'Subspace',
    'VAE':             'Deep learning',
    'DeepSVDD':        'Deep learning',
    'LUNAR':           'Graph/relational',
}

DETECTOR_DIVERSITY_ORDER = [
    'IsolationForest',
    'LOF',
    'HBOS',
    'COPOD',
    'VAE',
    'KNN',
    'MCD',
    'INNE',
    'PCA',
    'DeepSVDD',
    'LUNAR',
]

def build_detector(name: str, n_features: int, contamination: float = 0.05,
                   random_state: int = 42, quick: bool = False,
                   n_samples: int = 10000):
    epochs = 10 if quick else 50
    kw = dict(contamination=contamination)

    if name == 'IsolationForest':
        return IForest(n_estimators=100, random_state=random_state, **kw)

    elif name == 'INNE':
        n_est = 100 if n_samples > 50000 else 200
        return INNE(n_estimators=n_est, random_state=random_state, **kw)

    elif name == 'LOF':
        k = min(20, max(2, n_samples // 10))
        return LOF(n_neighbors=k, **kw)

    elif name == 'KNN':
        k = min(5, max(2, n_samples // 20))
        return KNN(n_neighbors=k, **kw)

    elif name == 'HBOS':
        return HBOS(n_bins=10, **kw)

    elif name == 'MCD':
        if n_samples < 5 * n_features:
            raise ValueError(
                f"MCD skipped: n_samples={n_samples} < 5*n_features={5*n_features} "
                f"(singular covariance matrix)"
            )
        return MCD(random_state=random_state, **kw)

    elif name == 'COPOD':
        return COPOD(**kw)

    elif name == 'PCA':
        return PyOD_PCA(**kw)

    elif name == 'VAE':
        h1 = max(16, n_features * 2)
        h2 = max(16, n_features)
        enc = [h1, h2]
        dec = enc[::-1]
        return VAE(encoder_neuron_list=enc, decoder_neuron_list=dec,
                   epoch_num=epochs, verbose=0,
                   device=GPU_CONFIG['vae_device'], **kw)

    elif name == 'DeepSVDD':
        hidden = [max(16, n_features * 2), max(16, n_features)]
        return DeepSVDD(n_features=n_features, hidden_neurons=hidden,
                        epochs=epochs, verbose=0,
                        random_state=random_state, **kw)

    elif name == 'LUNAR':
        return LUNAR(n_epochs=epochs, verbose=0, **kw)

    else:
        raise ValueError(f"Unknown detector: {name}")

ODDS_DATASETS = [
    'cardio',
    'lympho',
    'glass',
    'ionosphere',
    'letter',
    'mammography',
    'musk',
    'optdigits',
    'pendigits',
    'pima',
    'satellite',
    'satimage-2',
    'shuttle',
    'speech',
    'thyroid',
    'vertebral',
    'wbc',
    'annthyroid',
    'arrhythmia',
    'breastw',
]

def load_odds(name: str, data_dir: Path) -> Optional[Tuple[np.ndarray, np.ndarray, dict]]:
    path = data_dir / f'{name}.mat'
    if not path.exists():
        return None
    data = loadmat(str(path))
    X = data['X'].astype(np.float64)
    y = data['y'].astype(np.int32).ravel()
    info = {
        'name': name,
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'n_anomalies': int(y.sum()),
        'anomaly_rate': float(y.mean()),
    }
    return X, y, info

def load_all_datasets(data_dir: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray, dict]]:
    datasets = {}
    for name in ODDS_DATASETS:
        result = load_odds(name, data_dir)
        if result is not None:
            datasets[name] = result
    for path in sorted(data_dir.glob('*.mat')):
        name = path.stem
        if name not in datasets:
            result = load_odds(name, data_dir)
            if result is not None:
                datasets[name] = result
    if not datasets:
        raise FileNotFoundError(
            f"No .mat files with 'X' and 'y' keys found in '{data_dir}'.\n"
            "Download ODDS datasets from:\n"
            "  https://github.com/Minqi824/ADBench/tree/main/adbench/datasets/Classical\n"
            "Place .mat files in the --data_dir directory."
        )
    print(f"Loaded {len(datasets)} datasets: {', '.join(datasets.keys())}")
    return datasets

def estimate_contamination_gmm(
    ensemble_scores: np.ndarray,
    bounds: Tuple[float, float] = (0.01, 0.40),
    random_state: int = 42,
) -> Tuple[float, dict]:
    scores = ensemble_scores.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=random_state, max_iter=200)
    gmm.fit(scores)

    means   = gmm.means_.ravel()
    weights = gmm.weights_.ravel()

    anom_idx = int(np.argmax(means))
    raw_estimate = float(weights[anom_idx])
    rho_hat = float(np.clip(raw_estimate, bounds[0], bounds[1]))

    diagnostics = {
        'gmm_means': means.tolist(),
        'gmm_weights': weights.tolist(),
        'raw_estimate': raw_estimate,
        'rho_hat': rho_hat,
        'clipped': raw_estimate != rho_hat,
    }
    return rho_hat, diagnostics

def compute_em_weights(
    binary_predictions: np.ndarray,
    min_weight: float = 0.01,
) -> np.ndarray:
    n_detectors, n_samples = binary_predictions.shape
    majority = (binary_predictions.mean(axis=0) >= 0.5).astype(int)

    weights = np.zeros(n_detectors)
    for m in range(n_detectors):
        p_m = binary_predictions[m].mean()
        expected_agreement = p_m ** 2 + (1 - p_m) ** 2
        observed_agreement = (binary_predictions[m] == majority).mean()
        lift = max(0.0, observed_agreement - expected_agreement)
        weights[m] = lift

    weights = np.maximum(weights, min_weight)
    weights /= weights.sum()
    return weights

@dataclass
class EnsembleResult:
    dataset:          str
    k:                int
    weighting_scheme: str
    rho_hat:          float
    rho_diagnostics:  dict
    detector_scores:  Dict[str, List[float]]
    detector_preds:   Dict[str, List[int]]
    detector_weights: Dict[str, float]
    vote_counts:      List[int]
    ensemble_scores:  List[float]
    predictions:      List[int]
    f1:                float = 0.0
    precision:         float = 0.0
    recall:            float = 0.0
    roc_auc:           float = 0.0
    avg_precision:     float = 0.0
    runtime_sec:       float = 0.0

CPU_DETECTORS  = ['IsolationForest', 'INNE', 'LOF', 'KNN', 'HBOS', 'MCD', 'COPOD', 'PCA']
DEEP_DETECTORS = ['VAE', 'DeepSVDD', 'LUNAR']

_SVDD_GPU_IDX  = int(GPU_CONFIG['svdd_device'].replace('cuda:', '')) \
                 if GPU_CONFIG['available'] else 0
_LUNAR_GPU_IDX = int(GPU_CONFIG['lunar_device'].replace('cuda:', '')) \
                 if GPU_CONFIG['available'] else 0

def _fit_one_cpu(name: str, X: np.ndarray, n_features: int,
                 contamination: float, random_state: int, quick: bool) -> Tuple[str, Optional[np.ndarray]]:
    import warnings; warnings.filterwarnings('ignore')
    try:
        det = build_detector(name, n_features, contamination, random_state,
                             quick, n_samples=len(X))
        det.fit(X)
        return name, det.decision_scores_.copy()
    except ValueError as e:
        print(f"    [SKIP] {name}: {e}")
        return name, None

def _fit_deep_detectors(names: List[str], X: np.ndarray, n_features: int,
                        contamination: float, random_state: int,
                        quick: bool) -> Dict[str, np.ndarray]:
    import os, warnings; warnings.filterwarnings('ignore')
    results = {}
    for name in names:
        if name == 'VAE':
            det = build_detector(name, n_features, contamination, random_state, quick)
            det.fit(X)
            results[name] = det.decision_scores_.copy()

        elif name == 'DeepSVDD':
            orig = os.environ.get('CUDA_VISIBLE_DEVICES', None)
            if GPU_CONFIG['available']:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(_SVDD_GPU_IDX)
            try:
                det = build_detector(name, n_features, contamination, random_state, quick)
                det.fit(X)
                results[name] = det.decision_scores_.copy()
            finally:
                if orig is None:
                    os.environ.pop('CUDA_VISIBLE_DEVICES', None)
                else:
                    os.environ['CUDA_VISIBLE_DEVICES'] = orig

        elif name == 'LUNAR':
            orig = os.environ.get('CUDA_VISIBLE_DEVICES', None)
            if GPU_CONFIG['available']:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(_LUNAR_GPU_IDX)
            try:
                det = build_detector(name, n_features, contamination, random_state, quick)
                det.fit(X)
                results[name] = det.decision_scores_.copy()
            finally:
                if orig is None:
                    os.environ.pop('CUDA_VISIBLE_DEVICES', None)
                else:
                    os.environ['CUDA_VISIBLE_DEVICES'] = orig
    return results

def run_ensemble(
    X: np.ndarray,
    y_true: Optional[np.ndarray],
    detector_names: List[str],
    k: int,
    weighting_scheme: str,
    dataset_name: str,
    random_state: int = 42,
    quick: bool = False,
    n_jobs: int = -1,
) -> EnsembleResult:
    t0 = time.time()
    n_samples, n_features = X.shape

    MAX_SAMPLES = 50_000
    X_fit = X
    y_fit = y_true
    subsample_idx = None
    if n_samples > MAX_SAMPLES:
        rng = np.random.RandomState(random_state)
        subsample_idx = rng.choice(n_samples, MAX_SAMPLES, replace=False)
        subsample_idx.sort()
        X_fit = X[subsample_idx]
        y_fit = y_true[subsample_idx] if y_true is not None else None
        print(f"    [SUBSAMPLE] {dataset_name}: {n_samples}→{MAX_SAMPLES} samples")

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_fit)

    cpu_names  = [n for n in detector_names if n in CPU_DETECTORS]
    deep_names = [n for n in detector_names if n in DEEP_DETECTORS]

    raw_scores: Dict[str, np.ndarray] = {}

    if cpu_names:
        n_workers = n_jobs if n_jobs > 0 else -1
        if JOBLIB_AVAILABLE and len(cpu_names) > 1:
            results_cpu = Parallel(n_jobs=n_workers, prefer='threads')(
                delayed(_fit_one_cpu)(name, X_scaled, n_features,
                                      0.1, random_state, quick)
                for name in cpu_names
            )
        else:
            results_cpu = [_fit_one_cpu(name, X_scaled, n_features,
                                         0.1, random_state, quick)
                           for name in cpu_names]
        for name, scores in results_cpu:
            if scores is not None:
                raw_scores[name] = scores

    if deep_names:
        deep_results = _fit_deep_detectors(deep_names, X_scaled, n_features,
                                           0.1, random_state, quick)
        raw_scores.update(deep_results)

    if not raw_scores:
        raise RuntimeError(f"All detectors failed on dataset '{dataset_name}'")

    active_detectors = list(raw_scores.keys())

    norm_scores = {}
    for name, s in raw_scores.items():
        lo, hi = s.min(), s.max()
        norm_scores[name] = (s - lo) / (hi - lo + 1e-10)

    mean_ensemble_score = np.mean(list(norm_scores.values()), axis=0)
    rho_hat, rho_diag = estimate_contamination_gmm(mean_ensemble_score)

    binary_preds = {}
    for name, s in raw_scores.items():
        threshold = np.quantile(s, 1.0 - rho_hat)
        binary_preds[name] = (s > threshold).astype(int)

    preds_matrix = np.array([binary_preds[n] for n in active_detectors])

    if weighting_scheme == 'em':
        weights = compute_em_weights(preds_matrix)
    elif weighting_scheme == 'equal':
        weights = np.ones(len(active_detectors)) / len(active_detectors)
    elif weighting_scheme == 'agreement':
        majority = (preds_matrix.mean(axis=0) >= 0.5).astype(int)
        agreement = np.array([(preds_matrix[m] == majority).mean()
                               for m in range(len(active_detectors))])
        agreement = np.maximum(agreement, 0.01)
        weights = agreement / agreement.sum()
    else:
        raise ValueError(f"Unknown weighting_scheme: {weighting_scheme}")

    weight_dict = {n: float(w) for n, w in zip(active_detectors, weights)}

    n_fit = len(X_scaled)
    ensemble_score = np.zeros(n_fit)
    for i, name in enumerate(active_detectors):
        ensemble_score += weights[i] * norm_scores[name]

    k_eff = min(k, len(active_detectors))
    vote_counts = preds_matrix.sum(axis=0)
    final_preds = (vote_counts >= k_eff).astype(int)

    y_eval = y_fit if y_fit is not None else y_true

    result = EnsembleResult(
        dataset=dataset_name,
        k=k_eff,
        weighting_scheme=weighting_scheme,
        rho_hat=rho_hat,
        rho_diagnostics=rho_diag,
        detector_scores={n: norm_scores[n].tolist() for n in active_detectors},
        detector_preds={n: binary_preds[n].tolist() for n in active_detectors},
        detector_weights=weight_dict,
        vote_counts=vote_counts.tolist(),
        ensemble_scores=ensemble_score.tolist(),
        predictions=final_preds.tolist(),
        runtime_sec=time.time() - t0,
    )

    if y_eval is not None:
        result.f1        = float(f1_score(y_eval, final_preds, zero_division=0))
        result.precision = float(precision_score(y_eval, final_preds, zero_division=0))
        result.recall    = float(recall_score(y_eval, final_preds, zero_division=0))
        if len(np.unique(y_eval)) > 1:
            result.roc_auc       = float(roc_auc_score(y_eval, ensemble_score))
            result.avg_precision = float(average_precision_score(y_eval, ensemble_score))

    return result

def evaluate_standalone(
    X: np.ndarray,
    y_true: np.ndarray,
    detector_name: str,
    rho_hat: float,
    n_features: int,
    random_state: int = 42,
    quick: bool = False,
) -> dict:
    t0 = time.time()
    n_samples = len(X)

    MAX_SAMPLES = 50_000
    if n_samples > MAX_SAMPLES:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(n_samples, MAX_SAMPLES, replace=False)
        idx.sort()
        X = X[idx]
        y_true = y_true[idx]
        n_samples = MAX_SAMPLES

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    try:
        det = build_detector(detector_name, n_features,
                             contamination=rho_hat,
                             random_state=random_state,
                             quick=quick,
                             n_samples=n_samples)
        det.fit(X_scaled)
    except ValueError as e:
        print(f"    [SKIP] {detector_name}: {e}")
        return {
            'detector': detector_name, 'f1': float('nan'),
            'precision': float('nan'), 'recall': float('nan'),
            'roc_auc': float('nan'), 'avg_precision': float('nan'),
            'runtime_sec': float(time.time() - t0),
        }

    scores = det.decision_scores_
    threshold = np.quantile(scores, 1.0 - rho_hat)
    preds = (scores > threshold).astype(int)
    out = {
        'detector':      detector_name,
        'f1':            float(f1_score(y_true, preds, zero_division=0)),
        'precision':     float(precision_score(y_true, preds, zero_division=0)),
        'recall':        float(recall_score(y_true, preds, zero_division=0)),
        'runtime_sec':   float(time.time() - t0),
    }
    if len(np.unique(y_true)) > 1:
        out['roc_auc']       = float(roc_auc_score(y_true, scores))
        out['avg_precision'] = float(average_precision_score(y_true, scores))
    return out

def ablation_k(
    datasets: Dict,
    k_values: List[int],
    weighting_scheme: str = 'em',
    detector_names: List[str] = None,
    output_dir: Path = None,
    quick: bool = False,
) -> pd.DataFrame:
    if detector_names is None:
        detector_names = DETECTOR_DIVERSITY_ORDER

    records = []
    for ds_name, (X, y_true, info) in datasets.items():
        print(f"  k-ablation | {ds_name} ({info['n_samples']}×{info['n_features']}, "
              f"anom={info['anomaly_rate']:.1%})")
        for k in k_values:
            res = run_ensemble(X, y_true, detector_names, k,
                               weighting_scheme, ds_name, quick=quick)
            records.append({
                'dataset':        ds_name,
                'k':              k,
                'f1':              res.f1,
                'precision':       res.precision,
                'recall':          res.recall,
                'roc_auc':         res.roc_auc,
                'avg_precision':   res.avg_precision,
                'rho_hat':         res.rho_hat,
                'runtime_sec':     res.runtime_sec,
            })

    df = pd.DataFrame(records)

    if output_dir:
        df.to_csv(output_dir / 'ablation_k.csv', index=False)
        _plot_k_ablation(df, k_values, output_dir)

    return df

def _plot_k_ablation(df: pd.DataFrame, k_values: List[int], output_dir: Path):
    summary = df.groupby('k')[['precision', 'recall', 'f1']].agg(['mean', 'std'])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {'precision': '#1f77b4', 'recall': '#ff7f0e', 'f1': '#2ca02c'}
    for metric, color in colors.items():
        means = summary[(metric, 'mean')].values
        stds  = summary[(metric, 'std')].values
        ax.plot(k_values, means, marker='o', color=color, label=metric.capitalize())
        ax.fill_between(k_values, means - stds, means + stds,
                        alpha=0.15, color=color)
    ax.set_xlabel('Voting threshold k  (minimum detectors required)', fontsize=11)
    ax.set_ylabel('Score (macro-averaged over datasets)', fontsize=11)
    ax.set_title('Ablation Study: Effect of Voting Threshold k', fontsize=12)
    ax.set_xticks(k_values)
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / 'ablation_k.pdf', dpi=150)
    fig.savefig(output_dir / 'ablation_k.png', dpi=150)
    plt.close(fig)
    print(f"  → Saved ablation_k.pdf / .png")