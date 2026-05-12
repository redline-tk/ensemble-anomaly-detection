# Less Is More: Principled Diversity in Heterogeneous Anomaly Detection Ensembles

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official implementation for the paper:

> **Less Is More: Principled Diversity in Heterogeneous Anomaly Detection Ensembles**  
> Tea Krčmar, Dina Šabanović, Mirko Köhler, Ivica Lukić  
> *Under review*

## Overview

We propose a heterogeneous ensemble framework for unsupervised anomaly detection that:

- Combines **11 detectors** spanning 7 paradigmatically orthogonal anomaly signals
- Estimates contamination **adaptively per dataset** using a Gaussian Mixture Model
- Weights detectors using an **NCL-inspired scheme** that penalises correlated detectors
- Identifies through exhaustive search that a **compact 4-detector ensemble** {IsolationForest, INNE, HBOS, KNN} achieves 103% of full-ensemble AUPRC at 36% computational cost

Evaluated on 21 ODDS benchmark datasets with Friedman–Nemenyi statistical testing.

## Key Results

| Ensemble | Detectors | AUPRC | AUROC | F1 | Cost |
|---|---|---|---|---|---|
| Full-11-EM | 11 | 0.437 | 0.818 | 0.311 | 100% |
| **IF+INNE+HBOS+KNN** | **4** | **0.450** | **0.819** | 0.301 | **36%** |
| IF+INNE+HBOS+KNN+MCD | 5 | 0.444 | 0.822 | **0.317** | 45% |
| IF+HBOS+KNN | 3 | 0.446 | 0.815 | 0.302 | 27% |

Friedman test: χ² = 71.58, p < 0.001 — significant improvements over 6 individual detectors.

## Requirements

```bash
pip install -r requirements.txt
```

For GPU support (recommended for VAE, DeepSVDD, LUNAR):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/redline-tk/ensemble-anomaly-detection
cd ensemble-anomaly-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download datasets

Downloads all 22 ODDS benchmark datasets (~15 MB) from the ADBench repository:

```bash
python download_datasets.py --output_dir ./odds --verify
```

### 4. Run the full evaluation

```bash
python evaluation_framework.py \
    --data_dir ./odds \
    --output_dir ./results \
    --k 6 \
    --k_values 4,5,6,7,8 \
    --n_jobs -1
```

This runs all 8 phases:
- Phase 1: Standalone baselines (all 11 detectors + LSCP)
- Phase 2: Proposed ensemble (4 weighting schemes)
- Phase 3: Voting threshold ablation (k ∈ {4,5,6,7,8})
- Phase 4: Ensemble size ablation
- Phase 5: Weighting scheme ablation
- Phase 6: Contamination sensitivity analysis
- Phase 7: Friedman + Nemenyi statistical tests
- Phase 8: Publication tables and figures

Progress is logged to `results/run_progress.log`.

**Estimated runtime:** 3–6 hours on CPU; 1–2 hours with GPU.

### 5. Reproduce figures

```bash
python generate_figures.py \
    --results_dir ./results \
    --output_dir ./figures
```

## Compact Ensemble

To run only the recommended compact ensembles without the full evaluation:

```python
from evaluation_framework import run_ensemble, load_all_datasets
from pathlib import Path

datasets = load_all_datasets(Path('./odds'))
X, y, info = datasets['cardio']

result = run_ensemble(
    X, y,
    detector_names=['IsolationForest', 'INNE', 'HBOS', 'KNN'],
    k=6,
    weighting_scheme='equal',
    dataset_name='cardio',
)
print(f"F1: {result.f1:.4f}  AUROC: {result.roc_auc:.4f}  AUPRC: {result.avg_precision:.4f}")
```

## Repository Structure

```
├── evaluation_framework.py   # Main framework — all 8 evaluation phases
├── download_datasets.py      # Downloads 21 ODDS benchmark datasets
├── generate_figures.py       # Reproduces all publication figures
├── requirements.txt          # Python dependencies
├── results/                  # Pre-computed results CSVs
│   ├── baselines.csv
│   ├── ensemble_results.csv
│   ├── ablation_k.csv
│   ├── ablation_ensemble_size.csv
│   ├── ablation_weighting.csv
│   ├── ablation_contamination_sensitivity.csv
│   ├── all_3detector_combos.csv
│   ├── all_4detector_combos.csv
│   ├── final_ensemble_recommendations.csv
│   └── statistical_tests.json
└── figures/                  # Publication figures (PDF + PNG)
    ├── cd_diagram.pdf
    ├── ablation_k.pdf
    ├── ablation_ensemble_size.pdf
    ├── ablation_contamination_sensitivity.pdf
    ├── ablation_weighting.pdf
    └── compact_tradeoff.pdf
```

## Detectors

| Paradigm | Detector | Reference |
|---|---|---|
| Tree isolation | IsolationForest | Liu et al. 2008 |
| Tree isolation | INNE | Bandaragoda et al. 2018 |
| Density/distance | LOF | Breunig et al. 2000 |
| Density/distance | KNN | Ramaswamy et al. 2000 |
| Statistical | HBOS | Goldstein & Dengel 2012 |
| Statistical | MCD | Rousseeuw & Driessen 1999 |
| Probabilistic | COPOD | Li et al. 2020 |
| Subspace | PCA | Shyu et al. 2003 |
| Deep learning | VAE | An & Cho 2015 |
| Deep learning | DeepSVDD | Ruff et al. 2018 |
| Graph/relational | LUNAR | Goodge et al. 2022 |

## Affiliation

Faculty of Electrical Engineering, Computer Science and Information Technology Osijek  
Josip Juraj Strossmayer University of Osijek, Croatia  
📧 tea.krcmar@ferit.hr

## Datasets

Benchmark datasets are from the [ODDS repository](https://odds.cs.stonybrook.edu/) (Rayana, 2016).  
They are not included in this repository and are downloaded automatically via `download_datasets.py`.

## License

MIT License. See [LICENSE](LICENSE) for details.
