import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy.stats import friedmanchisquare

plt.rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['DejaVu Serif'],
    'font.weight':        'bold',
    'font.size':          12,
    'axes.titlesize':     16,
    'axes.labelsize':     14,
    'axes.titleweight':   'bold',
    'axes.labelweight':   'bold',
    'xtick.labelsize':    12,
    'ytick.labelsize':    12,
    'legend.fontsize':    12,
    'figure.dpi':         150,
    'axes.grid':          True,
    'grid.alpha':         0.3,
    'grid.linestyle':     '--',
})

Y_FMT = mticker.FormatStrFormatter('%.2f')

COLORS = {
    'f1':        '#2ca02c',
    'precision': '#1f77b4',
    'recall':    '#ff7f0e',
    'auroc':     '#1f77b4',
    'auprc':     '#ff7f0e',
    'ensemble':  '#d62728',
    'baseline':  '#1f77b4',
}

def save(fig, output_dir, name):
    for ext in ('pdf', 'png'):
        fig.savefig(output_dir / f'{name}.{ext}', dpi=150, bbox_inches='tight')
    plt.close(fig)

def plot_cd_diagram(results_dir, output_dir):
    b = pd.read_csv(results_dir / 'baselines.csv')
    e = pd.read_csv(results_dir / 'ensemble_results.csv')
    pivot_b   = b.pivot_table(index='dataset', columns='detector', values='f1')
    pivot_em  = e[e['scheme']=='em'].set_index('dataset')[['f1']].rename(columns={'f1':'Ensemble-EM'})
    pivot_eq  = e[e['scheme']=='equal'].set_index('dataset')[['f1']].rename(columns={'f1':'Ensemble-Equal'})
    pivot_ncl = e[e['scheme']=='ncl'].set_index('dataset')[['f1']].rename(columns={'f1':'Ensemble-NCL'})
    pivot_var = e[e['scheme']=='agreement'].set_index('dataset')[['f1']].rename(columns={'f1':'Ensemble-Var'})
    f1 = pd.concat([pivot_b, pivot_em, pivot_eq, pivot_ncl, pivot_var], axis=1).dropna()
    methods = f1.columns.tolist()
    ranks   = f1.rank(axis=1, ascending=False).mean().sort_values()
    n, k    = len(f1), len(methods)
    nemenyi = sp.posthoc_nemenyi_friedman(f1.values)
    nemenyi.index   = methods
    nemenyi.columns = methods
    q_table = {15: 3.391, 16: 3.479, 17: 3.507}
    q_alpha = q_table.get(k, 3.479)
    cd = q_alpha * np.sqrt(k * (k + 1) / (6 * n))
    methods_s = ranks.index.tolist()
    ranks_s   = ranks.values
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0.5, k + 0.5)
    ax.set_ylim(-1.5, k + 1.5)
    ax.set_yticks([])
    ax.set_xlabel('Average Rank')
    ax.set_title(f'Critical Difference Diagram (F1 Score)\n(CD = {cd:.2f}, α = 0.05)')
    ax.axhline(k + 0.5, color='black', linewidth=2)
    
    for r in range(1, k + 1):
        ax.plot([r, r], [k + 0.3, k + 0.7], color='black', linewidth=1.5)
        ax.text(r, k + 0.85, str(r), ha='center', fontsize=12, fontweight='bold')
    
    y_pos = np.linspace(0.2, k - 0.2, k)
    DARK_RED, BLACK = '#8B0000', '#000000'
    
    for i, (m, r) in enumerate(zip(methods_s, ranks_s)):
        is_ens = 'Ensemble' in m
        label_color = DARK_RED if is_ens else BLACK
        dot_color = COLORS['ensemble'] if is_ens else COLORS['baseline']
        y = y_pos[i]
        ax.plot([r, r], [k + 0.5, y], color='#cccccc', linewidth=1, linestyle='--')
        ax.plot(r, y, 'o', color=dot_color, markersize=8, zorder=5)
        ax.text(-0.1, y, f'{m} ({r:.2f})', ha='right', va='center', 
                fontsize=11, color=label_color, fontweight='bold')
                
    # Clean CD Arrow: Removed redundant text overlay
    ax.annotate('', xy=(ranks_s[0] + cd, k + 0.5), xytext=(ranks_s[0], k + 0.5), 
                arrowprops=dict(arrowstyle='<->', color='red', lw=2.5))
    
    drawn, bar_y = set(), k - 0.5
    for i, m1 in enumerate(methods_s):
        group = [m1]
        for m2 in methods_s[i + 1:]:
            if nemenyi.loc[m1, m2] > 0.05 and abs(ranks[m1] - ranks[m2]) < cd:
                group.append(m2)
        if len(group) > 1:
            key = frozenset(group)
            if key not in drawn:
                drawn.add(key)
                gr = [ranks[m] for m in group]
                ax.plot([min(gr), max(gr)], [bar_y, bar_y], color='#2ca02c', 
                        linewidth=5, alpha=0.7, solid_capstyle='round')
                bar_y -= 0.4
                
    ax.spines[['top', 'right', 'left']].set_visible(False)
    fig.tight_layout()
    save(fig, output_dir, 'cd_diagram')

def plot_k_ablation(results_dir, output_dir):
    df = pd.read_csv(results_dir / 'ablation_k.csv')
    fig, ax = plt.subplots(figsize=(8, 5))
    for metric, color, ls in [('f1', COLORS['f1'], '-'), ('precision', COLORS['precision'], '--'), ('recall', COLORS['recall'], ':')]:
        g = df.groupby('k')[metric]
        m, s = g.mean(), g.std()
        ax.plot(m.index, m.values, marker='o', color=color, linestyle=ls, label=metric.capitalize(), linewidth=2, markersize=8)
        ax.fill_between(m.index, m - s, m + s, alpha=0.15, color=color)
    ax.set_xticks([4, 5, 6, 7, 8])
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
    ax.yaxis.set_major_formatter(Y_FMT)
    ax.set_xlabel('Voting Threshold $k$')
    ax.set_ylabel('Score')
    ax.set_title('Impact of Voting Threshold $k$')
    ax.legend(loc='best', frameon=True)
    fig.tight_layout()
    save(fig, output_dir, 'ablation_k')

def plot_ensemble_size(results_dir, output_dir):
    df = pd.read_csv(results_dir / 'ablation_ensemble_size.csv')
    g  = df.groupby('n_detectors')['f1']
    m, s = g.mean(), g.std()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(m.index, m.values, marker='o', color=COLORS['baseline'], linewidth=2, markersize=8)
    ax.fill_between(m.index, m - s, m + s, alpha=0.15, color=COLORS['baseline'])
    labels = ['IF', 'LOF', 'HBOS', 'COPOD', 'VAE', 'KNN', 'MCD', 'INNE', 'PCA', 'DSVDD', 'LUNAR']
    for n_det, label in zip(m.index, labels):
        ax.annotate(label, (n_det, m[n_det]), textcoords='offset points', xytext=(0, 10), ha='center', fontsize=10, rotation=45, fontweight='bold')
    ax.set_xticks(list(m.index))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
    ax.yaxis.set_major_formatter(Y_FMT)
    ax.set_xlabel('Number of Detectors ($M$)')
    ax.set_ylabel('F1 Score')
    ax.set_title('Marginal Contribution of Detectors')
    fig.tight_layout()
    save(fig, output_dir, 'ablation_ensemble_size')

def plot_contamination_sensitivity(results_dir, output_dir):
    df = pd.read_csv(results_dir / 'ablation_contamination_sensitivity.csv')
    fig, ax = plt.subplots(figsize=(8, 5))
    for metric, color in [('f1', COLORS['f1']), ('precision', COLORS['precision']), ('recall', COLORS['recall'])]:
        g = df.groupby('delta')[metric].mean()
        ax.plot(g.index, g.values, marker='o', color=color, label=metric.capitalize(), linewidth=2, markersize=8)
    ax.axvline(0, color='gray', linestyle='--', linewidth=2, alpha=0.6, label='Nominal $\\hat{\\rho}$')
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(Y_FMT)
    ax.set_xlabel('Contamination Perturbation $\\delta$')
    ax.set_ylabel('Score')
    ax.set_title('Contamination Sensitivity Analysis')
    ax.legend(frameon=True)
    fig.tight_layout()
    save(fig, output_dir, 'ablation_contamination')

def plot_weighting(results_dir, output_dir):
    df = pd.read_csv(results_dir / 'ablation_weighting.csv')
    means = df.groupby('scheme')[['f1', 'roc_auc', 'avg_precision']].mean()
    means.columns = ['F1', 'AUROC', 'AUPRC']
    means = means.reindex(['equal', 'agreement', 'ncl', 'em'])
    means.index = ['Equal', 'Variance', 'NCL', 'EM']
    fig, ax = plt.subplots(figsize=(8, 5))
    x, w = np.arange(len(means)), 0.25
    for i, (col, color) in enumerate([('F1', COLORS['f1']), ('AUROC', COLORS['auroc']), ('AUPRC', COLORS['auprc'])]):
        ax.bar(x + i * w, means[col], w, label=col, color=color, alpha=0.85, edgecolor='black', linewidth=1)
    ax.set_xticks(x + w)
    ax.set_xticklabels(means.index)
    ax.yaxis.set_major_formatter(Y_FMT)
    ax.set_ylabel('Score')
    ax.set_title('Weighting Scheme Comparison')
    ax.legend(frameon=True)
    fig.tight_layout()
    save(fig, output_dir, 'ablation_weighting')

def plot_compact_tradeoff(results_dir, output_dir):
    df = pd.DataFrame([
        {'label': 'Ensemble', 'f1': 0.311, 'auprc': 0.437, 'cost': 100, 'color': '#7f7f7f'},
        {'label': 'IF+INNE+HBOS+KNN', 'f1': 0.301, 'auprc': 0.450, 'cost': 36, 'color': '#d62728'},
        {'label': 'IF+INNE+HBOS+KNN+MCD', 'f1': 0.317, 'auprc': 0.444, 'cost': 45, 'color': '#ff7f0e'},
        {'label': 'IF+HBOS+KNN', 'f1': 0.302, 'auprc': 0.446, 'cost': 27, 'color': '#2ca02c'},
        {'label': 'IF+INNE+MCD', 'f1': 0.320, 'auprc': 0.438, 'cost': 27, 'color': '#1f77b4'},
    ])

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    
    for ax, metric, title in zip(axes, ['auprc', 'f1'], ['AUPRC Score vs Cost', 'F1 Score vs Cost']):
        baseline = df[df['label'] == 'Ensemble'][metric].values[0]
        ax.axhline(baseline, color='black', linestyle='--', linewidth=2.5, alpha=0.5)
        
        for _, row in df.iterrows():
            ax.scatter(row['cost'], row[metric], color=row['color'], s=500, 
                       marker='o', edgecolors='black', linewidth=2, zorder=5)
            
            if row['label'] == 'Ensemble':
                ax.text(row['cost'], row[metric] + 0.0015, row['label'], 
                        fontsize=12, va='bottom', ha='center', fontweight='bold')
            else:
                ax.text(row['cost'] + 5, row[metric], row['label'], 
                        fontsize=12, va='center', ha='left', fontweight='bold')

        ax.set_title(title, pad=20)
        ax.set_xlabel('Computational Cost (%)')
        ax.set_ylabel(metric.upper() + ' Score')
        
        ax.set_xlim(15, 140)
        ax.set_ylim(df[metric].min() - 0.005, df[metric].max() + 0.005)
        
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
        ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout(pad=4.0)
    save(fig, output_dir, 'compact_tradeoff_high_vis')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--output_dir',  type=str, default='./figures')
    args = parser.parse_args()
    results_dir, output_dir = Path(args.results_dir), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_cd_diagram(results_dir, output_dir)
    plot_k_ablation(results_dir, output_dir)
    plot_ensemble_size(results_dir, output_dir)
    plot_contamination_sensitivity(results_dir, output_dir)
    plot_weighting(results_dir, output_dir)
    plot_compact_tradeoff(results_dir, output_dir)

if __name__ == '__main__':
    main()