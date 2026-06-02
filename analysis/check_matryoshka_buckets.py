"""Check Matryoshka SAE bucket statistics for layer 12."""

import os
import pickle as pkl
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'
sns.set_style("ticks")

FEATURES_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'features', 'gemma-2-2b',
    'sae', 'gemma-2-2b-res-matryoshka-dc', 'mean', '12.pkl'
)

MATRYOSHKA_LEVELS = [128, 512, 2048, 8192, 32768]
BUCKET_BOUNDARIES = [0] + MATRYOSHKA_LEVELS


def check_buckets(activations):
    n_sentences, n_features = activations.shape
    print(f"Activation shape: {activations.shape}, dtype: {activations.dtype}")
    print(f"Overall sparsity: {(activations == 0).mean():.4f}")
    print(f"N sentences: {n_sentences}, N features: {n_features}\n")

    # --- Table 1: Activation statistics ---
    print("=" * 90)
    print("Activation statistics per bucket")
    print("=" * 90)
    header = (f"{'Bucket':<8} {'Range':<16} {'Size':>6} "
              f"{'Sparsity':>10} {'Mean':>10} {'Mean (nz)':>10} {'Max':>8}")
    print(header)
    print("-" * len(header))

    for i in range(len(BUCKET_BOUNDARIES) - 1):
        lo, hi = BUCKET_BOUNDARIES[i], BUCKET_BOUNDARIES[i + 1]
        chunk = activations[:, lo:hi]
        sparsity = (chunk == 0).mean()
        mean_act = chunk.mean()
        nz = chunk[chunk != 0]
        mean_nz = nz.mean() if len(nz) > 0 else 0.0
        max_act = chunk.max()

        print(f"{i:<8} {lo:>5d}-{hi-1:<5d}   {hi-lo:>6d} "
              f"{sparsity:>10.4f} {mean_act:>10.4f} {mean_nz:>10.4f} {max_act:>8.2f}")

    # --- Table 2: Variance and regression-relevant statistics ---
    # Per-feature variance (across sentences) determines how much a feature
    # can explain in a regression. Features with zero variance are useless.
    # Features with high sparsity are near-binary (0 vs value), limiting their
    # ability to capture graded neural responses.
    print(f"\n{'=' * 90}")
    print("Regression-relevant statistics per bucket")
    print("=" * 90)
    header2 = (f"{'Bucket':<8} {'Size':>6} {'N nz':>8} {'N dead':>8} {'% nz':>8} "
               f"{'Mean var':>10} {'Med var':>10} "
               f"{'Mean fire':>10} {'Med fire':>10} {'Min':>6} {'Max':>6}")
    print(header2)
    print("-" * len(header2))

    for i in range(len(BUCKET_BOUNDARIES) - 1):
        lo, hi = BUCKET_BOUNDARIES[i], BUCKET_BOUNDARIES[i + 1]
        chunk = activations[:, lo:hi].astype(np.float32)
        size = hi - lo

        # Variance of each feature across the 200 sentences (only non-dead features)
        feat_var = chunk.var(axis=0)
        nz_mask = (chunk != 0).any(axis=0)
        feat_var_nz = feat_var[nz_mask]
        mean_var = feat_var_nz.mean() if len(feat_var_nz) > 0 else 0.0
        med_var = np.median(feat_var_nz) if len(feat_var_nz) > 0 else 0.0

        # N nz: features nonzero in at least one sentence
        # N dead: features that are always zero across all sentences
        n_nz = int((chunk != 0).any(axis=0).sum())
        n_dead = size - n_nz
        pct_nz = 100.0 * n_nz / size

        # For each non-dead feature, how many sentences does it fire on?
        fires_per_feat = (chunk != 0).sum(axis=0)  # shape: (n_features,)
        fires_nz = fires_per_feat[nz_mask]
        mean_fires = fires_nz.mean() if len(fires_nz) > 0 else 0.0
        med_fires = np.median(fires_nz) if len(fires_nz) > 0 else 0.0
        min_fires = int(fires_nz.min()) if len(fires_nz) > 0 else 0
        max_fires = int(fires_nz.max()) if len(fires_nz) > 0 else 0

        print(f"{i:<8} {size:>6} {n_nz:>8} {n_dead:>8} {pct_nz:>7.1f}% "
              f"{mean_var:>10.4f} {med_var:>10.4f} "
              f"{mean_fires:>10.1f} {med_fires:>10.1f} {min_fires:>6} {max_fires:>6}")

    # --- Summary interpretation ---
    print(f"\n{'=' * 90}")
    print("Interpretation for regressions")
    print("=" * 90)
    print("- 'N nz': features nonzero for >= 1 sentence; 'N dead': always zero across all sentences")
    print("- 'Mean/Med var': per-feature variance across sentences, non-dead only (higher = more predictive capacity)")
    print(f"- 'Mean/Med fire': sentences (out of {n_sentences}) each non-dead feature fires on")
    print("- High-bucket features are very sparse — they add dimensions but little signal,")
    print("  increasing overfitting risk unless feature selection (e.g. Lasso) filters them out.")


FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures', 'exploratory')
BUCKET_LABELS = [f"Bucket {i}\n({BUCKET_BOUNDARIES[i]}–{BUCKET_BOUNDARIES[i+1]-1})"
                 for i in range(len(BUCKET_BOUNDARIES) - 1)]
BUCKET_COLORS = sns.color_palette("viridis", len(BUCKET_BOUNDARIES) - 1)


def compute_firing_counts(activations):
    """For each bucket, return firing counts (n sentences) for non-dead features."""
    result = {}
    for i in range(len(BUCKET_BOUNDARIES) - 1):
        lo, hi = BUCKET_BOUNDARIES[i], BUCKET_BOUNDARIES[i + 1]
        chunk = activations[:, lo:hi]
        fires_per_feat = (chunk != 0).sum(axis=0)
        nz_mask = fires_per_feat > 0
        result[i] = fires_per_feat[nz_mask]
    return result


def plot_firing_violin(firing_counts):
    """Violin plot of firing counts per bucket (non-dead features only)."""
    fig, ax = plt.subplots(figsize=(7, 5))

    data = []
    labels = []
    for i, counts in firing_counts.items():
        data.append(counts)
        labels.append(BUCKET_LABELS[i])

    parts = ax.violinplot(data, positions=range(len(data)), showmedians=True,
                          showextrema=False)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(BUCKET_COLORS[i])
        pc.set_alpha(0.7)
    parts['cmedians'].set_color('black')

    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Sentences feature fires on (out of 200)")
    ax.set_title("Firing count distribution per Matryoshka bucket\n(non-dead features only)")
    ax.set_yscale('symlog', linthresh=1)
    ax.set_yticks([1, 2, 5, 10, 20, 50, 100, 200])
    ax.set_yticklabels([1, 2, 5, 10, 20, 50, 100, 200])
    ax.set_ylim(bottom=0)
    sns.despine()

    fig.tight_layout()
    plt.show()
    plt.close(fig)


def plot_firing_cdf(firing_counts, n_sentences):
    """Survival plot: number of nz features that fire on >= N sentences, per bucket."""
    thresholds = np.arange(1, n_sentences + 1)

    # Precompute raw counts per bucket
    raw = {}
    for i, counts in firing_counts.items():
        raw[i] = np.array([(counts >= t).sum() for t in thresholds])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel configs: (title, xlim, xticks)
    panels = [
        ("All sentences", (1, n_sentences), [1, 25, 50, 75, 100, 125, 150, 175, 200]),
        ("Zoom: 1–20 sentences", (1, 20), list(range(1, 21))),
        ("Zoom: 50–200 sentences", (50, n_sentences), [50, 75, 100, 125, 150, 175, 200]),
    ]

    for ax, (title, xlim, xticks) in zip(axes, panels):
        for i, rc in raw.items():
            ax.plot(thresholds, rc, color=BUCKET_COLORS[i],
                    label=f"Bucket {i} ({len(firing_counts[i])} nz)", linewidth=2)
        ax.set_xlabel("Fires on >= N sentences")
        ax.set_ylabel("Number of features")
        ax.set_title(title)
        ax.set_xlim(xlim)
        ax.set_xticks(xticks)
        # Auto-scale y to the visible data range
        x_lo, x_hi = xlim
        y_max = max(rc[(x_lo - 1):x_hi].max() for rc in raw.values())
        ax.set_ylim(0, y_max * 1.05)
        sns.despine(ax=ax)

    axes[0].legend(fontsize=8)

    fig.suptitle("Number of non-dead features firing on >= N sentences", fontsize=13, y=1.02)
    fig.tight_layout()
    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    activations = pkl.load(open(FEATURES_PATH, 'rb'))
    check_buckets(activations)

    firing_counts = compute_firing_counts(activations)
    plot_firing_violin(firing_counts)
    plot_firing_cdf(firing_counts, n_sentences=activations.shape[0])
