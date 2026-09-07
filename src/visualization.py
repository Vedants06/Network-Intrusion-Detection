# src/visualization.py
"""
Visualization utilities for the NIDS project.
All plots follow a consistent style.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ─── Global Style ───────────────────────────────────────────────
def set_plot_style():
    """Set consistent plot style for all visualizations."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'figure.dpi': 100,
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
    })

set_plot_style()

# Color palettes
BINARY_COLORS = ['#2ecc71', '#e74c3c']  # Green=Normal, Red=Attack
MULTI_COLORS = ['#2ecc71', '#e74c3c', '#f39c12', '#3498db', '#9b59b6']


def save_plot(fig, filepath):
    """Save figure to disk and close it."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"📊 Plot saved: {filepath}")


def plot_class_distribution(y, title, save_path=None, multi=False):
    """Plot bar chart of class distribution."""
    fig, ax = plt.subplots(figsize=(8, 5))

    unique, counts = np.unique(y, return_counts=True)
    colors = MULTI_COLORS if multi else BINARY_COLORS

    bars = ax.bar(range(len(unique)), counts, color=colors[:len(unique)],
                  edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.set_xticks(range(len(unique)))
    ax.set_xticklabels([str(u) for u in unique])

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{count:,}\n({count/len(y)*100:.1f}%)',
                ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    if save_path:
        save_plot(fig, save_path)
    else:
        plt.show()


def plot_pie_chart(y, labels_map, title, save_path=None):
    """Plot pie chart of class distribution."""
    fig, ax = plt.subplots(figsize=(8, 8))

    unique, counts = np.unique(y, return_counts=True)
    label_names = [labels_map.get(u, str(u)) for u in unique]
    colors = MULTI_COLORS[:len(unique)] if len(unique) > 2 else BINARY_COLORS

    wedges, texts, autotexts = ax.pie(
        counts, labels=label_names, autopct='%1.1f%%',
        colors=colors, startangle=90, pctdistance=0.85,
        wedgeprops=dict(edgecolor='white', linewidth=2)
    )

    for autotext in autotexts:
        autotext.set_fontweight('bold')

    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        save_plot(fig, save_path)
    else:
        plt.show()


def plot_correlation_heatmap(corr_matrix, title="Feature Correlation",
                              save_path=None):
    """Plot correlation heatmap."""
    fig, ax = plt.subplots(figsize=(20, 16))

    sns.heatmap(corr_matrix, annot=False, cmap='RdBu_r',
                center=0, vmin=-1, vmax=1,
                xticklabels=True, yticklabels=True,
                linewidths=0.1, ax=ax)

    ax.set_title(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    if save_path:
        save_plot(fig, save_path)
    else:
        plt.show()


def plot_feature_distributions(X, feature_names, y=None, top_n=10,
                                save_path=None):
    """Plot histograms of top N features."""
    n_cols = 5
    n_rows = (top_n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.flatten()

    for i in range(top_n):
        if i < len(feature_names):
            ax = axes[i]
            if y is not None:
                for cls in np.unique(y):
                    mask = y == cls
                    ax.hist(X[mask, i], bins=50, alpha=0.5,
                            label=f'Class {cls}', density=True)
                ax.legend(fontsize=8)
            else:
                ax.hist(X[:, i], bins=50, color='steelblue',
                        edgecolor='white', alpha=0.7)
            ax.set_title(feature_names[i], fontsize=10)
            ax.tick_params(labelsize=8)

    # Hide unused subplots
    for i in range(top_n, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('Feature Distributions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    if save_path:
        save_plot(fig, save_path)
    else:
        plt.show()


def plot_boxplots(X, feature_names, y, top_n=10, save_path=None):
    """Plot box plots of features grouped by class."""
    n_cols = 5
    n_rows = (top_n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.flatten()

    df = pd.DataFrame(X[:, :top_n], columns=feature_names[:top_n])
    df['class'] = y

    for i, feat in enumerate(feature_names[:top_n]):
        ax = axes[i]
        df.boxplot(column=feat, by='class', ax=ax)
        ax.set_title(feat, fontsize=10)
        ax.set_xlabel('')
        ax.tick_params(labelsize=8)
        plt.sca(ax)
        plt.xticks(rotation=0)

    for i in range(top_n, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('Feature Box Plots by Class', fontsize=16, fontweight='bold')
    plt.tight_layout()
    if save_path:
        save_plot(fig, save_path)
    else:
        plt.show()
