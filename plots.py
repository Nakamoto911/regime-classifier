import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
import pandas as pd
import networkx as nx
from matplotlib.colors import ListedColormap


def set_style():
    """Sets the plotting style."""
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_context("talk")


def plot_pca_variance(cumsum, n_components):
    """Plots cumulative explained variance."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(cumsum, label='Cumulative Explained Variance', lw=2)
    ax.axhline(0.95, color='r', linestyle='--', label='95% Threshold')
    ax.axvline(n_components, color='gray', linestyle=':', label=f'{n_components} Components')
    ax.set_title('Figure 1: PCA Cumulative Explained Variance')
    ax.set_xlabel('# of Components')
    ax.set_ylabel('Explained Variance')
    ax.legend()
    plt.tight_layout()
    return fig


def plot_scree(pca):
    """Plots the scree plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    n_features = 30
    ax.bar(range(1, n_features + 1), pca.explained_variance_ratio_[:n_features])
    ax.set_title(f'Scree Plot: Explained Variance per First {n_features} Principal Components')
    ax.set_xlabel('Principal Component Number')
    ax.set_ylabel('Explained Variance Ratio')
    ax.set_xticks(range(1, n_features + 1))
    ax.tick_params(axis='x', rotation=90)
    ax.grid(True)
    plt.tight_layout()
    return fig


def plot_recessions(ax, recession_dates):
    """Helper to plot NBER recessions."""
    for start, end in recession_dates:
        try:
            start_dt = pd.to_datetime(start)
            end_dt = pd.to_datetime(end)
            ax.axvspan(start_dt, end_dt, color='gray', alpha=0.3)
        except Exception:
            pass # Ignore validity errors for plotting if date ranges are weird


def plot_timeline_comparison(dates, final_labels, gmm_labels_aligned, recession_dates):
    """Figure 2: K-Means vs GMM Timeline."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    plot_recessions(ax1, recession_dates)
    ax1.scatter(dates, final_labels, s=10, c='blue', alpha=0.6)
    ax1.set_title('Figure 2: K-Means Regimes (Top) vs GMM (Bottom)')
    ax1.set_ylabel('Regime (K-Means)')
    ax1.set_yticks(range(6))
    ax1.grid(True)

    plot_recessions(ax2, recession_dates)
    ax2.scatter(dates, gmm_labels_aligned, s=10, c='green', alpha=0.6)
    ax2.set_ylabel('Regime (GMM)')
    ax2.set_yticks(range(6))
    ax2.set_xlabel('Date')
    ax2.grid(True)

    plt.tight_layout()
    return fig


def plot_probabilities(dates, k_probs, g_probs, recession_dates):
    """Figure 3: Crisis Probability."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # K-Means Probs
    k_crisis_prob = k_probs[:, 0]
    k_normal_prob = np.sum(k_probs[:, 1:], axis=1)

    plot_recessions(ax1, recession_dates)
    ax1.plot(dates, k_crisis_prob, label='P(Crisis)', color='red', linewidth=1)
    ax1.plot(dates, k_normal_prob, label='P(Normal)', color='blue', linewidth=0.5, alpha=0.5)
    ax1.set_title('Figure 3: Crisis Probability (K-Means vs GMM)')
    ax1.set_ylabel('Prob (K-Means)')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # GMM Probs
    g_crisis_prob = g_probs[:, 0]
    g_normal_prob = np.sum(g_probs[:, 1:], axis=1)

    plot_recessions(ax2, recession_dates)
    ax2.plot(dates, g_crisis_prob, label='P(Crisis)', color='red', linewidth=1)
    ax2.plot(dates, g_normal_prob, label='P(Normal)', color='green', linewidth=0.5, alpha=0.5)
    ax2.set_ylabel('Prob (GMM)')
    ax2.set_xlabel('Date')
    ax2.grid(True)

    plt.tight_layout()
    return fig


def plot_feature_heatmap(regime_means_norm):
    """Figure 4: Feature Heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(regime_means_norm, annot=True, cmap='Blues', fmt=".2f", ax=ax)
    ax.set_title('Figure 4: Macroeconomic Feature Heatmap (Normalized)')
    ax.set_xlabel('Regime')
    return fig


def plot_transition_matrices(trans_matrix_raw, trans_matrix_cond):
    """Figure 5: Transition Matrices."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.heatmap(trans_matrix_raw, annot=True, fmt=".2f", cmap='Blues', ax=ax1)
    ax1.set_title('Matrix A: Raw Transition Probs')
    ax1.set_ylabel('From Regime')
    ax1.set_xlabel('To Regime')

    sns.heatmap(trans_matrix_cond, annot=True, fmt=".2f", cmap='Reds', ax=ax2)
    ax2.set_title('Matrix B: Conditional (Exit) Probs')
    ax2.set_xlabel('To Regime')
    
    plt.tight_layout()
    return fig


def plot_network_graph(trans_matrix_cond):
    """Figure 6: Network Graph."""
    G = nx.DiGraph()
    labels_map = {
        0: "0: Crisis",
        1: "1: Recovery",
        2: "2: Growth",
        3: "3: Stagflation",
        4: "4: Pre-Recession",
        5: "5: Boom"
    }

    for i in range(6):
        G.add_node(i, label=labels_map[i])

    # Add edges
    for i in range(6):
        for j in range(6):
            weight = trans_matrix_cond[i, j]
            if weight > 0.15: # Threshold to reduce clutter
                G.add_edge(i, j, weight=weight)

    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42, k=2) # k regulates spacing
    weights = [G[u][v]['weight'] * 5 for u,v in G.edges()]

    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue', ax=ax)
    nx.draw_networkx_labels(G, pos, labels=labels_map, font_size=10, ax=ax)
    nx.draw_networkx_edges(G, pos, width=weights, arrowstyle='->', arrowsize=20, edge_color='gray', ax=ax)

    ax.set_title('Figure 6: Regime Transition Network Graph')
    ax.axis('off')
    return fig


def plot_pca_scatter(X_pca, labels):
    """Scatter Plot PC1 vs PC2."""
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown']
    unique_labels = np.unique(labels)
    # Ensure we don't go out of bounds if labels > 6 (though we expect 6)
    cmap_custom = ListedColormap(colors[:len(unique_labels)])

    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap=cmap_custom, s=20, alpha=0.8)

    legend_handles = []
    # Create manual legend handles to ensure correct color mapping
    for i, label_value in enumerate(sorted(unique_labels)):
        color_idx = i if i < len(colors) else -1
        color = colors[color_idx] if color_idx != -1 else 'black'
        legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', label=f'Regime {label_value}',
                                         markerfacecolor=color, markersize=10))

    ax.legend(handles=legend_handles, title='Macro Regimes', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_title('Scatter Plot of PC1 vs PC2, Colored by Macroeconomic Regime')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.grid(True)
    plt.tight_layout()
    return fig
