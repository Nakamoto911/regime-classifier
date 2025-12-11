import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
from matplotlib.colors import ListedColormap
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# CONSTANTS
# ==========================================
NBER_RECESSIONS = [
    ('1960-04-01', '1961-02-01'),
    ('1969-12-01', '1970-11-01'),
    ('1973-11-01', '1975-03-01'),
    ('1980-01-01', '1980-07-01'),
    ('1981-07-01', '1982-11-01'),
    ('1990-07-01', '1991-03-01'),
    ('2001-03-01', '2001-11-01'),
    ('2007-12-01', '2009-06-01'),
    ('2020-02-01', '2020-04-01')
]

REGIME_COLORS = {
    0: 'red',
    1: 'green',
    2: 'blue',
    3: 'orange',
    4: 'purple',
    5: 'gold'
}

def set_style():
    """Sets the plotting style."""
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_context("talk")

# ==========================================
# UI COMPONENTS
# ==========================================

def render_header():
    st.title("Macroeconomic Regime Detection System")

def render_sidebar(default_start, default_end):
    st.sidebar.header("Configuration")
    start_date = st.sidebar.text_input("Start Date", value=default_start)
    end_date = st.sidebar.text_input("End Date", value=default_end)
    return start_date, end_date

def show_data_success(shape):
    st.sidebar.success(f"Data Loaded! Shape: {shape}")

def show_pca_info(n_components):
    st.sidebar.info(f"PCA: 95% variance explained by {n_components} components.")

def render_dashboard(df_transformed, final_labels, k_probs, gmm_labels_aligned, gmm_probs_aligned, 
                    X_pca, pca_model, cumsum, n_components, X_pre, local_appendix_path):
    
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Analysis", "Transitions", "Diagnostics"])

    with tab1:
        st.header("Regime Timeline & Probabilities")
        st.plotly_chart(plot_timeline_comparison(df_transformed.index, final_labels, gmm_labels_aligned, NBER_RECESSIONS, REGIME_COLORS), use_container_width=True)
        st.plotly_chart(plot_probabilities(df_transformed.index, k_probs, gmm_probs_aligned, NBER_RECESSIONS), use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Feature Heatmap")
            
            # Prepare Heatmap Data
            heatmap_vars = ['RPI', 'UNRATE', 'UMCSENTx', 'FEDFUNDS', 'CPIAUCSL', 'S&P 500']
            heatmap_data = pd.DataFrame(index=df_transformed.index)
            heatmap_data['Regime'] = final_labels
            
            # Flexible lookup for heatmap cols
            found_cols = []
            for key in heatmap_vars:
                matches = [c for c in df_transformed.columns if key in c]
                if matches:
                    heatmap_data[key] = df_transformed[matches[0]]
                    found_cols.append(key)
            
            if len(found_cols) > 0:
                regime_means = heatmap_data.groupby('Regime').mean().T
                scaler_mm = MinMaxScaler()
                regime_means_norm = pd.DataFrame(
                    scaler_mm.fit_transform(regime_means.T).T,
                    columns=regime_means.columns,
                    index=regime_means.index
                )
                st.pyplot(plot_feature_heatmap(regime_means_norm))
            else:
                st.warning("Could not find sufficient variables for Heatmap.")
                
        with col2:
            st.subheader("PCA Component Analysis")
            st.plotly_chart(plot_pca_scatter(X_pca, final_labels, REGIME_COLORS, dates=df_transformed.index), use_container_width=True)
        
        st.subheader("PCA Loadings")
        # 1. Get PCA loadings
        pca_loadings = pd.DataFrame(
            pca_model.components_.T, 
            index=X_pre.columns, 
            columns=[f'PC{i+1}' for i in range(pca_model.n_components_)]
        )

        # Select only the first 5 principal components
        pca_loadings_five_components = pca_loadings.iloc[:, :5]
        
        # Handle Appendix
        df_descriptions = None
        # Try local default
        try:
             df_descriptions = pd.read_csv(local_appendix_path, encoding='latin1')
        except FileNotFoundError:
            pass
                
        if df_descriptions is not None and 'fred' in df_descriptions.columns and 'description' in df_descriptions.columns:
            # Prepare df_descriptions for merging
            df_descriptions_clean = df_descriptions[['fred', 'description']].copy()
            df_descriptions_clean.rename(columns={'fred': 'variable_name'}, inplace=True)

            # Merge PCA loadings with descriptions
            pca_loadings_reset = pca_loadings_five_components.reset_index()
            pca_loadings_reset.rename(columns={'index': 'variable_name'}, inplace=True)

            combined_loadings = pd.merge(
                pca_loadings_reset,
                df_descriptions_clean,
                on='variable_name',
                how='left'
            )
            
            # Reorder columns
            combined_loadings = combined_loadings[['description', 'variable_name'] + [f'PC{i+1}' for i in range(5)]]
            st.dataframe(combined_loadings)
        else:
            st.dataframe(pca_loadings_five_components)
            if df_descriptions is None:
                st.info(f"Ensure '{local_appendix_path}' exists locally to see variable descriptions.")

    with tab3:
        st.header("Transition Dynamics")
        
        # We need get_transition_matrix here or pass it?
        # Typically it's a data transformation. I can move the function here or keep it in app.py and pass the result.
        # The prompt said "app.py contains mostly the algorithms". get_transition_matrix is a simple algo.
        # I'll calculate it on the fly here or import it. Or better, move the helper function here since it's used for plotting.
        # Actually, let's just define the helper in plots or keep it in app and pass the matrix.
        # To keep app.py clean, passing the matrix is better if it was pre-calc, but it's calculated only for the plot.
        # I'll add the helper function to plots.py
        
        trans_matrix_raw = get_transition_matrix(final_labels)
        
        mat_cond = trans_matrix_raw.copy()
        np.fill_diagonal(mat_cond, 0)
        row_sums_cond = mat_cond.sum(axis=1)[:, np.newaxis]
        trans_matrix_cond = np.divide(mat_cond, row_sums_cond, where=row_sums_cond!=0)

        st.pyplot(plot_transition_matrices(trans_matrix_raw, trans_matrix_cond))
        
        st.subheader("Network Graph")
        st.pyplot(plot_network_graph(trans_matrix_cond, REGIME_COLORS))

    with tab4:
        st.header("Model Diagnostics")
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(plot_pca_variance(cumsum, n_components))
        with col2:
            st.pyplot(plot_scree(pca_model))

def get_transition_matrix(seq, normalize_rows=True):
    n = 6
    mat = np.zeros((n, n))
    for (i, j) in zip(seq, seq[1:]):
        mat[i, j] += 1
    if normalize_rows:
        row_sums = mat.sum(axis=1)[:, np.newaxis]
        mat = np.divide(mat, row_sums, where=row_sums!=0)
    return mat

# ==========================================
# PLOTTING FUNCTIONS
# ==========================================

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


def plot_timeline_comparison(dates, final_labels, gmm_labels_aligned, recession_dates, regime_colors):
    """Figure 2: K-Means vs GMM Timeline (Plotly)."""
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        subplot_titles=("K-Means Regimes", "GMM Regimes"),
        vertical_spacing=0.1
    )

    # Helper to add recession shapes
    rect_shapes = []
    for start, end in recession_dates:
        rect_shapes.append(dict(
            type="rect",
            xref="x", yref="paper",
            x0=start, x1=end,
            y0=0, y1=1,
            fillcolor="gray",
            opacity=0.3,
            layer="below",
            line_width=0,
            line_color="rgba(0,0,0,0)",
        ))

    # --- Top Plot: K-Means ---
    # We assign colors per point. Plotly Scatter with mode='markers' allows list of colors.
    # Convert labels to specific colors
    k_colors = [regime_colors.get(l, 'gray') for l in final_labels]
    
    fig.add_trace(
        go.Scatter(
            x=dates, 
            y=final_labels,
            mode='markers',
            marker=dict(color=k_colors, size=6),
            name='K-Means',
            hovertemplate='Date: %{x}<br>Regime: %{y}<extra></extra>'
        ),
        row=1, col=1
    )

    # --- Bottom Plot: GMM ---
    g_colors = [regime_colors.get(l, 'gray') for l in gmm_labels_aligned]
    
    fig.add_trace(
        go.Scatter(
            x=dates, 
            y=gmm_labels_aligned,
            mode='markers',
            marker=dict(color=g_colors, size=6),
            name='GMM',
            hovertemplate='Date: %{x}<br>Regime: %{y}<extra></extra>'
        ),
        row=2, col=1
    )

    # Layout Updates
    fig.update_layout(
        title_text="Figure 2: K-Means (Top) vs GMM (Bottom)",
        height=600,
        shapes=rect_shapes,
        showlegend=False
    )
    
    fig.update_yaxes(title_text="Regime", row=1, col=1, tickvals=list(range(6)))
    fig.update_yaxes(title_text="Regime", row=2, col=1, tickvals=list(range(6)))
    fig.update_xaxes(title_text="Date", row=2, col=1)

    return fig


def plot_probabilities(dates, k_probs, g_probs, recession_dates):
    """Figure 3: Crisis Probability (Plotly)."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=("K-Means Probability (Crisis vs Normal)", "GMM Probability (Crisis vs Normal)"),
        vertical_spacing=0.1
    )

    # Recession Shapes
    rect_shapes = []
    for start, end in recession_dates:
        rect_shapes.append(dict(
            type="rect",
            xref="x", yref="paper",
            x0=start, x1=end,
            y0=0, y1=1,
            fillcolor="gray",
            opacity=0.3,
            layer="below",
            line_width=0,
            line_color="rgba(0,0,0,0)",
        ))

    # --- K-Means Traces ---
    k_crisis_prob = k_probs[:, 0]
    k_normal_prob = np.sum(k_probs[:, 1:], axis=1)

    fig.add_trace(
        go.Scatter(x=dates, y=k_crisis_prob, mode='lines', line=dict(color='red', width=2), name='K-Means Crisis'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=dates, y=k_normal_prob, mode='lines', line=dict(color='blue', width=1), name='K-Means Normal'),
        row=1, col=1
    )

    # --- GMM Traces ---
    g_crisis_prob = g_probs[:, 0]
    g_normal_prob = np.sum(g_probs[:, 1:], axis=1)

    fig.add_trace(
        go.Scatter(x=dates, y=g_crisis_prob, mode='lines', line=dict(color='red', width=2), name='GMM Crisis'),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=dates, y=g_normal_prob, mode='lines', line=dict(color='green', width=1), name='GMM Normal'),
        row=2, col=1
    )

    fig.update_layout(
        title_text="Figure 3: Crisis Probability Overview",
        height=600,
        shapes=rect_shapes,
        showlegend=True
    )
    
    fig.update_yaxes(title_text="Probability", range=[-0.05, 1.05], row=1, col=1)
    fig.update_yaxes(title_text="Probability", range=[-0.05, 1.05], row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)

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


def plot_network_graph(trans_matrix_cond, regime_colors):
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
    
    # Node colors
    node_colors = [regime_colors.get(i, 'lightblue') for i in range(6)]

    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color=node_colors, ax=ax)
    nx.draw_networkx_labels(G, pos, labels=labels_map, font_size=10, ax=ax)
    nx.draw_networkx_edges(G, pos, width=weights, arrowstyle='->', arrowsize=20, edge_color='gray', ax=ax)

    ax.set_title('Figure 6: Regime Transition Network Graph')
    ax.axis('off')
    return fig


def plot_pca_scatter(X_pca, labels, regime_colors, dates=None):
    """Scatter Plot PC1 vs PC2 (Plotly)."""

    fig = go.Figure()

    unique_labels = sorted(np.unique(labels))
    
    for label_value in unique_labels:
        mask = labels == label_value
        x_pts = X_pca[mask, 0]
        y_pts = X_pca[mask, 1]
        
        # If dates are provided, slice them
        hover_text = []
        if dates is not None:
             # Ensure dates is aligned with X_pca
             # X_pca usually matches df_transformed, same as dates
             dates_masked = dates[mask]
             hover_text = [f"Date: {d.strftime('%Y-%m-%d')}<br>Regime: {label_value}" for d in dates_masked]
        else:
             hover_text = [f"Regime: {label_value}" for _ in range(len(x_pts))]

        color = regime_colors.get(label_value, 'black')
        
        fig.add_trace(go.Scatter(
            x=x_pts,
            y=y_pts,
            mode='markers',
            name=f'Regime {label_value}',
            marker=dict(color=color, size=8, opacity=0.8),
            text=hover_text,
            hoverinfo='text'
        ))

    fig.update_layout(
        title="Scatter Plot of PC1 vs PC2",
        xaxis_title="Principal Component 1",
        yaxis_title="Principal Component 2",
        height=600,
        showlegend=True
    )

    return fig
