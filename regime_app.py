import streamlit as st
import pandas as pd
import numpy as np
import regime_plots
from sklearn.preprocessing import RobustScaler, MinMaxScaler, normalize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances
import io

# ==========================================
# 1. CONFIGURATION & UTILS
# ==========================================
st.set_page_config(page_title="Macro Regime Detection", layout="wide")

# Constants
DEFAULT_START_DATE = '1959-12-01'
DEFAULT_END_DATE = '2023-01-01'
FRED_MD_URL = "https://files.stlouisfed.org/files/htdocs/fred-md/monthly/current.csv"
LOCAL_FRED_MD_PATH = '2025-11-MD.csv' # Keep reference, but allow override

# NBER Recession Dates (Hardcoded for stability)
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

regime_plots.set_style()

def apply_t_code(series, code):
    """Applies FRED-MD transformation codes to achieve stationarity."""
    x = series.to_numpy()
    if code == 1: # No transformation
        return pd.Series(x, index=series.index)
    elif code == 2: # First difference
        return pd.Series(np.diff(x, prepend=np.nan), index=series.index)
    elif code == 3: # Second difference
        return pd.Series(np.diff(np.diff(x, prepend=np.nan), prepend=np.nan), index=series.index)
    elif code == 4: # Log
        return pd.Series(np.log(x), index=series.index)
    elif code == 5: # First difference of log
        return pd.Series(np.diff(np.log(x), prepend=np.nan), index=series.index)
    elif code == 6: # Second difference of log
        return pd.Series(np.diff(np.diff(np.log(x), prepend=np.nan), prepend=np.nan), index=series.index)
    elif code == 7: # Delta (x_t / x_{t-1} - 1)
        return series.pct_change()
    else:
        return series

@st.cache_data
def load_and_preprocess_data(uploaded_file=None, default_path=None, start_date=DEFAULT_START_DATE, end_date=DEFAULT_END_DATE):
    """Loads, transforms and cleans the FRED-MD dataset."""
    
    if uploaded_file is not None:
        try:
            df_raw = pd.read_csv(uploaded_file)
        except Exception as e:
            return None, f"Error reading uploaded file: {e}"
    elif default_path:
        try:
            df_raw = pd.read_csv(default_path)
        except FileNotFoundError:
             return None, f"Default file not found at {default_path}. Please upload a CSV."
    else:
        return None, "No file provided."

    if 'sasdate' not in df_raw.columns:
        return None, "Column 'sasdate' not found in data."

    # Extract Transformation Codes (Row 2) - Index 0 in 0-based
    t_codes = df_raw.iloc[0].to_dict()
    df_data = df_raw.iloc[1:].copy()

    # Date Parsing
    df_data['sasdate'] = pd.to_datetime(df_data['sasdate'])
    df_data.set_index('sasdate', inplace=True)

    # Filter Dates
    df_data = df_data.loc[str(start_date):str(end_date)]

    # Apply Transformations
    transformed_series = []
    for col in df_data.columns:
        if col in t_codes:
            try:
                code = int(t_codes[col])
                transformed_series.append(apply_t_code(df_data[col].astype(float), code).rename(col))
            except ValueError:
                pass # Skip non-numeric or malformed columns
                
    df_transformed = pd.concat(transformed_series, axis=1)

    # Handle NaNs created by differencing
    df_transformed.ffill(inplace=True)
    df_transformed.bfill(inplace=True)
    df_transformed.dropna(axis=1, inplace=True)

    return df_transformed, None

def prepare_pca_data(df_transformed):
    """Prepares data for PCA by excluding specific keywords."""
    exclude_keywords = [
        'FEDFUNDS', 'TB3MS', 'TB6MS', 'GS1', 'GS5', 'GS10', 'AAA', 'BAA',
        'COMPAP', 'MORTG', 'TB3SMFFM', 'TB6SMFFM', 'T1YFFM', 'T5YFFM', 'T10YFFM', 'AAAFFM', 'BAAFFM',
        'EX', 'S&P','VIX', 'VIXCLSx','DIVIDEND','PE RATIO'
    ]
    pca_cols = [c for c in df_transformed.columns if not any(x in c for x in exclude_keywords)]
    X_pre = df_transformed[pca_cols].copy()
    
    # Standardization (Demean + Unit Variance)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_pre)
    
    # Winsorization
    X_scaled = np.clip(X_scaled, -10, 10)
    
    return X_scaled, X_pre

@st.cache_resource
def run_pca(X_scaled):
    """Runs PCA and returns model and transformed data."""
    pca = PCA()
    pca.fit(X_scaled)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    n_components_95 = np.argmax(cumsum >= 0.95) + 1
    X_pca = pca.transform(X_scaled)[:, :n_components_95]
    return pca, X_pca, cumsum, n_components_95

@st.cache_data
def run_clustering_models(X_pca, n_init=10):
    """
    Runs Modified K-Means (Step 1 & Step 2) and GMM.
    Returns:
        final_labels: Combined K-Means labels (0=Crisis, 1-5=Normal)
        kmeans_probs: Fuzzy probabilities for K-Means regimes
        gmm_labels_aligned: GMM labels aligned to K-Means
        gmm_probs_aligned: GMM probabilities aligned
    """
    # --- 4.1 Modified K-Means ---
    # Step 1: Crisis Detection (L2, k=2)
    kmeans_step1 = KMeans(n_clusters=2, random_state=42, n_init=n_init)
    labels_step1 = kmeans_step1.fit_predict(X_pca)

    # Identify Crisis Cluster (Smaller cluster)
    counts = np.bincount(labels_step1)
    crisis_label = np.argmin(counts)
    normal_label = np.argmax(counts)

    # Step 2: Normal Market Classification (Cosine, k=5)
    normal_indices = np.where(labels_step1 == normal_label)[0]
    X_normal = X_pca[normal_indices]

    # Normalize for Cosine Similarity equivalent in KMeans
    X_normal_norm = normalize(X_normal)
    kmeans_step2 = KMeans(n_clusters=5, random_state=42, n_init=n_init)
    labels_normal_sub = kmeans_step2.fit_predict(X_normal_norm)

    # Merge Labels
    final_labels = np.zeros(len(X_pca), dtype=int)
    final_labels[labels_step1 == crisis_label] = 0
    final_labels[normal_indices] = labels_normal_sub + 1

    # Probabilistic Fuzzy Logic
    kmeans_probs = calculate_fuzzy_probs(X_pca, kmeans_step1.cluster_centers_, crisis_label, final_labels)

    # --- 4.2 Benchmark GMM ---
    gmm = GaussianMixture(n_components=6, random_state=42)
    gmm.fit(X_pca)
    gmm_labels = gmm.predict(X_pca)
    gmm_probs = gmm.predict_proba(X_pca)

    # Align GMM labels
    mapping = {}
    for g_label in range(6):
        mask = (gmm_labels == g_label)
        k_modes = final_labels[mask]
        if len(k_modes) == 0: continue
        mode = np.bincount(k_modes).argmax()
        mapping[g_label] = mode

    gmm_labels_aligned = np.array([mapping.get(x, x) for x in gmm_labels])
    
    # Re-sort probs based on mapping for consistency
    gmm_probs_aligned = np.zeros_like(gmm_probs)
    for g, k in mapping.items():
        gmm_probs_aligned[:, k] += gmm_probs[:, g]

    return final_labels, kmeans_probs, gmm_labels_aligned, gmm_probs_aligned

def calculate_fuzzy_probs(X, centroids_step1, crisis_label, final_labels):
    """Calculates fuzzy probabilities based on distance to conceptual centroids."""
    # Reconstruct centroids in original PCA space
    # Crisis Centroid from Step 1
    c0 = centroids_step1[crisis_label]
    
    # Normal Centroids (approximated from Step 2 results in original space)
    c_others = []
    for i in range(5):
        idx = np.where(final_labels == (i+1))[0]
        if len(idx) > 0:
            c_others.append(X[idx].mean(axis=0))
        else:
            c_others.append(np.zeros(X.shape[1]))

    all_centroids = np.vstack([c0, c_others]) # Shape (6, n_features)

    # Calculate Distances
    dists = pairwise_distances(X, all_centroids, metric='euclidean')

    # Apply Formula: P(Ci) = 1 - (di / sum(dj)) (simplified for viz)
    # Note: Using formula from script
    denom_row = np.sum(dists, axis=1) # Sum of distances for each point
    
    # Avoid zero division
    denom_row[denom_row == 0] = 1e-9

    raw_probs = np.zeros_like(dists)
    for i in range(6):
        term = 1 - (dists[:, i] / denom_row)
        raw_probs[:, i] = term

    # Normalize by sum over m
    row_sums = raw_probs.sum(axis=1)[:, np.newaxis]
    P_base = raw_probs / row_sums

    # Scaling Logic for P_R0
    P_max_others = np.max(P_base[:, 1:], axis=1)
    val = 1 - P_base[:, 0]
    val = np.clip(val, 1e-9, 1.0)

    P_R0_scaled = -P_max_others * np.log2(val)
    P_R0_scaled = np.clip(P_R0_scaled, 0, 1) 

    # Re-normalize vector
    final_probs = P_base.copy()
    final_probs[:, 0] = P_R0_scaled
    
    remainder = 1 - final_probs[:, 0]
    sum_others = final_probs[:, 1:].sum(axis=1)
    sum_others[sum_others == 0] = 1 # Avoid div by zero

    for i in range(1, 6):
        final_probs[:, i] = remainder * (final_probs[:, i] / sum_others)

    return final_probs

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
# MAIN APP
# ==========================================

st.title("Macroeconomic Regime Detection System")

# SIDEBAR
st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload FRED-MD CSV", type=["csv"])
start_date = st.sidebar.text_input("Start Date", value=DEFAULT_START_DATE)
end_date = st.sidebar.text_input("End Date", value=DEFAULT_END_DATE)

# Load Data
df_transformed, err = load_and_preprocess_data(uploaded_file, default_path=LOCAL_FRED_MD_PATH, start_date=start_date, end_date=end_date)

if err:
    st.error(f"Failed to load data: {err}")
    st.info("Plese upload the FRED-MD 'current.csv' file.")
    st.stop()

st.sidebar.success(f"Data Loaded! Shape: {df_transformed.shape}")

# Preprocess & PCA
X_scaled, X_pre = prepare_pca_data(df_transformed)
pca_model, X_pca, cumsum, n_components = run_pca(X_scaled)

st.sidebar.info(f"PCA: 95% variance explained by {n_components} components.")

# Run Models
final_labels, k_probs, gmm_labels_aligned, gmm_probs_aligned = run_clustering_models(X_pca)

# TABS
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Analysis", "Transitions", "Diagnostics"])

with tab1:
    st.header("Regime Timeline & Probabilities")
    st.pyplot(regime_plots.plot_timeline_comparison(df_transformed.index, final_labels, gmm_labels_aligned, NBER_RECESSIONS))
    st.pyplot(regime_plots.plot_probabilities(df_transformed.index, k_probs, gmm_probs_aligned, NBER_RECESSIONS))

with tab2:
    st.header("Feature Analysis")
    
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
        st.pyplot(regime_plots.plot_feature_heatmap(regime_means_norm))
    else:
        st.warning("Could not find sufficient variables for Heatmap.")
        
    st.header("PCA Component Analysis")
    st.pyplot(regime_plots.plot_pca_scatter(X_pca, final_labels))

with tab3:
    st.header("Transition Dynamics")
    
    trans_matrix_raw = get_transition_matrix(final_labels)
    
    mat_cond = trans_matrix_raw.copy()
    np.fill_diagonal(mat_cond, 0)
    row_sums_cond = mat_cond.sum(axis=1)[:, np.newaxis]
    trans_matrix_cond = np.divide(mat_cond, row_sums_cond, where=row_sums_cond!=0)

    st.pyplot(regime_plots.plot_transition_matrices(trans_matrix_raw, trans_matrix_cond))
    
    st.subheader("Network Graph")
    st.pyplot(regime_plots.plot_network_graph(trans_matrix_cond))

with tab4:
    st.header("Model Diagnostics")
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(regime_plots.plot_pca_variance(cumsum, n_components))
    with col2:
        st.pyplot(regime_plots.plot_scree(pca_model))

