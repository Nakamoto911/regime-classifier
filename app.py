import streamlit as st
import pandas as pd
import numpy as np
import plots
from sklearn.preprocessing import RobustScaler, normalize
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
LOCAL_APPENDIX_PATH = 'FRED-MD_updated_appendix.csv'

plots.set_style()

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

    # Apply t_codes Transformations
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
    
    # Scaling data using a method robust to outliers (based on Median and IQR)
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
    # Pass both models to combine distributions strictly as per the paper
    kmeans_probs = calculate_fuzzy_probs(X_pca, kmeans_step1, kmeans_step2, crisis_label)

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

def calculate_fuzzy_probs(X, kmeans_step1, kmeans_step2, crisis_label):
    """
    Calculates fuzzy probabilities by combining distributions from two separate 
    clustering runs (L2 and Cosine) as described in Section 3.2.
    """
    # --- Step 1: L2 Distributions (Regime 0 vs Rest) ---
    # Use the L2 model (Step 1) on the original X
    dists_1 = pairwise_distances(X, kmeans_step1.cluster_centers_, metric='euclidean')
    
    # Apply Eq (1) for the 2 clusters from Step 1
    denom_1 = np.sum(dists_1, axis=1, keepdims=True)
    denom_1[denom_1 == 0] = 1e-9 # Avoid division by zero
    
    terms_1 = 1 - (dists_1 / denom_1)
    sum_terms_1 = np.sum(terms_1, axis=1, keepdims=True)
    probs_1 = terms_1 / sum_terms_1
    
    # Extract P(REGIME 0) - The probability of being in the Crisis cluster
    P_regime0_l2 = probs_1[:, crisis_label]
    
    # --- Step 2: Cosine Distributions (Regimes 1-5) ---
    # Use the Cosine model (Step 2) on Normalized X
    # Note: K-Means on normalized data w/ Euclidean distance is equivalent to Cosine clustering
    X_norm = normalize(X)
    dists_2 = pairwise_distances(X_norm, kmeans_step2.cluster_centers_, metric='euclidean')
    
    # Apply Eq (1) for the 5 clusters from Step 2
    denom_2 = np.sum(dists_2, axis=1, keepdims=True)
    denom_2[denom_2 == 0] = 1e-9
    
    terms_2 = 1 - (dists_2 / denom_2)
    sum_terms_2 = np.sum(terms_2, axis=1, keepdims=True)
    probs_2 = terms_2 / sum_terms_2
    
    # --- Step 3: Combine & Scale (Eq 2, 3, 4) ---
    # P_max from the "Normal" regimes (Eq 2)
    P_max_others = np.max(probs_2, axis=1)
    
    # Calculate scaled P_R0 (Eq 3 & 4)
    # P_R0 = -P_max * log2(1 - P(Regime 0))
    val = 1 - P_regime0_l2
    val = np.clip(val, 1e-9, 1.0) # Numerical stability
    
    P_R0_scaled = -P_max_others * np.log2(val)
    P_R0_scaled = np.clip(P_R0_scaled, 0, None) # Ensure non-negative
    
    # Construct final unnormalized vector
    # Col 0 = Scaled Crisis Prob, Col 1-5 = Cosine Probs
    final_probs_unnorm = np.zeros((X.shape[0], 6))
    final_probs_unnorm[:, 0] = P_R0_scaled
    final_probs_unnorm[:, 1:] = probs_2
    
    # Renormalize to sum to 1
    row_sums = final_probs_unnorm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    final_probs = final_probs_unnorm / row_sums
    
    return final_probs

# ==========================================
# MAIN APP
# ==========================================

# 1. UI: Header
plots.render_header()

# 2. UI: Sidebar & Config
start_date, end_date = plots.render_sidebar(DEFAULT_START_DATE, DEFAULT_END_DATE)

# 3. Data Loading
df_transformed, err = load_and_preprocess_data(None, default_path=LOCAL_FRED_MD_PATH, start_date=start_date, end_date=end_date)

if err:
    st.error(f"Failed to load data: {err}")
    st.info(f"Please ensure '{LOCAL_FRED_MD_PATH}' exists.")
    st.stop()

plots.show_data_success(df_transformed.shape)

# 4. Processing
X_scaled, X_pre = prepare_pca_data(df_transformed)
pca_model, X_pca, cumsum, n_components = run_pca(X_scaled)

plots.show_pca_info(n_components)

final_labels, k_probs, gmm_labels_aligned, gmm_probs_aligned = run_clustering_models(X_pca)

# 5. UI: Dashboard
plots.render_dashboard(
    df_transformed, 
    final_labels, 
    k_probs, 
    gmm_labels_aligned, 
    gmm_probs_aligned, 
    X_pca, 
    pca_model, 
    cumsum, 
    n_components, 
    X_pre, 
    LOCAL_APPENDIX_PATH
)
