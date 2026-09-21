import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import load_model
from src.evaluation_metrics import compute_all_metrics

BASE = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="NIDS | Network Intrusion Detection",
    page_icon=os.path.join(BASE, "assets", "favicon.ico") if os.path.exists(os.path.join(BASE, "assets", "favicon.ico")) else "🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Theme
# ----------------------------------------------------------------------------
ACCENT = "#3B82F6"
DANGER = "#EF4444"
OK = "#22C55E"
WARN = "#F59E0B"
MUTED = "#8B93A7"
PALETTE = [ACCENT, DANGER, WARN, "#A855F7", OK]

st.markdown(f"""
<style>
    .main .block-container {{ padding-top: 2rem; max-width: 1200px; }}
    h1, h2, h3 {{ font-weight: 650; letter-spacing: -0.01em; }}
    [data-testid="stSidebar"] {{ border-right: 1px solid rgba(128,128,128,0.15); }}
    [data-testid="stMetric"] {{
        background: rgba(128,128,128,0.06);
        border: 1px solid rgba(128,128,128,0.12);
        border-radius: 10px;
        padding: 14px 16px 10px 16px;
    }}
    [data-testid="stMetricValue"] {{ font-size: 1.6rem; }}
    div.stButton > button[kind="primary"] {{
        background: {ACCENT}; border: none; font-weight: 600;
    }}
    .nid-card {{
        background: rgba(128,128,128,0.06);
        border: 1px solid rgba(128,128,128,0.12);
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 14px;
    }}
    .nid-card h4 {{ margin-top: 0; margin-bottom: 10px; font-size: 0.95rem; color: {MUTED}; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }}
    .nid-field {{ display: flex; justify-content: space-between; padding: 3px 0; font-size: 0.92rem; border-bottom: 1px dashed rgba(128,128,128,0.15); }}
    .nid-field span:first-child {{ color: {MUTED}; }}
    .nid-field span:last-child {{ font-weight: 600; font-variant-numeric: tabular-nums; }}
    .nid-verdict {{
        border-radius: 12px; padding: 22px 24px; text-align: center; margin-bottom: 10px;
    }}
    .nid-verdict .label {{ font-size: 1.5rem; font-weight: 700; }}
    .nid-verdict .sub {{ font-size: 0.85rem; opacity: 0.85; margin-top: 4px; }}
    .nid-pill {{
        display: inline-block; padding: 2px 10px; border-radius: 999px;
        font-size: 0.75rem; font-weight: 600; letter-spacing: 0.02em;
    }}
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({
    "axes.edgecolor": "#8B93A7",
    "axes.labelcolor": "#8B93A7",
    "xtick.color": "#8B93A7",
    "ytick.color": "#8B93A7",
    "text.color": "#8B93A7",
    "axes.grid": True,
    "grid.alpha": 0.15,
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.transparent": True,
})


# ----------------------------------------------------------------------------
# Data / model loading
# ----------------------------------------------------------------------------
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree (Gini)": "decision_tree_gini.pkl",
    "Decision Tree (CART)": "decision_tree_cart_classifier.pkl",
    "AdaBoost": "adaboost.pkl",
    "XGBoost": "xgboost_model.pkl",
    "Random Forest": "random_forest.pkl",
    "Bagging": "bagging_model.pkl",
    "Voting (Hard)": "voting_hard.pkl",
    "Voting (Soft)": "voting_soft.pkl",
    "Stacking": "stacking_model.pkl",
    "SVM Linear": "svm_linear.pkl",
    "SVM RBF": "svm_rbf.pkl",
    "SVM Poly": "svm_poly.pkl",
}
# Only list models whose file actually exists on disk (cheap check, no loading).
AVAILABLE_MODELS = {
    name: fname for name, fname in MODEL_FILES.items()
    if os.path.exists(os.path.join(BASE, "models", "saved_models", fname))
}


@st.cache_resource(show_spinner="Loading model...")
def load_model_by_name(name):
    """Load a single model on demand. Cached per name so re-selecting is instant,
    but unused models are never pulled into memory."""
    path = os.path.join(BASE, "models", "saved_models", AVAILABLE_MODELS[name])
    return load_model(path)


@st.cache_resource
def load_data():
    d = os.path.join(BASE, "data", "processed")
    return {
        "X_test": np.load(os.path.join(d, "X_test.npy")),
        "y_test_bin": np.load(os.path.join(d, "y_test_binary.npy")),
        "y_test_multi": np.load(os.path.join(d, "y_test_multi.npy")),
        "y_test_sev": np.load(os.path.join(d, "y_test_severity.npy")),
    }


@st.cache_resource
def load_readable_test_set():
    """Human-readable (unscaled) version of the test rows, same row order as X_test."""
    path = os.path.join(BASE, "data", "processed", "test_preprocessed.csv")
    return pd.read_csv(path)


@st.cache_resource
def load_encoders():
    return load_model(os.path.join(BASE, "models", "encoders", "label_encoder_multi.pkl"))


data = load_data()
readable = load_readable_test_set()
le = load_encoders()
class_names = list(le.classes_)

CATEGORY_LABELS = {
    "normal": "Normal",
    "dos": "DoS",
    "probe": "Probe",
    "r2l": "R2L (Remote-to-Local)",
    "u2r": "U2R (User-to-Root)",
}

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.sidebar.markdown("### 🛰️ NIDS")
st.sidebar.caption("Network Intrusion Detection System")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Predict", "Model Comparison", "Clustering", "Dimensionality Reduction"],
    label_visibility="collapsed",
)

st.sidebar.divider()
st.sidebar.caption(f"{len(AVAILABLE_MODELS)} models available · {data['X_test'].shape[0]:,} test flows · {data['X_test'].shape[1]} features")


# ----------------------------------------------------------------------------
# Overview
# ----------------------------------------------------------------------------
if page == "Overview":
    st.title("Network Intrusion Detection")
    st.caption("Classical ML models trained on the NSL-KDD dataset to flag malicious network traffic.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Models", len(AVAILABLE_MODELS))
    c2.metric("Test flows", f"{len(data['y_test_bin']):,}")
    c3.metric("Features", data["X_test"].shape[1])
    c4.metric("Attack families", len(class_names))

    st.markdown("#### Class distribution")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(5, 3.4))
        unique_b, counts_b = np.unique(data["y_test_bin"], return_counts=True)
        ax.bar(["Normal", "Attack"], counts_b, color=[OK, DANGER], width=0.55)
        ax.set_title("Binary", fontsize=11, color=MUTED)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        st.pyplot(fig, width='stretch')

    with col2:
        fig, ax = plt.subplots(figsize=(5, 3.4))
        unique_m, counts_m = np.unique(data["y_test_multi"], return_counts=True)
        ax.bar([class_names[i] for i in unique_m], counts_m, color=PALETTE, width=0.55)
        ax.set_title("Multiclass", fontsize=11, color=MUTED)
        ax.tick_params(axis="x", rotation=30)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        st.pyplot(fig, width='stretch')

    st.markdown("#### Modules covered")
    st.dataframe(
        pd.DataFrame([
            {"Module": "1", "Topic": "ML Fundamentals", "Models / Techniques": "Bias-Variance, Learning Curves"},
            {"Module": "2", "Topic": "Regression & Trees", "Models / Techniques": "Linear/Logistic Regression, Decision Trees (Gini, CART)"},
            {"Module": "3", "Topic": "Ensemble Learning", "Models / Techniques": "AdaBoost, XGBoost, Random Forest, Bagging, Voting, Stacking"},
            {"Module": "4", "Topic": "SVM Classification", "Models / Techniques": "Linear, RBF, Poly kernels, SVR, Multiclass"},
            {"Module": "5", "Topic": "Clustering", "Models / Techniques": "DBSCAN, EM/GMM, MST"},
            {"Module": "6", "Topic": "Dimensionality Reduction", "Models / Techniques": "PCA, LDA, SVD"},
        ]),
        width='stretch',
        hide_index=True,
    )


# ----------------------------------------------------------------------------
# Predict
# ----------------------------------------------------------------------------
elif page == "Predict":
    st.title("Predict")
    st.caption("Pull a real recorded network flow from the test set and classify it — no manual feature entry required.")

    selected_model = st.selectbox("Model", list(AVAILABLE_MODELS.keys()))
    model = load_model_by_name(selected_model)

    st.markdown("##### 1 · Choose a traffic sample")
    cat_col, btn_col = st.columns([3, 1])
    with cat_col:
        category_choice = st.selectbox(
            "Draw a random flow from this category",
            list(CATEGORY_LABELS.keys()),
            format_func=lambda k: CATEGORY_LABELS[k],
        )
    with btn_col:
        st.write("")
        st.write("")
        draw = st.button("🎲 New sample", width='stretch')

    if "sample_idx" not in st.session_state or draw or st.session_state.get("sample_cat") != category_choice:
        candidates = readable.index[readable["attack_category"] == category_choice].to_numpy()
        st.session_state.sample_idx = int(np.random.choice(candidates))
        st.session_state.sample_cat = category_choice

    idx = st.session_state.sample_idx
    row = readable.iloc[idx]
    vector = data["X_test"][idx].reshape(1, -1)

    st.markdown("##### 2 · Review the flow")
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown(f"""
        <div class="nid-card"><h4>Connection</h4>
        <div class="nid-field"><span>Protocol</span><span>{row['protocol_type']}</span></div>
        <div class="nid-field"><span>Service</span><span>{row['service']}</span></div>
        <div class="nid-field"><span>Flag</span><span>{row['flag']}</span></div>
        <div class="nid-field"><span>Duration</span><span>{row['duration']}s</span></div>
        </div>""", unsafe_allow_html=True)
    with f2:
        st.markdown(f"""
        <div class="nid-card"><h4>Volume</h4>
        <div class="nid-field"><span>Src bytes</span><span>{row['src_bytes']:,}</span></div>
        <div class="nid-field"><span>Dst bytes</span><span>{row['dst_bytes']:,}</span></div>
        <div class="nid-field"><span>Count</span><span>{row['count']}</span></div>
        <div class="nid-field"><span>Srv count</span><span>{row['srv_count']}</span></div>
        </div>""", unsafe_allow_html=True)
    with f3:
        st.markdown(f"""
        <div class="nid-card"><h4>Error rates</h4>
        <div class="nid-field"><span>Serror rate</span><span>{row['serror_rate']:.2f}</span></div>
        <div class="nid-field"><span>Rerror rate</span><span>{row['rerror_rate']:.2f}</span></div>
        <div class="nid-field"><span>Same srv rate</span><span>{row['same_srv_rate']:.2f}</span></div>
        <div class="nid-field"><span>Logged in</span><span>{"Yes" if row['logged_in'] else "No"}</span></div>
        </div>""", unsafe_allow_html=True)

    with st.expander("Fine-tune this sample (optional)"):
        st.caption("Values are on the model's standardized scale, bounded to the observed range in the dataset.")
        tweak_features = ["duration", "src_bytes", "dst_bytes", "count", "srv_count", "serror_rate", "same_srv_rate"]
        feature_names = list(np.load(os.path.join(BASE, "data", "processed", "feature_names.npy"), allow_pickle=True))
        for feat in tweak_features:
            fi = feature_names.index(feat)
            col_min, col_max = float(data["X_test"][:, fi].min()), float(data["X_test"][:, fi].max())
            vector[0, fi] = st.slider(feat.replace("_", " ").title(), col_min, col_max, float(vector[0, fi]))

    st.markdown("##### 3 · Predict")
    if st.button("🔍 Classify this flow", type="primary"):
        pred = model.predict(vector)[0]
        is_attack = int(pred) == 1
        color = DANGER if is_attack else OK
        label_text = "ATTACK DETECTED" if is_attack else "NORMAL TRAFFIC"

        conf_text = ""
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(vector)[0]
            conf_text = f"{probs[int(pred)] * 100:.1f}% confidence"

        st.markdown(f"""
        <div class="nid-verdict" style="background:{color}1a; border:1px solid {color}55;">
            <div class="label" style="color:{color};">{label_text}</div>
            <div class="sub">{conf_text}</div>
        </div>
        """, unsafe_allow_html=True)

        actual = row["attack_category"]
        actual_label = row["label"]
        correct = (actual == "normal") != is_attack
        badge_color = OK if correct else WARN
        st.markdown(
            f"Ground truth: **{actual_label}** ({CATEGORY_LABELS[actual]}) "
            f"&nbsp; <span class='nid-pill' style='background:{badge_color}22;color:{badge_color};'>"
            f"{'model agrees' if correct else 'model disagrees'}</span>",
            unsafe_allow_html=True,
        )

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(vector)[0]
            st.progress(float(probs[1]), text=f"Attack probability: {probs[1]:.1%}")


# ----------------------------------------------------------------------------
# Model comparison
# ----------------------------------------------------------------------------
elif page == "Model Comparison":
    st.title("Model Comparison")
    st.caption("Binary classification performance across all trained models, evaluated on the held-out test set.")

    metric_choice = st.selectbox("Sort by", ["F1-Score", "Accuracy", "AUC", "Kappa"])

    precomputed_path = os.path.join(BASE, "data", "processed", "model_comparison.csv")
    live_mode = st.toggle(
        "Recompute live instead of using the precomputed table",
        value=False,
        help="Loads every model into memory and re-runs inference on the full test set. "
             "Off by default to keep the app's memory footprint low.",
    )

    @st.cache_data(show_spinner="Scoring models (first run only, cached after)...")
    def score_all_models(model_names, X_test, y_test_bin):
        results = []
        for name in model_names:
            model = load_model_by_name(name)
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
            m = compute_all_metrics(y_test_bin, preds, probs)
            results.append({
                "Model": name,
                "Accuracy": round(m["accuracy"], 4),
                "Precision": round(m["precision"], 4),
                "Recall": round(m["recall"], 4),
                "F1-Score": round(m["f1"], 4),
                "Kappa": round(m["kappa"], 4),
                "AUC": round(m["auc"], 4),
            })
        return pd.DataFrame(results)

    if live_mode:
        results_df = score_all_models(tuple(AVAILABLE_MODELS.keys()), data["X_test"], data["y_test_bin"])
    elif os.path.exists(precomputed_path):
        results_df = pd.read_csv(precomputed_path)
        st.caption("Showing precomputed results. Toggle above to recompute live.")
    else:
        results_df = score_all_models(tuple(AVAILABLE_MODELS.keys()), data["X_test"], data["y_test_bin"])

    df = results_df.sort_values(metric_choice, ascending=False)
    st.dataframe(
        df.style.background_gradient(subset=[metric_choice], cmap="Blues"),
        width='stretch',
        hide_index=True,
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(df["Model"], df[metric_choice], color=ACCENT)
    ax.set_xlabel(metric_choice)
    ax.invert_yaxis()
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig, width='stretch')


# ----------------------------------------------------------------------------
# Clustering
# ----------------------------------------------------------------------------
elif page == "Clustering":
    st.title("Unsupervised Clustering")
    st.caption("Supervised models only catch known attack patterns. Clustering groups similar traffic and flags outliers as potential zero-day threats.")

    plot_dir = os.path.join(BASE, "plots", "module5")
    plots = [
        ("dbscan_vs_em_comparison.png", "MST vs GMM vs DBSCAN clusters"),
        ("elbow_method.png", "DBSCAN epsilon selection"),
        ("silhouette_analysis.png", "Silhouette analysis"),
        ("mst_clustering.png", "Minimum spanning tree"),
        ("zero_day_noise.png", "Zero-day attacks in DBSCAN noise"),
    ]
    for fname, title in plots:
        fpath = os.path.join(plot_dir, fname)
        if os.path.exists(fpath):
            st.markdown(f"**{title}**")
            st.image(fpath, width='stretch')


# ----------------------------------------------------------------------------
# Dimensionality reduction
# ----------------------------------------------------------------------------
elif page == "Dimensionality Reduction":
    st.title("Dimensionality Reduction")
    st.caption("122 features compressed to ~20 components at 95% retained variance — roughly a 3x training speedup with minimal accuracy loss.")

    plot_dir = os.path.join(BASE, "plots", "module6")
    plots = [
        ("pca_explained_variance.png", "PCA explained variance"),
        ("scree_plot.png", "Scree plot"),
        ("pca_2d_projection.png", "2D PCA projection"),
        ("lda_projection.png", "LDA projection"),
        ("svd_components.png", "SVD singular values"),
        ("before_after_accuracy.png", "Before vs after comparison"),
    ]
    for fname, title in plots:
        fpath = os.path.join(plot_dir, fname)
        if os.path.exists(fpath):
            st.markdown(f"**{title}**")
            st.image(fpath, width='stretch')
