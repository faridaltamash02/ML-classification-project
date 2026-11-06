"""Bank Marketing Evaluation Studio.

Run from the project root with: python -m streamlit run app.py
The application loads previously saved, trusted model pipelines only; it never fits models.
"""

from __future__ import annotations

from html import escape
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

# Match the files already trained for the assignment.
MODELS = {
    "Naive Bayes": {
        "file": "naive_bayes.joblib",
        "description": "Fast probabilistic baseline",
        "number": "01",
    },
    "Logistic Regression": {
        "file": "logistic_regression.joblib",
        "description": "Linear and interpretable",
        "number": "02",
    },
    "Decision Tree": {
        "file": "decision_tree.joblib",
        "description": "Rule-based classifier",
        "number": "03",
    },
    "KNN": {
        "file": "knn.joblib",
        "description": "Nearest-neighbor classifier",
        "number": "04",
    },
    "Support Vector Machine": {
        "file": "support_vector_machine.joblib",
        "description": "Maximum-margin classifier",
        "number": "05",
    },
}

FEATURE_COLUMNS = [
    "age", "job", "marital", "education", "default", "balance", "housing",
    "loan", "contact", "day", "month", "campaign", "pdays", "previous",
    "poutcome",
]
NUMERIC_COLUMNS = ["age", "balance", "day", "campaign", "pdays", "previous"]
CATEGORICAL_COLUMNS = [c for c in FEATURE_COLUMNS if c not in NUMERIC_COLUMNS]

st.set_page_config(
    page_title="Bank Marketing Evaluation | ML Studio",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>

/* File uploader - Browse files button */
[data-testid="stFileUploader"] button {
    background-color: #1E6048 !important;
    color: #FFFFFF !important;
    border: 1px solid #1E6048 !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 10px 22px !important;
    transition: all 0.2s ease !important;
}

/* Ensure button text is white */
[data-testid="stFileUploader"] button * {
    color: #FFFFFF !important;
}

/* Hover effect */
[data-testid="stFileUploader"] button:hover {
    background-color: #164D39 !important;
    border-color: #164D39 !important;
    color: #FFFFFF !important;
}

/* File uploader container */
[data-testid="stFileUploaderDropzone"] {
    background-color: #FFFFFF !important;
    border: 2px dashed #C9DED2 !important;
    border-radius: 16px !important;
}

/* Drag and drop instruction text */
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] small {
    color: #465A50 !important;
}


/* Uploaded file name */
[data-testid="stFileUploaderFileName"] {
    color: #263B31 !important;
    font-weight: 600 !important;
}

/* Uploaded file details, including file size */
[data-testid="stFileUploaderFileData"] {
    color: #465A50 !important;
}

[data-testid="stFileUploaderFileData"] span,
[data-testid="stFileUploaderFileData"] small {
    color: #465A50 !important;
}

/* Keep Browse files button text white */
[data-testid="stFileUploader"] button,
[data-testid="stFileUploader"] button * {
    color: #FFFFFF !important;
}

/* Keep uploaded file remove button visible */
[data-testid="stFileUploaderFile"] button,
[data-testid="stFileUploaderFile"] button * {
    color: #263B31 !important;
}

/* ==========================================
   1. EXPANDER HEADERS
   ========================================== */

[data-testid="stExpander"] {
    background: #FFFFFF !important;
    border: 1px solid #DEE8E1 !important;
    border-radius: 14px !important;
    overflow: hidden;
    margin-bottom: 16px;
}

[data-testid="stExpander"] details summary {
    background: #EAF3ED !important;
    border-radius: 12px !important;
    padding: 14px 18px !important;
}

/* Expander title and arrow */
[data-testid="stExpander"] details summary p,
[data-testid="stExpander"] details summary span,
[data-testid="stExpander"] details summary svg {
    color: #173F30 !important;
    fill: none;
}

[data-testid="stExpander"] details summary:hover {
    background: #DBEBE1 !important;
}

/* Expander content */
[data-testid="stExpanderDetails"] {
    background: #FFFFFF !important;
    color: #20372D !important;
    padding: 18px !important;
}

/* ==========================================
   2. CLASSIFICATION REPORT TABLE
   ========================================== */

[data-testid="stDataFrame"] {
    background: #FFFFFF !important;
    border: 1px solid #DEE8E1 !important;
    border-radius: 12px !important;
    overflow: hidden;
}

/*
   Streamlit dataframe uses a canvas-based grid
   in many versions, so CSS alone may not style
   every cell. Configure light theme separately.
*/

/* ==========================================
   3. DOWNLOAD CSV BUTTON
   ========================================== */

[data-testid="stDownloadButton"] button {
    background-color: #1E6048 !important;
    color: #FFFFFF !important;
    border: 1px solid #1E6048 !important;
    border-radius: 12px !important;
    padding: 12px 22px !important;
    font-weight: 600 !important;
    transition: background-color 0.2s ease !important;
}

/* Button label */
[data-testid="stDownloadButton"] button p,
[data-testid="stDownloadButton"] button span,
[data-testid="stDownloadButton"] button svg {
    color: #FFFFFF !important;
    stroke: #FFFFFF !important;
}

[data-testid="stDownloadButton"] button:hover {
    background-color: #164D39 !important;
    border-color: #164D39 !important;
}

[data-testid="stDownloadButton"] button:focus-visible {
    outline: 3px solid #91C7AC !important;
    outline-offset: 2px;
}

/* ==========================================
   4. INLINE CODE LABELS
   ========================================== */

[data-testid="stExpanderDetails"] code {
    background-color: #EAF3ED !important;
    color: #176044 !important;
    border-radius: 5px !important;
    padding: 2px 6px !important;
}



</style>
""", unsafe_allow_html=True)

# Theme-aware, responsive visual treatment approximating the supplied reference.
st.markdown(
    """
<style>
:root {
  --ink: #192922;
  --muted: #75857d;
  --dark-green: #18543f;
  --deep-green: #123e2f;
  --soft-green: #e4efe9;
  --soft-red: #f5e8e3;
  --card: #ffffff;
  --paper: #f7f9f7;
  --line: #e4e9e5;
}
.stApp {background: var(--paper); color: var(--ink);}
.block-container {max-width: 1700px; padding: 1.2rem 2.1rem 2.8rem;}
header[data-testid="stHeader"] {background: transparent;}
#MainMenu, footer {visibility: hidden;}
/* Header */
.hero {display:flex;align-items:center;gap:20px;padding:21px 0 26px;border-bottom:1px solid var(--line);margin-bottom:22px;}
.hero-logo {width:66px;height:66px;border-radius:18px;background:var(--deep-green);color:white;
  display:flex;align-items:center;justify-content:center;font-size:33px;box-shadow:0 10px 20px #153d2b1c;flex-shrink:0;}
.eyebrow {text-transform:uppercase;letter-spacing:.19em;font-weight:800;color:#73857a;font-size:11px;}
.hero h1 {font-size:clamp(26px,3.0vw,41px);letter-spacing:-.05em;line-height:1.12;margin:5px 0 6px;font-weight:800;color:var(--ink);}
.hero p {margin:0;color:#687a71;font-size:15px;}
.hero-spacer {flex:1;}
.ready-pill {border:1px solid #dbe9df;background:#ecf6ef;color:#25654e;border-radius:999px;
  font-size:13px;font-weight:750;padding:11px 15px;white-space:nowrap;}
.ready-dot {display:inline-block;width:9px;height:9px;border-radius:50%;background:#3ca36b;margin-right:7px;box-shadow:0 0 0 5px #d4eddc;}
/* content columns */
div[data-testid="stHorizontalBlock"] {gap:1.65rem;}
.section-title {display:flex;align-items:flex-start;gap:13px;margin:12px 0 20px;}
.section-number {background:#dfece4;border-radius:10px;padding:9px 11px;font-size:13px;font-weight:800;color:#30614b;line-height:1;}
.section-title h2 {font-size:20px;letter-spacing:-.035em;color:var(--ink);margin:0 0 5px;font-weight:800;}
.section-title p {font-size:13px;color:var(--muted);margin:0;}
.field-label {font-size:13px;font-weight:750;color:#35463d;letter-spacing:.01em;margin:16px 0 7px;}
.small-caps {font-size:11px;color:#96a39b;text-transform:uppercase;letter-spacing:.12em;font-weight:800;}
/* Inputs */
div[data-testid="stFileUploader"] {background:#fff;border-radius:15px;}
div[data-testid="stFileUploader"] section {border:1px dashed #cbdad0;border-radius:15px;background:#fcfefd;}
div[data-testid="stRadio"]>div {gap:7px;}
div[data-testid="stRadio"] label {border:1px solid #e0e7e1;background:#fff;border-radius:13px;padding:12px 11px;min-height:52px;transition:border-color .2s,background .2s;}
div[data-testid="stRadio"] label:has(input:checked) {border-color:#2c7758;background:#e6f0ea;box-shadow:inset 3px 0 #236f50;}
div[data-testid="stRadio"] label p {font-weight:650;color:#243a30;}
div[data-testid="stRadio"] input {accent-color:#237250;}
div.stButton>button[kind="primary"] {background:#1b6047;color:#fff;border:none;border-radius:12px;
  font-weight:800;min-height:54px;box-shadow:0 8px 15px #16412e17;}
div.stButton>button[kind="primary"]:hover {background:#124631;color:white;border:none;}
/* Result cards */
.metric-grid {display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:13px;margin:10px 0 15px;}
.metric-tile {background:white;border:1px solid #e4e9e5;border-radius:17px;padding:19px 19px 17px;min-height:137px;
  box-shadow:0 5px 17px #1f3f2e08;}
.metric-top {display:flex;justify-content:space-between;align-items:center;gap:6px;}
.metric-name {font-size:12px;color:#6d7d73;font-weight:800;}
.metric-dot {font-size:19px;line-height:0;color:#68a082;}
.metric-val {font-size:clamp(21px,2.5vw,37px);letter-spacing:-.055em;font-weight:850;color:#182d23;margin-top:17px;}
.metric-desc {font-size:11px;color:#91a098;margin-top:5px;}
.results-panel {background:white;border:1px solid var(--line);border-radius:21px;padding:25px 24px;
  box-shadow:0 8px 21px #1d392b07;margin-top:9px;}
.panel-heading {font-weight:800;color:#1b2b22;font-size:21px;margin:7px 0 20px;letter-spacing:-.04em;}
.legend {font-size:11px;color:#7c8c83;text-align:right;margin-top:-43px;margin-bottom:18px;}
.legend .good {color:#8abda2}.legend .bad {color:#d5a79b}
.matrix-grid {display:grid;grid-template-columns:1fr 1fr;gap:9px;margin:0 0 12px 19px;}
.matrix-cell {padding:20px 22px;border-radius:16px;min-height:132px;position:relative;}
.matrix-good {background:var(--soft-green);}
.matrix-bad {background:var(--soft-red);}
.matrix-label {font-size:11px;color:#62756a;}
.matrix-number {font-size:clamp(26px,3vw,39px);color:#1c3528;font-weight:850;letter-spacing:-.06em;margin-top:12px;}
.matrix-code {position:absolute;right:18px;bottom:16px;color:#91a69a;font-size:11px;font-weight:850;}
.axis-label {font-size:10px;color:#99a79d;letter-spacing:.18em;text-align:center;text-transform:uppercase;font-weight:800;}
.note {color:#75877a;font-size:12px;line-height:1.6;margin:10px 0 0;}
.model-chip {color:#245942;background:#edf6ef;padding:7px 11px;border-radius:25px;font-size:12px;font-weight:800;}
.report-title {color:#1f3329;font-size:17px;font-weight:800;margin:12px 0;}
.app-footer {margin-top:27px;border-top:1px solid var(--line);padding:18px 0;color:#7c8b81;font-size:12px;display:flex;justify-content:space-between;}
.
@media(max-width:900px) {
  .block-container {padding:1.1rem .9rem 2rem;}
  .metric-grid {grid-template-columns:repeat(2,minmax(0,1fr));}
  .hero {gap:12px;}
  .hero-logo {width:49px;height:49px;border-radius:13px;font-size:25px;}
  .ready-pill {display:none;}
  .results-panel {padding:18px 12px;}
  .matrix-cell {min-height:112px;padding:14px;}
  .matrix-grid {margin-left:0;}
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading trained model...")
def load_model(filename: str):
    """Load trusted, application-owned artifacts only, never uploaded model files."""
    allowed = {item["file"] for item in MODELS.values()}
    if filename not in allowed:
        raise ValueError("Unknown model selection.")
    path = MODEL_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Model not found: {path.name}")
    return joblib.load(path)


def prepare_test_data(df: pd.DataFrame):
    if df.empty:
        raise ValueError("The CSV is empty.")
    if df.columns.duplicated().any():
        raise ValueError("CSV has duplicate column names.")
    required = FEATURE_COLUMNS + ["y"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    x = df.loc[:, FEATURE_COLUMNS].copy()
    y = df["y"].copy()
    if y.isna().any() or not set(y.unique()).issubset({"yes", "no"}):
        raise ValueError("Target 'y' must contain only 'yes' and 'no', with no missing values.")

    for col in NUMERIC_COLUMNS:
        x[col] = pd.to_numeric(x[col], errors="coerce")
        x[col] = x[col].replace([np.inf, -np.inf], np.nan)
    for col in CATEGORICAL_COLUMNS:
        # Preserve real missing values for the fitted imputer.
        x[col] = x[col].astype("object")
    return x, y


def model_scores(model, x):
    classes = list(model.classes_)
    if "yes" not in classes or len(classes) != 2:
        raise ValueError(f"Expected binary classes containing 'yes', got {classes}.")
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, classes.index("yes")]
    if hasattr(model, "decision_function"):
        score = model.decision_function(x)
        return score if classes[1] == "yes" else -score
    raise ValueError("Selected model does not expose suitable AUC scores.")


def evaluate(model, x, y):
    predictions = model.predict(x)
    scores = model_scores(model, x)
    metrics = {
        "Accuracy": accuracy_score(y, predictions),
        "AUC": roc_auc_score((y == "yes").astype(int), scores) if y.nunique() == 2 else float("nan"),
        "Precision": precision_score(y, predictions, pos_label="yes", zero_division=0),
        "Recall": recall_score(y, predictions, pos_label="yes", zero_division=0),
        "F1": f1_score(y, predictions, pos_label="yes", zero_division=0),
        "MCC": matthews_corrcoef(y, predictions),
    }
    cm = confusion_matrix(y, predictions, labels=["no", "yes"])
    report = classification_report(
        y, predictions, labels=["no", "yes"], output_dict=True, zero_division=0
    )
    return metrics, cm, report


def section_heading(number: str, name: str, subtitle: str):
    st.markdown(
        f'<div class="section-title"><div class="section-number">{escape(number)}</div>'
        f'<div><h2>{escape(name)}</h2><p>{escape(subtitle)}</p></div></div>',
        unsafe_allow_html=True,
    )


def show_metrics(metrics):
    descriptions = {
        "Accuracy": "Overall predictions",
        "AUC": "Ranking quality",
        "Recall": "Positive cases found",
        "MCC": "Balanced correlation",
        "Precision": "Positive prediction quality",
        "F1": "Precision-recall balance",
    }
    tiles = []
    for name in ("Accuracy", "AUC", "Recall", "MCC", "Precision", "F1"):
        val = metrics[name]
        display = (
            "N/A" if pd.isna(val) else f"{val * 100:.2f}%" if name in ("Accuracy", "Recall", "Precision")
            else f"{val:.4f}"
        )
        tiles.append(
            f'<div class="metric-tile"><div class="metric-top">'
            f'<span class="metric-name">{escape(name if name != "AUC" else "ROC AUC")}</span>'
            f'<span class="metric-dot">•</span></div><div class="metric-val">{display}</div>'
            f'<div class="metric-desc">{descriptions[name]}</div></div>'
        )
    st.markdown('<div class="metric-grid">' + ''.join(tiles) + '</div>', unsafe_allow_html=True)


def show_matrix(cm):
    tn, fp, fn, tp = cm.ravel()
    cells = [
        ("True negative", tn, "TN", "matrix-good"),
        ("False positive", fp, "FP", "matrix-bad"),
        ("False negative", fn, "FN", "matrix-bad"),
        ("True positive", tp, "TP", "matrix-good"),
    ]
    cell_markup = ''.join(
        f'<div class="matrix-cell {color}"><div class="matrix-label">{title}</div>'
        f'<div class="matrix-number">{int(number):,}</div><div class="matrix-code">{abbr}</div></div>'
        for title, number, abbr, color in cells
    )
    st.markdown(
        '<div class="results-panel">'
        '<div class="eyebrow">Prediction breakdown</div>'
        '<div class="panel-heading">Confusion matrix</div>'
        '<div class="legend"><span class="good">■</span> Correct &nbsp;&nbsp;'
        '<span class="bad">■</span> Incorrect</div>'
        '<div class="matrix-grid">' + cell_markup + '</div>'
        '<div class="axis-label">Predicted class → No / Yes &nbsp;·&nbsp; '
        'Rows: Actual No / Yes</div>'
        '</div>',
        unsafe_allow_html=True,
    )


st.markdown(
    '<div class="hero">'
    '<div class="hero-logo">↗</div>'
    '<div><div class="eyebrow">Model performance studio</div>'
    '<h1>Bank Marketing Evaluation</h1>'
    '<p>Compare classifiers and inspect their performance.</p></div>'
    '<div class="hero-spacer"></div>'
    '<div class="ready-pill"><span class="ready-dot"></span>5 models ready</div>'
    '</div>',
    unsafe_allow_html=True,
)

left, right = st.columns([0.92, 2.35], gap="large")

with left:
    section_heading("01", "Configure evaluation", "Choose a dataset and classifier")
    st.markdown('<div class="field-label">Test dataset</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload test CSV (15 features + y)", type="csv", label_visibility="collapsed"
    )
    st.markdown('<div class="field-label">Choose a model</div>', unsafe_allow_html=True)
    names = list(MODELS)
    model_name = st.radio(
        "Model", names, index=0, label_visibility="collapsed",
        format_func=lambda name: f'{name} · {MODELS[name]["description"]}'
    )
    clicked = st.button("Evaluate model  →", type="primary", use_container_width=True)
    st.caption("Saved models only · No training or fitting on uploaded data")

if clicked:
    if uploaded is None:
        st.warning("Upload test_data.csv before evaluating a model.")
    else:
        try:
            uploaded.seek(0)
            raw = pd.read_csv(uploaded)
            x, y = prepare_test_data(raw)
            fitted_model = load_model(MODELS[model_name]["file"])
            with st.spinner("Calculating predictions and performance metrics..."):
                metrics, cm, report = evaluate(fitted_model, x, y)
            st.session_state["evaluation_result"] = {
                "model_name": model_name,
                "filename": uploaded.name,
                "rows": len(raw),
                "positives": int((y == "yes").sum()),
                "metrics": metrics,
                "cm": cm,
                "report": report,
            }
        except (ValueError, FileNotFoundError, OSError, pd.errors.ParserError) as exc:
            st.error(f"Unable to evaluate: {exc}")
            st.session_state.pop("evaluation_result", None)

with right:
    state = st.session_state.get("evaluation_result")
    subtitle = (
        f'Performance on {state["filename"]}' if state
        else 'Upload a test CSV and evaluate a model'
    )
    section_heading("02", "Evaluation results", subtitle)
    if state:
        st.markdown(
            f'<div class="small-caps" style="text-align:right;margin-top:-43px;margin-bottom:25px">'
            f'ACTIVE MODEL &nbsp; <span class="model-chip">{escape(state["model_name"])}</span>'
            f'</div>', unsafe_allow_html=True
        )
        st.caption(f'{state["rows"]:,} test observations · {state["positives"]:,} actual subscribers')
        show_metrics(state["metrics"])
        show_matrix(state["cm"])
        with st.expander("Detailed classification report"):
            report_df = pd.DataFrame(state["report"]).T
            st.dataframe(report_df.round(4), use_container_width=True)
        with st.expander("Metric definitions and interpretation"):
            st.markdown(
                "**Accuracy**: Fraction classified correctly.  \n"
                "**ROC AUC**: Ranking quality across thresholds.  \n"
                "**Precision**: Correct `yes` predictions divided by all `yes` predictions.  \n"
                "**Recall**: Correct `yes` predictions divided by all actual `yes` cases.  \n"
                "**F1**: Harmonic mean of Precision and Recall.  \n"
                "**MCC**: Binary classification correlation accounting for TP, TN, FP, and FN."
            )
        export = pd.DataFrame([{"ML Model Name": state["model_name"], **state["metrics"]}])
        # st.download_button(
        #     "Download evaluation metrics (CSV)",
        #     data=export.to_csv(index=False),
        #     file_name="model_evaluation.csv", mime="text/csv"
        # )
        st.download_button(
            label="Download evaluation metrics (CSV)",
            data=export.to_csv(index=False),
            file_name="evaluation_metrics.csv",
            mime="text/csv",
            type="primary",
            icon=":material/download:",
            use_container_width=False
        )
        st.markdown('<p class="note">Precision, Recall and F1 use <b>yes</b> as the positive class. '
                    'AUC is unavailable when the uploaded CSV contains only one target class.</p>',
                    unsafe_allow_html=True)
    else:
        st.info("Upload your saved test_data.csv on the left, choose one of the five trained models, and click Evaluate model.")
        st.markdown('<div class="results-panel"><div class="eyebrow">Your evaluation dashboard</div>'
                    '<div class="panel-heading">Ready for your test data</div>'
                    '<p class="note">Once you evaluate a model, this panel will display six metrics, '
                    'a color-coded confusion matrix, and a classification report.</p></div>',
                    unsafe_allow_html=True)

st.markdown(
    '<div class="app-footer"><div><b>ML EVALUATION LAB</b> &nbsp; '
    'Metrics calculated on your uploaded test dataset.</div><div>● System ready</div></div>',
    unsafe_allow_html=True
)
