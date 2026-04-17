"""
app.py — Assembly Scrap Prediction Dashboard
LozanoLsa · Project 12 · XGBoost Classification · 2026 · FREE PROJECT

Model: XGBoost Binary Classifier
Domain: Mechatronics Assembly — Scrap Risk Detection
"""
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              RocCurveDisplay, precision_recall_curve,
                              average_precision_score)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="XGBoost · Assembly Scrap Predictor",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── FULL CSS INJECTION ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;600&family=Instrument+Serif:ital@0;1&display=swap');

:root {
    --bg:       #080c12;
    --surface:  #0e1420;
    --card:     #121922;
    --card2:    #161f2e;
    --border:   #1e2d45;
    --blue:     #3b82f6;
    --blue2:    #60a5fa;
    --teal:     #2dd4bf;
    --danger:   #f87171;
    --warn:     #fbbf24;
    --ok:       #4ade80;
    --purp:     #a78bfa;
    --text:     #c8d8f0;
    --muted:    #4e6a8a;
    --fh: 'Syne', sans-serif;
    --fm: 'JetBrains Mono', monospace;
    --fs: 'Instrument Serif', Georgia, serif;
}

.stApp { background: var(--bg) !important; color: var(--text); font-family: var(--fh); }
.block-container { padding: 1.8rem 2.4rem 3rem !important; max-width: 1400px !important; }
#MainMenu, footer, header { visibility: hidden; }

[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border) !important; }
[data-testid="stSidebar"] label { font-family: var(--fm) !important; font-size: 0.7rem !important; color: var(--text) !important; letter-spacing: 0.06em !important; text-transform: uppercase !important; }

[data-testid="stSlider"] [role="slider"] { background: var(--blue) !important; border: 2px solid var(--blue2) !important; box-shadow: 0 0 8px rgba(59,130,246,0.5) !important; }
[data-testid="stSlider"] [data-testid="stSliderThumbValue"] { font-family: var(--fm) !important; font-size: 0.65rem !important; color: var(--blue2) !important; background: var(--card) !important; border: 1px solid var(--border) !important; padding: 1px 5px !important; border-radius: 3px !important; }
[data-testid="stSlider"] > div > div > div > div { background: var(--blue) !important; }

[data-testid="stSelectbox"] > div > div { background: var(--card) !important; border: 1px solid var(--border) !important; color: var(--text) !important; font-family: var(--fm) !important; font-size: 0.78rem !important; border-radius: 3px !important; }

[data-testid="stMetric"] { background: var(--card) !important; border: 1px solid var(--border) !important; border-top: 2px solid var(--blue) !important; padding: 1rem 1.1rem 0.9rem !important; border-radius: 3px !important; }
[data-testid="stMetricLabel"] > div { font-family: var(--fm) !important; font-size: 0.6rem !important; text-transform: uppercase !important; letter-spacing: 0.18em !important; color: var(--muted) !important; font-weight: 400 !important; }
[data-testid="stMetricValue"] > div { font-family: var(--fm) !important; font-size: 1.7rem !important; font-weight: 600 !important; color: var(--blue2) !important; line-height: 1.1 !important; }

[data-testid="stTabs"] [role="tablist"] { border-bottom: 1px solid var(--border) !important; gap: 0 !important; background: transparent !important; }
[data-testid="stTabs"] [role="tab"] { font-family: var(--fm) !important; font-size: 0.68rem !important; text-transform: uppercase !important; letter-spacing: 0.12em !important; color: var(--muted) !important; padding: 0.5rem 1.2rem !important; border: none !important; border-radius: 0 !important; background: transparent !important; transition: all 0.2s !important; }
[data-testid="stTabs"] [role="tab"]:hover { color: var(--blue2) !important; background: rgba(59,130,246,0.06) !important; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] { color: var(--blue) !important; border-bottom: 2px solid var(--blue) !important; background: transparent !important; }
[data-testid="stTabsContent"] { padding-top: 1.4rem !important; }

[data-testid="stAlert"] { border-radius: 2px !important; font-family: var(--fm) !important; font-size: 0.75rem !important; letter-spacing: 0.04em !important; border: none !important; }

[data-testid="stExpander"] { background: var(--card) !important; border: 1px solid var(--border) !important; border-radius: 2px !important; margin-bottom: 6px !important; }
[data-testid="stExpander"] summary { font-family: var(--fm) !important; font-size: 0.72rem !important; color: var(--text) !important; letter-spacing: 0.06em !important; }

[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: 2px !important; }
[data-testid="stDataFrame"] th { font-family: var(--fm) !important; font-size: 0.62rem !important; text-transform: uppercase !important; letter-spacing: 0.12em !important; background: var(--card2) !important; color: var(--muted) !important; border-bottom: 1px solid var(--border) !important; }
[data-testid="stDataFrame"] td { font-family: var(--fm) !important; font-size: 0.72rem !important; color: var(--text) !important; background: var(--card) !important; }

hr { border-color: var(--border) !important; margin: 1.2rem 0 !important; }
[data-testid="stCaptionContainer"] p { font-family: var(--fm) !important; font-size: 0.62rem !important; color: var(--muted) !important; letter-spacing: 0.08em !important; }

h1, h2, h3 { font-family: var(--fh) !important; color: var(--text) !important; }
p, li { font-family: var(--fh) !important; font-size: 0.88rem !important; }

.lsa-header { border-bottom: 1px solid var(--border); padding-bottom: 1.2rem; margin-bottom: 0.2rem; }
.lsa-project-tag { font-family: var(--fm); font-size: 0.6rem; color: var(--blue); text-transform: uppercase; letter-spacing: 0.22em; margin-bottom: 4px; }
.lsa-title { font-family: var(--fh); font-size: 1.85rem; font-weight: 800; color: #fff; line-height: 1.1; letter-spacing: -0.02em; }
.lsa-tagline { font-family: var(--fs); font-style: italic; font-size: 0.9rem; color: var(--muted); margin-top: 4px; }
.lsa-chip { display: inline-block; background: rgba(59,130,246,0.1); border: 1px solid rgba(59,130,246,0.3); color: var(--blue2); font-family: var(--fm); font-size: 0.58rem; letter-spacing: 0.1em; text-transform: uppercase; padding: 2px 8px; border-radius: 2px; margin-right: 5px; }
.lsa-chip-free { display: inline-block; background: rgba(74,222,128,0.1); border: 1px solid rgba(74,222,128,0.3); color: #4ade80; font-family: var(--fm); font-size: 0.58rem; letter-spacing: 0.1em; text-transform: uppercase; padding: 2px 8px; border-radius: 2px; margin-right: 5px; }
.lsa-section { font-family: var(--fm); font-size: 0.6rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.2em; margin-bottom: 10px; padding-bottom: 5px; border-bottom: 1px solid var(--border); }
.lsa-footer { margin-top: 2.5rem; padding-top: 0.8rem; border-top: 1px solid var(--border); font-family: var(--fm); font-size: 0.58rem; color: var(--muted); letter-spacing: 0.1em; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
DATA_PATH     = "assy_scrap_data.csv"
DATA_PATH_ALT = "12_XGBoost_Advanced_Scrap_Prediction/assy_scrap_data.csv"
RANDOM_STATE  = 42
FEATURES = ["day_of_week", "applied_torque_nm", "screwdriving_speed_rpm", "motor_current_a",
            "press_pressure_bar", "operator_cycle_time_s", "relative_humidity_pct",
            "shop_floor_temp_c", "line_vibration_mm_s", "operator_experience_yrs",
            "shift_change", "material_batch"]
TARGET    = "is_scrap"
THRESHOLD = 0.25

FEAT_LABELS = {
    "day_of_week":             "Day of Week",
    "applied_torque_nm":       "Applied Torque (Nm)",
    "screwdriving_speed_rpm":  "Screwdriving Speed (rpm)",
    "motor_current_a":         "Motor Current (A)",
    "press_pressure_bar":      "Press Pressure (bar)",
    "operator_cycle_time_s":   "Operator Cycle Time (s)",
    "relative_humidity_pct":   "Relative Humidity (%)",
    "shop_floor_temp_c":       "Shop Floor Temp (°C)",
    "line_vibration_mm_s":     "Line Vibration (mm/s)",
    "operator_experience_yrs": "Operator Experience (yrs)",
    "shift_change":            "Shift Change",
    "material_batch":          "Material Batch ID",
}

# ─── MATPLOTLIB PALETTE ───────────────────────────────────────────────────────
C_BG    = "#080c12"
C_CARD  = "#121922"
C_BLUE  = "#3b82f6"
C_BLUE2 = "#60a5fa"
C_DANGER= "#f87171"
C_WARN  = "#fbbf24"
C_OK    = "#4ade80"
C_PURP  = "#a78bfa"
C_TEXT  = "#c8d8f0"
C_MUTED = "#4e6a8a"

def dark_fig(w=9, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_CARD)
    ax.tick_params(colors=C_MUTED, labelsize=9)
    ax.xaxis.label.set_color(C_MUTED)
    ax.yaxis.label.set_color(C_MUTED)
    ax.title.set_color(C_TEXT)
    for sp in ax.spines.values():
        sp.set_edgecolor("#1e2d45")
    return fig, ax

# ─── DATA & MODEL ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        return pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    except:
        return pd.read_csv(DATA_PATH_ALT, parse_dates=["timestamp"])

@st.cache_resource
def train_model(df):
    X, y = df[FEATURES], df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                           random_state=RANDOM_STATE, stratify=y)
    mdl = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                        subsample=0.8, colsample_bytree=0.8,
                        objective="binary:logistic", eval_metric="logloss",
                        random_state=RANDOM_STATE, n_jobs=-1)
    mdl.fit(Xtr, ytr)
    yp     = mdl.predict_proba(Xte)[:, 1]
    yp_cls = (yp >= THRESHOLD).astype(int)
    metrics = {
        "auc" : round(roc_auc_score(yte, yp), 4),
        "prec": round(precision_score(yte, yp_cls), 4),
        "rec" : round(recall_score(yte, yp_cls), 4),
        "f1"  : round(f1_score(yte, yp_cls), 4),
        "acc" : round(accuracy_score(yte, yp_cls), 4),
    }
    imp = (pd.DataFrame({"Feature": FEATURES, "Importance": mdl.feature_importances_})
             .sort_values("Importance", ascending=False).reset_index(drop=True))
    return mdl, Xtr, Xte, yte, yp, metrics, imp

df = load_data()
model, X_train, X_test, y_test, y_pred_proba, metrics, imp_df = train_model(df)
y_pred_cls = (y_pred_proba >= THRESHOLD).astype(int)

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="lsa-header">
    <div class="lsa-project-tag">ML Project #12 · XGBoost Classification · Mechatronics Assembly</div>
    <div class="lsa-title">Scrap Doesn't Happen Randomly</div>
    <div class="lsa-tagline">Torque deviation, vibration, and shift changeovers leave a pattern. XGBoost reads it before the unit fails.</div>
    <div style="margin-top:10px;">
        <span class="lsa-chip">XGBOOST</span>
        <span class="lsa-chip">12 FEATURES</span>
        <span class="lsa-chip">AUC {metrics['auc']:.4f}</span>
        <span class="lsa-chip">RECALL {metrics['rec']:.1%} @ thr={THRESHOLD}</span>
        <span class="lsa-chip-free">FREE PROJECT</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── TOP KPI ROW ──────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("ROC-AUC",     f"{metrics['auc']:.4f}",       "Threshold-independent discrimination")
k2.metric("Recall",      f"{metrics['rec']:.1%}",        f"@ threshold {THRESHOLD} · 7 in 10 caught")
k3.metric("F1-Score",    f"{metrics['f1']:.4f}",         "Balanced precision-recall")
k4.metric("Scrap Rate",  f"{df[TARGET].mean()*100:.1f}%","Overall line scrap rate")

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "DATA EXPLORER", "PERFORMANCE", "RISK SIMULATOR", "RISK DRIVERS", "ACTION PLAN"
])

# ══ TAB 1 ══════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="lsa-section">// Dataset — 10,000 assembly cycles</div>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Records", "10,000")
    c2.metric("Features", "12")
    c3.metric("Scrap Rate", f"{df[TARGET].mean()*100:.1f}%")
    with st.expander("Preview first 20 rows"):
        st.dataframe(df.head(20), use_container_width=True)

    st.divider()
    ca, cb = st.columns(2)
    with ca:
        st.markdown('<div class="lsa-section">// Scrap rate by vibration band</div>',
                    unsafe_allow_html=True)
        df["vib_band"] = pd.cut(df["line_vibration_mm_s"], [0, 1.5, 3.0, 3.5, 6.0],
                                labels=["Low", "Moderate", "High", "Very High"])
        scrap_vib = df.groupby("vib_band", observed=True)[TARGET].mean() * 100
        fig, ax = dark_fig(7, 4)
        bar_c = [C_OK if v < 30 else C_WARN if v < 40 else C_DANGER
                 for v in scrap_vib.values]
        ax.bar(scrap_vib.index, scrap_vib.values, color=bar_c, alpha=0.82, edgecolor=C_BG)
        ax.axhline(df[TARGET].mean() * 100, color="white", ls="--", lw=1.5,
                   label=f"Overall avg {df[TARGET].mean()*100:.1f}%")
        ax.set_xlabel("Vibration Band"); ax.set_ylabel("Scrap Rate (%)")
        ax.legend(fontsize=8, facecolor=C_CARD, labelcolor=C_TEXT)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.caption("Very High vibration (>3.5 mm/s) drives 46% scrap rate vs 27% overall — "
                   "the strongest single categorical signal in the data.")

    with cb:
        st.markdown('<div class="lsa-section">// Scrap rate by operator experience</div>',
                    unsafe_allow_html=True)
        df["exp_band"] = pd.cut(df["operator_experience_yrs"], [-1, 1, 4, 9, 15],
                                labels=["Novice", "Junior", "Mid", "Senior"])
        scrap_exp = df.groupby("exp_band", observed=True)[TARGET].mean() * 100
        fig, ax = dark_fig(7, 4)
        bar_c2 = [C_DANGER if v > 35 else C_WARN if v > 30 else C_OK
                  for v in scrap_exp.values]
        ax.bar(scrap_exp.index, scrap_exp.values, color=bar_c2, alpha=0.82, edgecolor=C_BG)
        ax.axhline(df[TARGET].mean() * 100, color="white", ls="--", lw=1.5)
        ax.set_xlabel("Experience Level"); ax.set_ylabel("Scrap Rate (%)")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.caption("Novice operators (0-1 year) show 35% scrap rate. Effect flattens after 2 years — "
                   "nonlinear threshold XGBoost captures natively.")

    st.divider()
    st.markdown('<div class="lsa-section">// Feature distribution — OK vs Scrap</div>',
                unsafe_allow_html=True)
    sel = st.selectbox("Feature:", FEATURES, format_func=lambda x: FEAT_LABELS.get(x, x))
    fig, ax = dark_fig(10, 4)
    ok    = df[df[TARGET] == 0][sel]
    scrap = df[df[TARGET] == 1][sel]
    bins  = np.linspace(df[sel].min(), df[sel].max(), 35)
    ax.hist(ok,    bins=bins, alpha=0.60, color=C_BLUE,   density=True, label="OK")
    ax.hist(scrap, bins=bins, alpha=0.60, color=C_DANGER, density=True, label="Scrap")
    ax.set_xlabel(FEAT_LABELS.get(sel, sel)); ax.set_ylabel("Density")
    ax.legend(fontsize=9, facecolor=C_CARD, labelcolor=C_TEXT)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    st.caption("Overlapping distributions confirm single-feature classification is insufficient — "
               "XGBoost models all 12 variables simultaneously.")

# ══ TAB 2 ══════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="lsa-section">// XGBoost performance — test set (n=2,000)</div>',
                unsafe_allow_html=True)
    ca, cb = st.columns(2)
    with ca:
        st.markdown('<div class="lsa-section">// ROC curve</div>', unsafe_allow_html=True)
        fig, ax = dark_fig(6, 5)
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, color=C_BLUE,
                                       name=f"XGBoost (AUC={metrics['auc']:.3f})")
        ax.plot([0, 1], [0, 1], color=C_MUTED, ls="--", lw=1.2, label="Random")
        ax.legend(fontsize=8, facecolor=C_CARD, labelcolor=C_TEXT)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.caption("AUC reflects realistic production data with genuine stochastic scrap. "
                   "The model learns structure that control charts miss.")

    with cb:
        st.markdown('<div class="lsa-section">// Precision-recall curve</div>',
                    unsafe_allow_html=True)
        fig, ax = dark_fig(6, 5)
        prec_arr, rec_arr, thr_arr = precision_recall_curve(y_test, y_pred_proba)
        ap = average_precision_score(y_test, y_pred_proba)
        ax.plot(rec_arr, prec_arr, color=C_BLUE, lw=1.5, label=f"XGBoost (AP={ap:.3f})")
        ax.axhline(y_test.mean(), color=C_MUTED, ls="--", lw=1.2, label="Baseline")
        idx = np.argmin(np.abs(thr_arr - THRESHOLD))
        ax.scatter([rec_arr[idx]], [prec_arr[idx]], s=80, color=C_WARN, zorder=5,
                   label=f"thr={THRESHOLD}")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.legend(fontsize=8, facecolor=C_CARD, labelcolor=C_TEXT)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.caption("Amber dot = operational threshold. High recall = fewer missed scrap units; "
                   "precision tradeoff depends on cost of false alarms.")

    st.divider()
    st.markdown('<div class="lsa-section">// Threshold comparison — 0.50 default vs 0.25 recall-optimised</div>',
                unsafe_allow_html=True)
    y50 = (y_pred_proba >= 0.5).astype(int)
    y25 = (y_pred_proba >= 0.25).astype(int)
    cm50 = confusion_matrix(y_test, y50); cm25 = confusion_matrix(y_test, y25)
    tn50, fp50, fn50, tp50 = cm50.ravel()
    tn25, fp25, fn25, tp25 = cm25.ravel()
    tbl = pd.DataFrame({
        "Threshold":         ["0.50 (default)", "0.25 (recall-optimised)"],
        "Recall":            [f"{recall_score(y_test,y50):.1%}",    f"{recall_score(y_test,y25):.1%}"],
        "Precision":         [f"{precision_score(y_test,y50):.1%}", f"{precision_score(y_test,y25):.1%}"],
        "F1":                [f"{f1_score(y_test,y50):.3f}",        f"{f1_score(y_test,y25):.3f}"],
        "Missed Scrap (FN)": [fn50, fn25],
        "False Alarms (FP)": [fp50, fp25],
    })
    st.dataframe(tbl, use_container_width=True, hide_index=True)
    st.markdown(f"""
    <div style="background:var(--card);border:1px solid var(--border);
                border-left:3px solid {C_WARN};border-radius:2px;
                padding:0.9rem 1.2rem;margin-top:10px;">
        <div style="font-family:var(--fm);font-size:0.58rem;color:var(--muted);
                    text-transform:uppercase;letter-spacing:.18em;margin-bottom:6px;">// Business decision</div>
        <div style="font-family:var(--fm);font-size:0.72rem;color:var(--text);line-height:1.7;">
            Threshold selection is a cost function, not a statistics question.
            If a missed scrap costs more than a false alarm (almost always true), use thr=0.25.
            At 0.25 you catch <strong style="color:{C_WARN};">{tp25} of {tp25+fn25}</strong> scrap units
            vs only {tp50} at 0.50.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="lsa-section">// Metric explanations</div>', unsafe_allow_html=True)
    for name, expl in {
        "ROC-AUC": "Threshold-independent discrimination power. Measures how well the model separates scrap from OK across all possible thresholds.",
        "Recall":  f"At threshold {THRESHOLD}: out of all truly scrap units, how many the model flags. High recall = fewer missed scraps reaching the customer.",
        "F1 Score":"Harmonic mean of precision and recall. Best single metric when class imbalance is present — more informative than accuracy alone.",
    }.items():
        with st.expander(f"{name}  —  {metrics.get(name.lower().replace('-','').replace(' ','')[:3], '—')}"):
            st.write(expl)

# ══ TAB 3 ══════════════════════════════════════════════════════════════════════
with tab3:
    medians = X_train.mean(numeric_only=True).to_dict()
    ci, co  = st.columns([1.1, 1])

    with ci:
        st.markdown('<div class="lsa-section">// Process parameters</div>', unsafe_allow_html=True)
        torque   = st.slider("Applied Torque (Nm)",      1.2,  2.4, 1.85, 0.05)
        speed    = st.slider("Screwdriving Speed (rpm)", 600, 1000,  800,  10)
        current  = st.slider("Motor Current (A)",        0.5,  3.0, 1.8,  0.05)
        pressure = st.slider("Press Pressure (bar)",     3.5,  6.5, 5.0,  0.1)
        vibration= st.slider("Line Vibration (mm/s)",    0.3,  6.0, 1.0,  0.1)

        st.markdown('<div class="lsa-section">// Operator & environment</div>',
                    unsafe_allow_html=True)
        exp      = st.slider("Operator Experience (years)", 0, 15, 5)
        shift    = st.selectbox("Shift Change", options=[0, 1],
                                format_func=lambda x: "Normal shift" if x == 0 else "Shift changeover")
        cycle_t  = st.slider("Operator Cycle Time (s)", 10, 35, 18, 1)
        humidity = st.slider("Relative Humidity (%)",   25, 75, 40, 1)
        temp     = st.slider("Shop Floor Temp (°C)",    18, 32, 24, 1)
        dow      = st.selectbox("Day of Week", list(range(7)),
                                format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
        batch    = st.slider("Material Batch ID", 1, 50, 25)

    params = medians.copy()
    params.update({
        "applied_torque_nm": torque, "screwdriving_speed_rpm": speed,
        "motor_current_a": current, "press_pressure_bar": pressure,
        "line_vibration_mm_s": vibration, "operator_experience_yrs": exp,
        "shift_change": shift, "operator_cycle_time_s": cycle_t,
        "relative_humidity_pct": humidity, "shop_floor_temp_c": temp,
        "day_of_week": dow, "material_batch": batch,
    })
    xsim    = pd.DataFrame([[params[c] for c in FEATURES]], columns=FEATURES)
    prob    = model.predict_proba(xsim)[0, 1]
    is_flag = prob >= THRESHOLD
    qual_c  = C_DANGER if is_flag else C_OK
    qual_l  = "FLAG — SCRAP RISK DETECTED" if is_flag else "OK TO PROCEED"
    badge_bg= "#2e0f0f" if is_flag else "#0f2e1a"

    with co:
        st.markdown(
            f'''<div style="background:var(--card);border:1px solid var(--border);
                        border-radius:4px;padding:1.6rem 1.8rem;">
                <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:800;
                            color:#fff;margin-bottom:1rem;">Prediction Result</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:3.4rem;
                            font-weight:700;color:{qual_c};line-height:1;
                            letter-spacing:-0.02em;">{prob:.1%}</div>
                <div style="margin-top:14px;">
                    <span style="background:{badge_bg};color:{qual_c};
                                 font-family:'JetBrains Mono',monospace;font-size:0.72rem;
                                 font-weight:600;letter-spacing:.08em;
                                 padding:5px 16px;border-radius:20px;">{qual_l}</span>
                </div>
                <div style="margin-top:18px;font-family:'JetBrains Mono',monospace;
                            font-size:0.68rem;color:var(--muted);line-height:2.1;">
                    P(Scrap) threshold : {THRESHOLD}<br>
                    Margin &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;:
                    <strong style="color:{qual_c};">{prob-THRESHOLD:+.1%}</strong>
                    {'(flagged)' if is_flag else '(clear)'}
                </div>
            </div>''',
            unsafe_allow_html=True
        )

    st.divider()
    st.markdown('<div class="lsa-section">// P(Scrap) position vs threshold</div>',
                unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(9, 1.4))
    fig.patch.set_facecolor(C_BG); ax.set_facecolor(C_BG)
    ax.barh(0, 0.95, left=0.05, height=0.55, color="#1e2d45")
    # Green = safe zone (below threshold)
    ax.barh(0, THRESHOLD - 0.05, left=0.05, height=0.55,
            color=(0.29, 0.87, 0.50, 0.18))
    ax.axvline(THRESHOLD, color=C_WARN, lw=1.5, ls=":")
    mc = C_DANGER if is_flag else C_OK
    ax.plot([prob, prob], [-0.38, 0.38], color=mc, lw=2.5)
    ax.scatter([prob], [0], s=130, color=mc, zorder=5)
    ax.set_xlim(0, 1); ax.set_ylim(-0.65, 0.65); ax.set_yticks([])
    ax.tick_params(colors=C_MUTED, labelsize=8)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_xlabel("P(Scrap)", color=C_MUTED, fontsize=9)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    st.caption(f"Green zone = below threshold {THRESHOLD} · Amber line = decision boundary.")

    st.divider()
    st.markdown('<div class="lsa-section">// Three reference scenarios</div>',
                unsafe_allow_html=True)
    scen = {
        "A · Controlled Conditions":          {
            "applied_torque_nm":1.85,"motor_current_a":1.8,"line_vibration_mm_s":1.0,
            "press_pressure_bar":5.0,"operator_experience_yrs":5,"shift_change":0,
            "relative_humidity_pct":40,"shop_floor_temp_c":24,
        },
        "B · All Risks Active":               {
            "applied_torque_nm":1.4,"motor_current_a":2.5,"line_vibration_mm_s":4.8,
            "press_pressure_bar":4.2,"operator_experience_yrs":0,"shift_change":1,
            "relative_humidity_pct":70,"shop_floor_temp_c":29,
        },
        "C · Process Fixed, Human Risk Remains":{
            "applied_torque_nm":1.85,"motor_current_a":1.8,"line_vibration_mm_s":1.5,
            "press_pressure_bar":5.0,"operator_experience_yrs":0,"shift_change":1,
            "relative_humidity_pct":70,"shop_floor_temp_c":29,
        },
    }
    sc_preds = {}; cols_s = st.columns(3)
    for col, (name, p) in zip(cols_s, scen.items()):
        base2 = medians.copy(); base2.update(p)
        pm    = model.predict_proba(
            pd.DataFrame([[base2[c] for c in FEATURES]], columns=FEATURES)
        )[0, 1]
        sc_preds[name] = pm
        flag  = pm >= THRESHOLD
        c_col = C_DANGER if flag else C_OK
        with col:
            st.markdown(f"""
            <div style="background:var(--card);border:1px solid var(--border);
                        border-left:3px solid {c_col};border-radius:2px;padding:1.1rem 1.2rem;">
                <div style="font-family:var(--fm);font-size:0.6rem;color:var(--muted);
                            letter-spacing:.15em;text-transform:uppercase;margin-bottom:8px;">{name}</div>
                <div style="font-family:var(--fm);font-size:2.4rem;font-weight:700;
                            color:{c_col};line-height:1;">{pm:.1%}</div>
                <div style="font-family:var(--fm);font-size:0.72rem;color:var(--muted);
                            margin-top:4px;">{'Flag' if flag else 'OK'}</div>
            </div>""", unsafe_allow_html=True)

    pv = list(sc_preds.values())
    st.markdown(f"""
    <div style="background:var(--card);border:1px solid var(--border);
                border-left:3px solid {C_WARN};border-radius:2px;
                padding:0.9rem 1.2rem;margin-top:10px;">
        <div style="font-family:var(--fm);font-size:0.58rem;color:var(--muted);
                    text-transform:uppercase;letter-spacing:.18em;margin-bottom:6px;">// Process vs human risk</div>
        <div style="font-family:var(--fm);font-size:0.72rem;color:var(--text);line-height:1.7;">
            Correcting torque and vibration (B→C) reduces risk by
            <strong style="color:{C_WARN};">{pv[1]-pv[2]:.0%} pp</strong>.
            Residual risk ({pv[2]:.0%}) from novice operator + shift change cannot be
            eliminated by process adjustment alone — pair with a mentor.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ══ TAB 4 ══════════════════════════════════════════════════════════════════════
with tab4:
    ca, cb = st.columns([1.2, 1])
    with ca:
        st.markdown('<div class="lsa-section">// XGBoost feature importance (gain)</div>',
                    unsafe_allow_html=True)
        fig, ax = dark_fig(7, 6)
        imp_s = imp_df.sort_values("Importance", ascending=True)
        q75 = imp_df["Importance"].quantile(0.75)
        q50 = imp_df["Importance"].quantile(0.50)
        bar_c = [C_DANGER if v > q75 else C_WARN if v > q50 else C_BLUE
                 for v in imp_s["Importance"]]
        bars  = ax.barh(
            [FEAT_LABELS.get(f, f) for f in imp_s["Feature"]],
            imp_s["Importance"], color=bar_c, alpha=0.82, edgecolor="none", height=0.65
        )
        for bar, val in zip(bars, imp_s["Importance"]):
            ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=9, color=C_TEXT)
        ax.set_xlabel("Gain (avg reduction in impurity per split)")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.caption("Red = top quartile. shift_change tops the ranking — changeover windows "
                   "are the highest-information splits for predicting scrap.")

    with cb:
        st.markdown('<div class="lsa-section">// Feature importance table</div>',
                    unsafe_allow_html=True)
        imp_tbl = imp_df.copy()
        imp_tbl["Feature"]    = imp_tbl["Feature"].map(lambda x: FEAT_LABELS.get(x, x))
        imp_tbl["Importance"] = imp_tbl["Importance"].map("{:.5f}".format)
        imp_tbl["Rank"]       = range(1, len(imp_df) + 1)
        st.dataframe(imp_tbl[["Rank", "Feature", "Importance"]],
                     use_container_width=True, hide_index=True)

        st.markdown(f"""
        <div style="background:var(--card);border:1px solid var(--border);
                    border-left:3px solid {C_WARN};border-radius:2px;
                    padding:1rem 1.2rem;margin-top:12px;">
            <div style="font-family:var(--fm);font-size:0.58rem;color:var(--muted);
                        text-transform:uppercase;letter-spacing:.18em;margin-bottom:6px;">// XGBoost vs linear models</div>
            <div style="font-family:var(--fm);font-size:0.72rem;color:var(--text);line-height:1.7;">
                Linear models assign one coefficient per feature. XGBoost can represent
                <em>threshold effects</em> (torque below 1.6 Nm suddenly jumps scrap risk) and
                <em>interactions</em> (low torque + high vibration = synergistic risk).
                Neither is captured by a single coefficient.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="lsa-section">// Risk heatmap — torque × vibration</div>',
                unsafe_allow_html=True)
    torque_r    = np.linspace(1.2, 2.3, 50)
    vibration_r = np.linspace(0.3, 6.0, 50)
    T, V = np.meshgrid(torque_r, vibration_r)
    base_surf = medians.copy()
    grid = []
    for t, v in zip(T.ravel(), V.ravel()):
        row = base_surf.copy()
        row["applied_torque_nm"]   = t
        row["line_vibration_mm_s"] = v
        grid.append([row[c] for c in FEATURES])
    Z = model.predict_proba(pd.DataFrame(grid, columns=FEATURES))[:, 1].reshape(T.shape)
    fig, ax = dark_fig(10, 5.5)
    cf   = ax.contourf(T, V, Z, levels=25, cmap="RdYlGn_r", alpha=0.88)
    cbar = plt.colorbar(cf, ax=ax)
    cbar.set_label("P(Scrap)", color=C_MUTED)
    cbar.ax.yaxis.set_tick_params(color=C_MUTED)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=C_MUTED)
    cs2 = ax.contour(T, V, Z, levels=[THRESHOLD], colors=["white"], linewidths=2.0)
    ax.clabel(cs2, fmt=f"thr={THRESHOLD}", fontsize=9, colors="white")
    ax.contourf(T, V, Z, levels=[0, THRESHOLD],
                colors=["lime"], alpha=0.12, hatches=["////"])
    ax.set_xlabel("Applied Torque (Nm)"); ax.set_ylabel("Line Vibration (mm/s)")
    ax.set_title("Scrap Risk Map — Torque × Vibration (medians for other features)",
                 color=C_TEXT)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    st.caption("Green-hatched = safe zone. Both parameters must be in range simultaneously — "
               "correct torque alone is not enough if vibration is high.")

# ══ TAB 5 ══════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="lsa-section">// Operational recommendations</div>',
                unsafe_allow_html=True)
    actions = [
        (C_DANGER, f"Use threshold {THRESHOLD} as the go/no-go gate, not 0.50",
         f"At threshold 0.50, the model catches only 16% of scrap (84% escape). "
         f"At threshold {THRESHOLD}, recall rises to {metrics['rec']:.0%} with acceptable false alarm rate. "
         f"For most assembly operations the cost of a missed scrap exceeds the cost of a re-inspection."),
        (C_BLUE,   "Mount anti-vibration pads when line_vibration_mm_s > 3.5",
         "Very-high vibration (>3.5 mm/s) raises scrap rate to 46% vs 27% average. "
         "Intervention cost (pad replacement, fixture tightening) is typically <€50/event; "
         "scrap cost is orders of magnitude higher. Trigger maintenance at 3.0 mm/s."),
        (C_WARN,   "Pair novice operators with mentors during shift changeovers",
         "shift_change and operator_experience_yrs are the top two XGBoost features by gain. "
         "Their interaction is the highest-risk scenario in the dataset (Scenario B: 95% P(scrap)). "
         "Never assign a 0-year operator as the sole operator during a changeover window."),
        (C_OK,     "Alert when humidity > 65% and act before production starts",
         "Humidity >65% adds a significant risk factor independently of process parameters — "
         "ESD damage, adhesive cure time variation, and connector oxidation all increase. "
         "Integrate the humidity sensor trigger into the pre-shift checklist."),
        (C_PURP,   "Retrain monthly as batches rotate and operators change",
         "The model was trained on specific material batch IDs and an operator pool. "
         "As new batches arrive and operator experience levels shift, retrain with accumulated data. "
         "XGBoost retraining with 10,000 records takes under 30 seconds — this is not a barrier."),
    ]
    for color, title, body in actions:
        st.markdown(f"""
        <div style="background:var(--card);border:1px solid var(--border);
                    border-left:3px solid {color};border-radius:2px;
                    padding:1.1rem 1.3rem;margin-bottom:10px;">
            <div style="font-family:var(--fm);font-size:0.72rem;font-weight:600;
                        color:{color};margin-bottom:6px;">{title}</div>
            <div style="font-family:var(--fm);font-size:0.7rem;color:var(--muted);
                        line-height:1.7;">{body}</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    <div style="background:var(--card);border:1px solid var(--border);border-radius:2px;
                padding:1rem 1.3rem;text-align:center;">
        <div style="font-family:var(--fm);font-size:0.6rem;color:var(--muted);
                    text-transform:uppercase;letter-spacing:.18em;margin-bottom:6px;">// Free project</div>
        <div style="font-family:var(--fm);font-size:0.68rem;color:var(--muted);line-height:1.7;">
            Full dataset + simulator included. Check the rest of the portfolio at
            <span style="color:#3b82f6;">lozanolsa.gumroad.com</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="lsa-footer">
    LozanoLsa · Turning Operations into Predictive Systems · Assembly Scrap Predictor · Project 12 · v2.0
</div>
""", unsafe_allow_html=True)
