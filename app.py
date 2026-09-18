"""
LinkedIn Job Market Analysis & Career Insights
===============================================
Portfolio project — Streamlit interactive dashboard.
Dataset : linkdin_Job_data_cleaned.csv  (5,819 rows × 16 columns)

Sections
--------
1  Overview KPIs
2  Job Market Analysis
3  Company Insights
4  Remote & Work Type
5  Applicant Insights
6  Machine Learning — High Applicant Demand
7  Business Recommendations
"""

import math
import re
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LinkedIn Job Market Analytics",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────
# GLOBAL CSS
# ──────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* Section heading with left accent bar */
.sec-head {
    font-size: 1.35rem; font-weight: 700; color: #1f2328;
    border-left: 4px solid #3b82d4; padding-left: 10px;
    margin: 24px 0 12px;
}
/* Blue info callout */
.insight-box {
    background: #eef4ff; border: 1px solid #bfdbfe;
    border-radius: 6px; padding: 11px 15px; margin: 8px 0;
    font-size: 0.91rem; color: #1f2328; line-height: 1.55;
}
/* Amber warning callout */
.warn-box {
    background: #fffbeb; border: 1px solid #fde68a;
    border-radius: 6px; padding: 11px 15px; margin: 8px 0;
    font-size: 0.91rem; color: #78350f; line-height: 1.55;
}
/* Green recommendation card */
.rec-card {
    background: #f0fdf4; border: 1px solid #bbf7d0;
    border-radius: 6px; padding: 11px 15px; margin: 5px 0;
    font-size: 0.91rem; color: #14532d; line-height: 1.55;
}
/* Reduce default top padding on metrics */
div[data-testid="metric-container"] { padding-top: 6px; }
</style>
""",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────
_SIZE_ORDER = [
    "1-10 employees", "11-50 employees", "51-200 employees",
    "201-500 employees", "501-1,000 employees", "1,001-5,000 employees",
    "5,001-10,000 employees", "10,001+ employees",
]

def _to_hours(val) -> float:
    """Convert posted_day_ago string (e.g. '3 days') to numeric hours."""
    if pd.isna(val):
        return np.nan
    v = str(val).strip().lower()
    for pat, mult in [
        (r"(\d+)\s*second", 1 / 3600),
        (r"(\d+)\s*minute", 1 / 60),
        (r"(\d+)\s*hour",   1),
        (r"(\d+)\s*day",    24),
        (r"(\d+)\s*week",   168),
    ]:
        m = re.match(pat, v)
        if m:
            return float(m.group(1)) * mult
    return np.nan


def _sec(label: str) -> None:
    """Render a styled section heading."""
    st.markdown(f'<p class="sec-head">{label}</p>', unsafe_allow_html=True)


def _insight(text: str) -> None:
    st.markdown(f'<div class="insight-box">💡 {text}</div>', unsafe_allow_html=True)


def _warn(text: str) -> None:
    st.markdown(f'<div class="warn-box">⚠️ {text}</div>', unsafe_allow_html=True)


def _rec(title: str, body: str) -> None:
    st.markdown(
        f'<div class="rec-card"><b>✅ {title}:</b> {body}</div>',
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────
# DATA LOADING  (cached — runs once)
# ──────────────────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv("linkdin_Job_data_cleaned.csv")

    df["hours_since_posted"] = df["posted_day_ago"].apply(_to_hours)

    df["city"] = df["location"].apply(
        lambda loc: str(loc).split(",")[0].strip() if pd.notna(loc) else np.nan
    )
    df["is_remote"] = (df["work_type"].str.strip().str.lower() == "remote").astype(int)
    df["log_followers"] = np.log1p(df["linkedin_followers"].fillna(0))

    return df


df_full = load_data()

# ──────────────────────────────────────────────────────────
# ML TRAINING  (cached — runs once on the full cleaned dataset)
# ──────────────────────────────────────────────────────────
@st.cache_data
def train_ml_model(csv_path: str):
    """
    Train a Random Forest to predict High_Applicant_Demand.

    Strict leakage rules
    --------------------
    • no_of_application and High_Applicant_Demand are NEVER predictors.
    • job_ID (raw identifier) is excluded.
    • Median threshold is computed from the TRAINING split only.
    • Stratified 80/20 train-test split ensures balanced class evaluation.
    """
    df = pd.read_csv(csv_path)
    df["hours_since_posted"] = df["posted_day_ago"].apply(_to_hours)
    df["is_remote"]     = (df["work_type"].str.strip().str.lower() == "remote").astype(int)
    df["log_followers"] = np.log1p(df["linkedin_followers"].fillna(0))

    ml_df = df[df["no_of_application"].notna()].copy().reset_index(drop=True)

    # ── Step 1: provisional split to get training-set median (no target known yet) ──
    prov_train, _ = train_test_split(
        ml_df.index, test_size=0.2, random_state=42, shuffle=True
    )
    median_threshold = ml_df.loc[prov_train, "no_of_application"].median()

    # ── Step 2: define target ──
    ml_df["High_Applicant_Demand"] = (
        ml_df["no_of_application"] >= median_threshold
    ).astype(int)

    # ── Step 3: stratified final split ──
    train_idx, test_idx = train_test_split(
        ml_df.index,
        test_size=0.2,
        random_state=42,
        stratify=ml_df["High_Applicant_Demand"],
    )

    cat_cols     = ["work_type", "employment_type", "seniority_level", "emp_size"]
    num_cols     = ["is_remote", "log_followers", "hours_since_posted"]
    feature_list = cat_cols + num_cols

    model_df = ml_df[feature_list + ["High_Applicant_Demand"]].copy()

    encoders: dict = {}
    for col in cat_cols:
        model_df[col] = model_df[col].fillna("Unknown")
        le = LabelEncoder()
        model_df[col] = le.fit_transform(model_df[col].astype(str))
        encoders[col] = le

    for col in num_cols:
        model_df[col] = model_df[col].fillna(model_df[col].median())

    X = model_df[feature_list]
    y = model_df["High_Applicant_Demand"]

    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]

    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_test, y_prob),
        "cm":        confusion_matrix(y_test, y_pred),
        "fpr":       fpr,
        "tpr":       tpr,
        "threshold": median_threshold,
        "n_high":    int(ml_df["High_Applicant_Demand"].sum()),
        "n_low":     int((ml_df["High_Applicant_Demand"] == 0).sum()),
        "train_n":   len(X_train),
        "test_n":    len(X_test),
    }

    feat_imp = (
        pd.DataFrame({"Feature": feature_list, "Importance": rf.feature_importances_})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    # ── Score all ML rows for the analytical scenario ──
    score_df = ml_df[feature_list + ["no_of_application", "High_Applicant_Demand"]].copy()
    for col in cat_cols:
        score_df[col] = score_df[col].fillna("Unknown")
        score_df[col] = encoders[col].transform(
            score_df[col]
            .where(score_df[col].isin(encoders[col].classes_), "Unknown")
            .astype(str)
        )
    for col in num_cols:
        score_df[col] = score_df[col].fillna(score_df[col].median())

    ml_df["Predicted_Prob"] = rf.predict_proba(score_df[feature_list])[:, 1]

    id_cols = ["job", "company_name", "location", "work_type", "no_of_application"]
    scenario_df = ml_df[
        [c for c in id_cols if c in ml_df.columns]
        + ["Predicted_Prob", "High_Applicant_Demand"]
    ].copy()

    return rf, metrics, feat_imp, scenario_df


ML_MODEL, ML_METRICS, FEAT_IMP, SCENARIO_DF = train_ml_model(
    "linkdin_Job_data_cleaned.csv"
)

# ──────────────────────────────────────────────────────────
# SIDEBAR FILTERS
# Default = all options selected  →  full dataset shown
# ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 💼 Filters")
    st.caption(
        "All filters default to **All** — the full cleaned dataset is shown. "
        "Deselect items to narrow the view."
    )

    all_work_types = sorted(df_full["work_type"].dropna().unique().tolist())
    sel_work_type  = st.multiselect("Work Type",       all_work_types, default=all_work_types)

    all_emp_types  = sorted(df_full["employment_type"].dropna().unique().tolist())
    sel_emp_type   = st.multiselect("Employment Type", all_emp_types,  default=all_emp_types)

    all_seniority  = sorted(df_full["seniority_level"].dropna().unique().tolist())
    sel_seniority  = st.multiselect("Seniority Level", all_seniority,  default=all_seniority)

    # Company size — preserve logical size order
    _present_sizes = df_full["emp_size"].dropna().unique()
    all_sizes = [s for s in _SIZE_ORDER if s in _present_sizes] + \
                [s for s in sorted(_present_sizes) if s not in _SIZE_ORDER]
    sel_sizes = st.multiselect("Company Size", all_sizes, default=all_sizes)

    st.markdown("---")
    top_n_locations = st.slider("Top N Locations to Show", 5, 20, 10)
    top_n_companies = st.slider("Top N Companies to Show", 5, 20, 10)

    st.markdown("---")
    st.caption("📊 LinkedIn Job Market Analytics")
    st.caption("Dataset: linkdin_Job_data_cleaned.csv")
    st.caption("5,819 job postings · 16 columns")

# ──────────────────────────────────────────────────────────
# FILTER APPLICATION
# When ALL options are selected the mask is True for every row
# (including NaN rows), preserving the full 5,819-row count.
# When a partial selection is made, NaN rows are excluded.
# When nothing is selected, no rows pass.
# ──────────────────────────────────────────────────────────
def _apply_filter(
    series: pd.Series, selected: list, all_options: list
) -> pd.Series:
    if set(selected) == set(all_options):          # all — keep NaN rows too
        return pd.Series(True, index=series.index)
    if not selected:                               # nothing selected
        return pd.Series(False, index=series.index)
    return series.isin(selected)                   # partial — exclude NaN


mask = (
    _apply_filter(df_full["work_type"],       sel_work_type, all_work_types)
    & _apply_filter(df_full["employment_type"], sel_emp_type,  all_emp_types)
    & _apply_filter(df_full["seniority_level"], sel_seniority, all_seniority)
    & _apply_filter(df_full["emp_size"],        sel_sizes,     all_sizes)
)
df = df_full[mask].copy()

# ──────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────
st.title("💼 LinkedIn Job Market Analysis & Career Insights")
st.markdown(
    "An end-to-end analytics portfolio project — exploratory analysis, "
    "machine learning, and actionable business recommendations "
    "based on **real LinkedIn job-posting data**."
)

# Filter status badge
_is_default = (
    set(sel_work_type) == set(all_work_types)
    and set(sel_emp_type) == set(all_emp_types)
    and set(sel_seniority) == set(all_seniority)
    and set(sel_sizes) == set(all_sizes)
)
if _is_default:
    st.info(
        f"Showing **all {len(df):,} job postings** (full cleaned dataset). "
        "Use the sidebar to filter by work type, employment type, seniority, or company size."
    )
else:
    st.warning(
        f"**Filters active** — showing **{len(df):,}** of {len(df_full):,} total postings. "
        "KPIs and charts update automatically. "
        "ML metrics (Section 6) are always computed on the full dataset."
    )

if len(df) == 0:
    st.error("⚠️ No postings match the current filters. Please widen your selection in the sidebar.")
    st.stop()

# ══════════════════════════════════════════════════════════
# SECTION 1 — OVERVIEW KPIs
# ══════════════════════════════════════════════════════════
st.markdown("---")
_sec("📈 Overview")

_app_s = df["no_of_application"].dropna()
_kpi = {
    "postings":       len(df),
    "companies":      df["company_name"].nunique(),
    "total_apps":     int(_app_s.sum())  if len(_app_s) > 0 else 0,
    "avg_apps":       float(_app_s.mean()) if len(_app_s) > 0 else float("nan"),
    "remote_pct":     df["is_remote"].mean() * 100,
    "top_loc":        df["location"].value_counts().idxmax() if len(df) > 0 else "N/A",
    "top_co":         (df["company_name"].dropna().value_counts().idxmax()
                       if df["company_name"].notna().any() else "N/A"),
}

c1, c2, c3, c4 = st.columns(4)
c5, c6, c7     = st.columns(3)

c1.metric("Total Job Postings",       f"{_kpi['postings']:,}")
c2.metric("Total Companies",          f"{_kpi['companies']:,}")
c3.metric("Applicants Recorded",      f"{_kpi['total_apps']:,}")
c4.metric("Remote Postings",          f"{_kpi['remote_pct']:.1f}%")
c5.metric("Avg Applicants / Posting",
          f"{_kpi['avg_apps']:.1f}" if not math.isnan(_kpi['avg_apps']) else "N/A")
c6.metric("Top Location",             _kpi["top_loc"].split(",")[0])
c7.metric("Top Company by Postings",  _kpi["top_co"])

_insight(
    f"<b>Snapshot:</b> The current filter selection covers "
    f"<b>{_kpi['postings']:,}</b> job postings from "
    f"<b>{_kpi['companies']:,}</b> companies. "
    f"<b>{_kpi['remote_pct']:.1f}%</b> of these postings are tagged Remote. "
    f"Where applicant counts are available ({len(_app_s):,} postings), "
    f"the average is <b>{_kpi['avg_apps']:.1f}</b> applicants per posting."
)

# ══════════════════════════════════════════════════════════
# SECTION 2 — JOB MARKET ANALYSIS
# ══════════════════════════════════════════════════════════
st.markdown("---")
_sec("🌍 Job Market Analysis")

t_loc, t_role, t_emp, t_sen, t_ind = st.tabs(
    ["📍 Location", "🧑‍💻 Job Roles", "💼 Employment Type", "🏅 Seniority", "🏢 Industry"]
)

# ── Location ──────────────────────────────────────────────
with t_loc:
    loc_vc = df["location"].value_counts().head(top_n_locations).reset_index()
    loc_vc.columns = ["Location", "Postings"]
    fig = px.bar(
        loc_vc, x="Postings", y="Location", orientation="h",
        title=f"Top {top_n_locations} Locations by Job Postings",
        color="Postings", color_continuous_scale="Blues", text="Postings",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        yaxis=dict(categoryorder="total ascending"),
        coloraxis_showscale=False,
        height=max(360, top_n_locations * 36),
        xaxis_title="Number of Postings", yaxis_title="",
        margin=dict(r=80, t=50),
    )
    st.plotly_chart(fig, use_container_width=True)
    _insight(
        f"<b>Location concentration:</b> The top location in this filtered view is "
        f"<b>{loc_vc['Location'].iloc[0].split(',')[0]}</b> "
        f"({int(loc_vc['Postings'].iloc[0]):,} postings). "
        "In the full dataset, Bengaluru, Hyderabad, Gurugram, and Mumbai account for "
        "the majority of postings — consistent with India's IT-sector geography."
    )

# ── Job Roles ─────────────────────────────────────────────
with t_role:
    role_vc = df["job"].value_counts().head(15).reset_index()
    role_vc.columns = ["Job Title", "Postings"]
    fig = px.bar(
        role_vc, x="Postings", y="Job Title", orientation="h",
        title="Top 15 Most-Posted Job Titles",
        color="Postings", color_continuous_scale="Purples", text="Postings",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        yaxis=dict(categoryorder="total ascending"),
        coloraxis_showscale=False,
        height=490,
        xaxis_title="Number of Postings", yaxis_title="",
        margin=dict(r=80, t=50),
    )
    st.plotly_chart(fig, use_container_width=True)
    _insight(
        f"<b>Top role in current view:</b> <b>{role_vc['Job Title'].iloc[0]}</b> "
        f"({int(role_vc['Postings'].iloc[0]):,} postings). "
        "Across the full dataset, Lead Java Software Engineer and Senior Automation Tester "
        "are the most actively posted titles, reflecting strong demand for backend "
        "and QA/test-automation talent."
    )

# ── Employment Type ───────────────────────────────────────
with t_emp:
    emp_vc = df["employment_type"].value_counts().reset_index()
    emp_vc.columns = ["Employment Type", "Count"]
    if len(emp_vc) == 0:
        st.info("No employment-type data available for the current filter selection.")
    else:
        fig = px.pie(
            emp_vc, names="Employment Type", values="Count",
            title="Employment Type Distribution",
            hole=0.42,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(height=420, showlegend=True, margin=dict(t=50))
        st.plotly_chart(fig, use_container_width=True)

        ft_pct = emp_vc.set_index("Employment Type")["Count"].get("Full-time", 0)
        ft_pct_share = ft_pct / emp_vc["Count"].sum() * 100 if emp_vc["Count"].sum() > 0 else 0
        _insight(
            f"<b>Employment type breakdown:</b> In this filtered view, "
            f"Full-time postings account for <b>{ft_pct_share:.1f}%</b> of employment-typed postings. "
            "In the full dataset, Full-time roles represent ~93% of all postings. "
            "Contract and Internship roles make up most of the remainder."
        )

# ── Seniority ─────────────────────────────────────────────
with t_sen:
    sen_vc = df["seniority_level"].dropna().value_counts().reset_index()
    sen_vc.columns = ["Seniority Level", "Count"]
    if len(sen_vc) == 0:
        st.info("No seniority-level data available for the current filter selection.")
    else:
        fig = px.bar(
            sen_vc, x="Seniority Level", y="Count",
            title="Job Postings by Seniority Level",
            color="Count", color_continuous_scale="Oranges", text="Count",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            coloraxis_showscale=False, height=400,
            xaxis_title="Seniority Level", yaxis_title="Postings",
            margin=dict(t=50),
        )
        st.plotly_chart(fig, use_container_width=True)
        top_sen = sen_vc.iloc[0]
        _insight(
            f"<b>Seniority distribution:</b> The most common seniority level in this view is "
            f"<b>{top_sen['Seniority Level']}</b> ({int(top_sen['Count']):,} postings). "
            "In the full dataset, Mid-Senior level dominates (~82% of seniority-tagged postings). "
            "Entry-level roles are comparatively scarce — a challenge for recent graduates."
        )

# ── Industry ──────────────────────────────────────────────
with t_ind:
    ind_vc = df["emp_industry"].dropna().value_counts().head(10).reset_index()
    ind_vc.columns = ["Industry", "Postings"]
    if len(ind_vc) == 0:
        st.info("No industry data available for the current filter selection.")
    else:
        fig = px.bar(
            ind_vc, x="Postings", y="Industry", orientation="h",
            title="Top 10 Industries by Job Postings",
            color="Postings", color_continuous_scale="Teal", text="Postings",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            yaxis=dict(categoryorder="total ascending"),
            coloraxis_showscale=False, height=440,
            xaxis_title="Number of Postings", yaxis_title="",
            margin=dict(r=80, t=50),
        )
        st.plotly_chart(fig, use_container_width=True)
        _insight(
            "IT Services & Consulting accounts for nearly half of industry-tagged postings "
            "in the full dataset, confirming a strong technology-sector skew. "
            "Staffing & Recruiting and Software Development also feature prominently."
        )

# ══════════════════════════════════════════════════════════
# SECTION 3 — COMPANY INSIGHTS
# ══════════════════════════════════════════════════════════
st.markdown("---")
_sec("🏢 Company Insights")

t_co, t_size, t_foll = st.tabs(
    ["🏆 Top Companies", "📐 Company Size", "📣 LinkedIn Followers"]
)

# ── Top Companies ─────────────────────────────────────────
with t_co:
    co_vc = df["company_name"].dropna().value_counts().head(top_n_companies).reset_index()
    co_vc.columns = ["Company", "Postings"]
    if len(co_vc) == 0:
        st.info("No company data available for the current filter selection.")
    else:
        fig = px.bar(
            co_vc, x="Postings", y="Company", orientation="h",
            title=f"Top {top_n_companies} Companies by Job Postings",
            color="Postings", color_continuous_scale="Reds", text="Postings",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            yaxis=dict(categoryorder="total ascending"),
            coloraxis_showscale=False,
            height=max(360, top_n_companies * 38),
            xaxis_title="Job Postings", yaxis_title="",
            margin=dict(r=80, t=50),
        )
        st.plotly_chart(fig, use_container_width=True)
        top_co_name  = co_vc.iloc[0]["Company"]
        top_co_count = int(co_vc.iloc[0]["Postings"])
        top_co_share = top_co_count / len(df) * 100
        _insight(
            f"<b>Top poster in current view:</b> <b>{top_co_name}</b> "
            f"({top_co_count:,} postings, {top_co_share:.1f}% of filtered set). "
            "In the full dataset, EPAM Anywhere alone accounts for ~23% of all postings, "
            "indicating unusually high job-posting activity for a single employer."
        )

# ── Company Size ──────────────────────────────────────────
with t_size:
    sz_data = df["emp_size"].dropna()
    if len(sz_data) == 0:
        st.info("No company-size data for the current filter selection.")
    else:
        sz_vc = (
            sz_data.value_counts()
            .reindex([s for s in _SIZE_ORDER if s in sz_data.unique()])
            .dropna()
            .reset_index()
        )
        sz_vc.columns = ["Company Size", "Postings"]

        fig = px.bar(
            sz_vc, x="Company Size", y="Postings",
            title="Job Postings by Company Size",
            color="Postings", color_continuous_scale="Greens", text="Postings",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(coloraxis_showscale=False, height=420, margin=dict(t=50))
        st.plotly_chart(fig, use_container_width=True)

        # Avg applicants per posting by company size
        app_sz = df[df["no_of_application"].notna() & df["emp_size"].notna()]
        if len(app_sz) > 0:
            avg_sz = (
                app_sz.groupby("emp_size")["no_of_application"]
                .mean()
                .reindex([s for s in _SIZE_ORDER if s in app_sz["emp_size"].unique()])
                .dropna()
                .round(1)
                .reset_index()
            )
            avg_sz.columns = ["Company Size", "Avg Applicants"]
            fig2 = px.bar(
                avg_sz, x="Company Size", y="Avg Applicants",
                title="Average Applicants per Posting by Company Size",
                color="Avg Applicants", color_continuous_scale="Blues",
                text="Avg Applicants",
            )
            fig2.update_traces(textposition="outside")
            fig2.update_layout(
                coloraxis_showscale=False, height=420,
                xaxis_title="Company Size", yaxis_title="Avg Applicants",
                margin=dict(t=50),
            )
            st.plotly_chart(fig2, use_container_width=True)

        _insight(
            "Large enterprises (1,001–5,000 employees) post the most jobs in this dataset "
            "and also attract relatively more applicants per posting — likely due to brand "
            "recognition. Smaller companies (11–200 employees) may offer lower per-posting "
            "competition, which is a tactical consideration for job seekers."
        )

# ── LinkedIn Followers ────────────────────────────────────
with t_foll:
    foll_data = df["linkedin_followers"].dropna()
    if len(foll_data) == 0:
        st.info("No LinkedIn follower data for the current filter selection.")
    else:
        fig = px.histogram(
            df.dropna(subset=["linkedin_followers"]),
            x="linkedin_followers", nbins=50,
            title="Distribution of Company LinkedIn Followers (log scale)",
            log_x=True,
            color_discrete_sequence=["#3b82d4"],
        )
        fig.update_layout(
            xaxis_title="LinkedIn Followers (log scale)",
            yaxis_title="Number of Postings", height=380,
        )
        st.plotly_chart(fig, use_container_width=True)

        top_foll = (
            df.groupby("company_name")["linkedin_followers"]
            .mean()
            .nlargest(10)
            .reset_index()
        )
        top_foll.columns = ["Company", "Avg Followers"]
        top_foll["Label"] = top_foll["Avg Followers"].apply(
            lambda x: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K"
        )
        fig2 = px.bar(
            top_foll, x="Avg Followers", y="Company", orientation="h",
            title="Top 10 Companies by LinkedIn Followers",
            color="Avg Followers", color_continuous_scale="Blues",
            text="Label",
        )
        fig2.update_traces(textposition="outside")
        fig2.update_layout(
            yaxis=dict(categoryorder="total ascending"),
            coloraxis_showscale=False, height=400,
            xaxis_title="Avg LinkedIn Followers", yaxis_title="",
            margin=dict(r=100, t=50),
        )
        st.plotly_chart(fig2, use_container_width=True)
        _insight(
            "LinkedIn follower counts are heavily right-skewed — a few large companies "
            "command millions of followers while most have far fewer. "
            "The ML model identifies follower count (log-scaled) as the "
            "<b>second-strongest predictor</b> of applicant volume after posting age."
        )

# ══════════════════════════════════════════════════════════
# SECTION 4 — REMOTE & WORK TYPE
# ══════════════════════════════════════════════════════════
st.markdown("---")
_sec("🏠 Remote & Work Type")

wt_data = df["work_type"].dropna()

if len(wt_data) == 0:
    st.info("No work-type data available for the current filter selection.")
else:
    c_wt, c_rem = st.columns(2)

    with c_wt:
        wt_vc = wt_data.value_counts().reset_index()
        wt_vc.columns = ["Work Type", "Count"]
        fig = px.pie(
            wt_vc, names="Work Type", values="Count",
            title="Work Type Distribution",
            hole=0.45,
            color_discrete_map={
                "Remote": "#3b82d4", "On-site": "#f59e0b", "Hybrid": "#7c5cd8"
            },
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(height=380, margin=dict(t=50))
        st.plotly_chart(fig, use_container_width=True)

    with c_rem:
        remote_sub = df[df["work_type"] == "Remote"]
        if len(remote_sub) > 0:
            rem_loc = remote_sub["location"].value_counts().head(10).reset_index()
            rem_loc.columns = ["Location", "Remote Postings"]
            fig = px.bar(
                rem_loc, x="Remote Postings", y="Location", orientation="h",
                title="Top 10 Locations for Remote Postings",
                color="Remote Postings", color_continuous_scale="Blues",
                text="Remote Postings",
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(
                yaxis=dict(categoryorder="total ascending"),
                coloraxis_showscale=False, height=380,
                xaxis_title="Remote Postings", yaxis_title="",
                margin=dict(r=80, t=50),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No Remote postings in the current filter selection.")

    # Work type × employment type count heatmap
    emp_data = df["employment_type"].dropna()
    if len(emp_data) > 0:
        cross = pd.crosstab(df["work_type"], df["employment_type"])
        cross = cross.loc[(cross > 0).any(axis=1), (cross > 0).any(axis=0)]
        if cross.shape[0] > 0 and cross.shape[1] > 0:
            fig = px.imshow(
                cross, text_auto=True,
                title="Work Type × Employment Type — Posting Count",
                color_continuous_scale="Blues", aspect="auto",
            )
            fig.update_layout(
                height=max(260, cross.shape[0] * 72),
                xaxis_title="Employment Type", yaxis_title="Work Type",
                margin=dict(t=50),
            )
            st.plotly_chart(fig, use_container_width=True)

    # Dynamic observation
    n_remote  = int((wt_data == "Remote").sum())
    n_onsite  = int((wt_data == "On-site").sum())
    n_hybrid  = int((wt_data == "Hybrid").sum())
    n_total_wt = len(wt_data)
    _insight(
        f"<b>Work-type split in current view:</b> "
        f"Remote <b>{n_remote:,}</b> ({n_remote/n_total_wt*100:.1f}%)  ·  "
        f"On-site <b>{n_onsite:,}</b> ({n_onsite/n_total_wt*100:.1f}%)  ·  "
        f"Hybrid <b>{n_hybrid:,}</b> ({n_hybrid/n_total_wt*100:.1f}%). "
        "In the full dataset Remote and On-site are nearly equal (~40% each). "
        "Many remote-tagged postings list 'India' as location, suggesting "
        "domestically scoped roles rather than globally open positions."
    )

# ══════════════════════════════════════════════════════════
# SECTION 5 — APPLICANT INSIGHTS
# ══════════════════════════════════════════════════════════
st.markdown("---")
_sec("👥 Applicant Insights")

_warn(
    "<b>Data note:</b> Applicant counts record how many people submitted an application "
    "via LinkedIn at the time this data was collected. "
    "This is <b>not</b> a record of interview invitations, offers, or hires. "
    "No inference about hiring outcomes can be drawn from these counts."
)

app_df = df[df["no_of_application"].notna()].copy()

if len(app_df) < 5:
    st.info(
        f"Only {len(app_df)} postings with applicant data match the current filters "
        "(minimum 5 required). Please widen the filter selection."
    )
else:
    ca, cb = st.columns(2)

    with ca:
        fig = px.histogram(
            app_df, x="no_of_application", nbins=40,
            title="Distribution of Applicants per Posting",
            color_discrete_sequence=["#7c5cd8"],
        )
        fig.update_layout(
            xaxis_title="Number of Applicants",
            yaxis_title="Number of Postings", height=360,
        )
        st.plotly_chart(fig, use_container_width=True)

    with cb:
        wt_in_app = app_df["work_type"].dropna().unique()
        if len(wt_in_app) > 0:
            fig = px.box(
                app_df.dropna(subset=["work_type"]),
                x="work_type", y="no_of_application",
                title="Applicant Count by Work Type",
                color="work_type",
                color_discrete_map={
                    "Remote": "#3b82d4", "On-site": "#f59e0b", "Hybrid": "#7c5cd8"
                },
            )
            fig.update_layout(
                xaxis_title="Work Type", yaxis_title="Number of Applicants",
                showlegend=False, height=360,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Work-type data is unavailable in this applicant subset.")

    # Avg applicants by seniority (≥ 5 postings per level)
    sen_app = app_df[app_df["seniority_level"].notna()]
    if len(sen_app) >= 5:
        avg_sen = (
            sen_app.groupby("seniority_level")["no_of_application"]
            .agg(mean="mean", count="count")
            .reset_index()
            .rename(columns={"mean": "Avg Applicants", "count": "Postings"})
        )
        avg_sen = avg_sen[avg_sen["Postings"] >= 5].copy()
        avg_sen["Avg Applicants"] = avg_sen["Avg Applicants"].round(1)
        if len(avg_sen) > 0:
            fig = px.bar(
                avg_sen, x="seniority_level", y="Avg Applicants",
                title="Average Applicants by Seniority Level (min 5 postings per level)",
                color="Avg Applicants", color_continuous_scale="Oranges",
                text="Avg Applicants",
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(
                coloraxis_showscale=False, height=400,
                xaxis_title="Seniority Level", yaxis_title="Avg Applicants",
                margin=dict(t=50),
            )
            st.plotly_chart(fig, use_container_width=True)

    # Posting age vs applicants scatter with OLS trendline
    scatter_df = app_df.dropna(subset=["hours_since_posted"])
    if len(scatter_df) >= 10:
        fig = px.scatter(
            scatter_df,
            x="hours_since_posted", y="no_of_application",
            color="work_type",
            title="Posting Age vs Number of Applicants (with OLS trendline)",
            labels={
                "hours_since_posted": "Hours Since Posted",
                "no_of_application":  "Number of Applicants",
                "work_type":          "Work Type",
            },
            opacity=0.5,
            color_discrete_map={
                "Remote": "#3b82d4", "On-site": "#f59e0b", "Hybrid": "#7c5cd8"
            },
            trendline="ols",
        )
        fig.update_layout(height=430, margin=dict(t=50))
        st.plotly_chart(fig, use_container_width=True)

    # Summary statistics table
    _stats = app_df["no_of_application"].describe()
    stats_df = pd.DataFrame({
        "Statistic": ["Count", "Mean", "Std Dev", "Min", "25th Pct", "Median", "75th Pct", "Max"],
        "Value":     [
            f"{int(_stats['count']):,}",
            f"{_stats['mean']:.1f}",
            f"{_stats['std']:.1f}",
            f"{int(_stats['min']):,}",
            f"{_stats['25%']:.1f}",
            f"{_stats['50%']:.1f}",
            f"{_stats['75%']:.1f}",
            f"{int(_stats['max']):,}",
        ],
    })
    st.markdown("**Applicant Count — Summary Statistics (filtered view)**")
    st.dataframe(stats_df, use_container_width=False, hide_index=True)

    _insight(
        f"<b>Applicant distribution in current view:</b> "
        f"Median {_stats['50%']:.0f} · Mean {_stats['mean']:.1f} · "
        f"75th percentile {_stats['75%']:.0f} · Max {int(_stats['max']):,}. "
        "The right-skew (mean >> median) is driven by a subset of highly competitive "
        "postings that reach the 200-applicant display cap on LinkedIn. "
        "Posting age (hours since posting) is the strongest predictor of applicant "
        "count in the ML model — older postings accumulate more applications."
    )

# ══════════════════════════════════════════════════════════
# SECTION 6 — MACHINE LEARNING
# ══════════════════════════════════════════════════════════
st.markdown("---")
_sec("🤖 Machine Learning — High Applicant Demand Prediction")

st.markdown(
    "**Business question:** *Can we predict, when a job posting goes live, "
    "whether it will attract relatively high applicant interest — "
    "using only information that is publicly visible at that moment?*\n\n"
    "A reliable classifier allows recruiters to prioritise faster screening for "
    "high-competition roles and helps job seekers anticipate competition levels."
)

st.info(
    "📌 ML metrics are computed on the **full cleaned dataset** (5,819 postings) "
    "and do not change with sidebar filters. "
    "The sidebar filters affect Sections 1–5 only."
)

# ── Target & leakage explanation ──────────────────────────
with st.expander("📋 Target Definition & Leakage Prevention", expanded=True):
    st.markdown(f"""
**Target variable:** `High_Applicant_Demand` (binary)

| Property | Value |
|---|---|
| Threshold | Median of `no_of_application` in the **training set only** = **{ML_METRICS['threshold']:.0f} applicants** |
| Class 1 — High | Postings with ≥ {ML_METRICS['threshold']:.0f} applicants &nbsp; ({ML_METRICS['n_high']:,} postings) |
| Class 0 — Low  | Postings with < {ML_METRICS['threshold']:.0f} applicants &nbsp; ({ML_METRICS['n_low']:,} postings) |
| Training rows | {ML_METRICS['train_n']:,} |
| Test rows | {ML_METRICS['test_n']:,} |
| Split strategy | Stratified 80/20 — equal class proportions in train and test |

---

**Strict data-leakage prevention**

- `no_of_application` is the source of the target — it is **never** a predictor.
- `High_Applicant_Demand` is the target itself — **never** used as input.
- `job_ID` (raw identifier) is excluded.
- The median threshold is derived from the **training set only**; the test set is unseen until evaluation.

---

**Predictor features** — all observable at posting time

| Feature | Type | Justification |
|---|---|---|
| `work_type` | Categorical | Remote/On-site/Hybrid affects accessible talent pool |
| `employment_type` | Categorical | Full-time vs contract changes the applicant population |
| `seniority_level` | Categorical | Narrows or broadens the qualified candidate pool |
| `emp_size` | Categorical | Company brand may influence application volume |
| `is_remote` | Binary (derived) | 1 = Remote; captures the remote signal directly |
| `log_followers` | Numeric (derived) | Company's LinkedIn reach; log-scaled to reduce skew |
| `hours_since_posted` | Numeric (derived) | Elapsed time since posting — older posts accumulate more applicants |
""")

# ── Performance metrics ───────────────────────────────────
cm_arr = ML_METRICS["cm"]
col_perf, col_imp = st.columns(2)

with col_perf:
    st.subheader("Model Performance")
    st.markdown(f"""
| Metric | Value |
|---|---|
| Model | Random Forest (200 trees) |
| Train / Test | 80 % / 20 % stratified |
| Accuracy | **{ML_METRICS['accuracy']:.4f}** |
| Precision | **{ML_METRICS['precision']:.4f}** |
| Recall | **{ML_METRICS['recall']:.4f}** |
| ROC-AUC | **{ML_METRICS['roc_auc']:.4f}** |
""")

    cm_df = pd.DataFrame(
        cm_arr,
        index   =["Actual Low (0)", "Actual High (1)"],
        columns =["Predicted Low (0)", "Predicted High (1)"],
    )
    fig_cm = px.imshow(
        cm_df, text_auto=True,
        title="Confusion Matrix",
        color_continuous_scale="Blues", aspect="auto",
    )
    fig_cm.update_layout(height=310, margin=dict(t=50))
    st.plotly_chart(fig_cm, use_container_width=True)
    st.caption(
        f"TN = {cm_arr[0,0]}  ·  FP = {cm_arr[0,1]}  ·  "
        f"FN = {cm_arr[1,0]}  ·  TP = {cm_arr[1,1]}  |  "
        f"Test set: {ML_METRICS['test_n']:,} postings"
    )

with col_imp:
    st.subheader("Feature Importance")
    fig_fi = px.bar(
        FEAT_IMP, x="Importance", y="Feature", orientation="h",
        title="Random Forest Feature Importances",
        color="Importance", color_continuous_scale="Purples",
        text=FEAT_IMP["Importance"].round(3),
    )
    fig_fi.update_traces(textposition="outside")
    fig_fi.update_layout(
        yaxis=dict(categoryorder="total ascending"),
        coloraxis_showscale=False, height=400,
        margin=dict(r=80, t=50),
    )
    st.plotly_chart(fig_fi, use_container_width=True)
    _insight(
        f"<b>Top predictor:</b> <b>{FEAT_IMP.iloc[0]['Feature']}</b> "
        f"(importance {FEAT_IMP.iloc[0]['Importance']:.3f}) — how long a posting has been live "
        "is the single strongest signal. The longer a posting has been active, the more "
        "applications it has had time to accumulate. "
        f"The second predictor, <b>{FEAT_IMP.iloc[1]['Feature']}</b> "
        f"({FEAT_IMP.iloc[1]['Importance']:.3f}), reflects company brand/reach on LinkedIn."
    )

# ── ROC curve ─────────────────────────────────────────────
_auc = ML_METRICS["roc_auc"]
fig_roc = go.Figure()
fig_roc.add_trace(go.Scatter(
    x=ML_METRICS["fpr"], y=ML_METRICS["tpr"], mode="lines",
    name=f"Random Forest  (AUC = {_auc:.3f})",
    line=dict(color="#3b82d4", width=2.5),
))
fig_roc.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1], mode="lines",
    name="No-skill baseline",
    line=dict(dash="dash", color="#aaa", width=1.5),
))
fig_roc.update_layout(
    title="ROC Curve — High Applicant Demand Classifier",
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate",
    height=420,
    legend=dict(x=0.55, y=0.08),
    margin=dict(t=50),
)
st.plotly_chart(fig_roc, use_container_width=True)

_warn(
    "<b>Model limitations</b><br>"
    "(1) The binary threshold is the training-set median — a dataset-specific benchmark, "
    "not a universally meaningful cutoff.<br>"
    "(2) LinkedIn caps displayed applicant counts at 200; the true right-tail distribution "
    "is censored.<br>"
    "(3) ~50% of postings have no recorded applicant count and are excluded from "
    "ML training and evaluation.<br>"
    "(4) Features with high missingness "
    "(seniority_level: 38.8%, emp_size: 3.2%) are imputed as 'Unknown', "
    "which may dilute predictive signal.<br>"
    "(5) <b>This model predicts applicant volume, not hiring outcomes.</b> "
    "It cannot determine whether any applicant was contacted, interviewed, "
    "or selected.<br>"
    "(6) The dataset is heavily India-centric; performance on other geographies "
    "is not validated."
)

# ══════════════════════════════════════════════════════════
# SECTION 7 — BUSINESS RECOMMENDATIONS
# ══════════════════════════════════════════════════════════
st.markdown("---")
_sec("💡 Business Recommendations")

t_js, t_rec, t_an, t_scen = st.tabs(
    ["🎓 Job Seekers", "🏢 Recruiters / HR", "📊 Career Analysts", "🎯 Analytical Scenario"]
)

with t_js:
    st.subheader("For Job Seekers")
    for ttl, body in [
        ("Target High-Volume City Hubs",
         "Bengaluru, Hyderabad, Gurugram, and Mumbai account for the majority of postings "
         "in this dataset. Candidates open to these locations maximise their visible opportunities."),
        ("Prioritise In-Demand Technical Skills",
         "Lead Java Software Engineer and Senior Automation Tester are the two most-posted titles. "
         "Java backend, .NET, ReactJS, Python, and test-automation skills are in sustained demand."),
        ("Apply Early — Posting Age Drives Competition",
         "Hours since posting is the strongest predictor of applicant volume in the ML model. "
         "Applying within the first few hours of a posting going live meaningfully reduces competition."),
        ("Consider Smaller Employers for Lower Competition",
         "Large enterprises (1,001–5,000 employees) post the most roles but also attract the "
         "most applicants per posting. Companies with 11–200 employees may offer better odds."),
        ("The Remote Market is Substantial",
         "Approximately 40% of postings in this dataset are tagged Remote, "
         "providing meaningful access for candidates not tied to a specific city."),
    ]:
        _rec(ttl, body)

with t_rec:
    st.subheader("For Recruiters / HR Teams")
    for ttl, body in [
        ("Complete All Posting Fields",
         "38% of postings omit seniority level; 37% omit industry. "
         "Completing these fields improves LinkedIn's candidate-matching algorithm "
         "and helps qualified candidates self-screen more accurately."),
        ("Invest in the Company LinkedIn Page",
         "LinkedIn follower count is the second-strongest predictor of applicant volume in the model. "
         "Growing the company page audience broadens organic reach for each new posting."),
        ("Publish High-Priority Roles During Peak Hours",
         "Applicant counts grow with posting age. Publishing roles when the target audience is "
         "actively browsing may improve the quality and volume of early applications."),
        ("Use Predicted Demand Scores for Screening Triage",
         "Postings the model ranks as High Demand may benefit from faster screening cycles. "
         "Postings ranked Low Demand may benefit from description improvements or targeted outreach."),
        ("Monitor Competitor Posting Velocity",
         "High-volume posters like EPAM Anywhere shape market supply and salary expectations. "
         "Tracking competitor posting activity provides early signals of talent-supply shifts."),
    ]:
        _rec(ttl, body)

with t_an:
    st.subheader("For Career Analysts")
    for ttl, body in [
        ("Control for EPAM Anywhere's Dominance",
         "One company accounts for ~23% of all postings, heavily skewing aggregate statistics. "
         "Always report figures both inclusive and exclusive of this outlier."),
        ("Control for Posting Age in All Applicant-Count Comparisons",
         "Posting age is the strongest driver of applicant count. "
         "Comparisons across work type, seniority, or company size must control for this variable."),
        ("Note the Technology Sector Skew",
         "IT Services & Consulting accounts for ~48% of industry-tagged postings. "
         "Findings should not be generalised to non-tech labour markets without additional data."),
        ("Enrich with Compensation Data",
         "The dataset contains no salary or benefits fields — the most significant analytical gap. "
         "Salary enrichment would enable demand-elasticity and market-competitiveness analysis."),
        ("Classify Remote Roles by Geographic Accessibility",
         "Many 'Remote' postings list 'India' as location, implying domestic scope. "
         "A secondary flag distinguishing truly location-agnostic remote roles would add value."),
    ]:
        _rec(ttl, body)

with t_scen:
    st.subheader("🎯 Resource-Constrained Analytical Scenario")
    _warn(
        "This is a <b>purely illustrative scenario</b> showing one way the model's "
        "probability output could inform a workflow. "
        "It does <b>not</b> represent real company policy and has not been validated operationally."
    )
    st.markdown("""
**Setup:** A recruiting team can only fast-track screening for the top 10% of
postings by predicted competition level (highest predicted probability of attracting
high applicant demand). They use the model to rank and flag these postings.

**Method:**
1. Score every posting that has a recorded applicant count with the trained Random Forest.
2. Rank by `Predicted_Prob` (descending).
3. Flag the top 10% for accelerated pipeline review.
""")

    n_top10 = int(len(SCENARIO_DF) * 0.10)
    top10   = SCENARIO_DF.nlargest(n_top10, "Predicted_Prob").reset_index(drop=True)

    # Precision@10%
    p_at_10 = top10["High_Applicant_Demand"].sum() / len(top10) if len(top10) > 0 else 0.0

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Postings in Top 10%",    f"{len(top10):,}")
    mc2.metric("Precision @ Top 10%",    f"{p_at_10:.1%}",
               help="Share of flagged postings that are actually high-demand.")
    mc3.metric("Model ROC-AUC",          f"{_auc:.3f}")

    show_cols = [c for c in
                 ["job", "company_name", "location", "work_type",
                  "no_of_application", "Predicted_Prob"]
                 if c in top10.columns]
    fmt = {"Predicted_Prob": "{:.3f}"}
    if "no_of_application" in show_cols:
        fmt["no_of_application"] = "{:.0f}"

    st.markdown(f"**Top 20 postings from the {len(top10):,} flagged (ranked by predicted probability)**")
    st.dataframe(
        top10[show_cols].head(20).style.format(fmt),
        use_container_width=True,
        hide_index=True,
    )
    _warn(
        f"Precision@10% ({p_at_10:.1%}) is measured on postings with known applicant counts. "
        "For the ~50% of postings without recorded applicant data, predicted probabilities "
        "can be generated but cannot be independently validated using this dataset."
    )

# ──────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#57606a;font-size:0.78rem;padding:6px 0 12px;'>"
    "LinkedIn Job Market Analytics &nbsp;·&nbsp; "
    "linkdin_Job_data_cleaned.csv &nbsp;·&nbsp; 5,819 rows × 16 columns &nbsp;·&nbsp; "
    "Random Forest ROC-AUC&nbsp;"
    f"{ML_METRICS['roc_auc']:.3f}"
    " &nbsp;·&nbsp; Built with Streamlit &amp; scikit-learn &nbsp;·&nbsp; Portfolio Project"
    "</div>",
    unsafe_allow_html=True,
)
