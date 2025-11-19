# enterprise_dashboard_v3_clean.py
# Cleaned single-file Enterprise AI Analytics (v3_clean)
# Removed: Auth, Forecasting, AutoML, AI Agent, Semantic Search, Notebook Mode,
# Feature Studio, Deployment pages.
# Kept: Home, Data Explorer, Cleaning & Prep, Auto-EDA, Visualizations, Drill-Down,
# Outliers, AI Formula Builder, AI Data Cleaner, Live Connectors, Designer, SQL Engine,
# Export Center, Data Quality, TS Anomaly, Streaming (Simulated), Autosave & Versioning.
#
# WARNING: AI code execution must be explicitly enabled (two toggles). Do not enable on public servers.
# BEFORE RUNNING:
# - Install dependencies from your requirements.txt
# - Set GROQ_API_KEY in environment or .streamlit/secrets.toml if you want Groq features.

import os
import io
import json
import time
import sqlite3
import base64
import joblib
import traceback
from datetime import datetime
from urllib.parse import urlparse

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ML utilities used by kept pages
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Optional/third-party libs (graceful)
try:
    from ydata_profiling import ProfileReport
    YDATA_AVAILABLE = True
except Exception:
    YDATA_AVAILABLE = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except Exception:
    GROQ_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

try:
    from pptx import Presentation
    from pptx.util import Inches
    PPTX_AVAILABLE = True
except Exception:
    PPTX_AVAILABLE = False

# Optional time-series change detection
try:
    import ruptures as rpt
    RUPTURES_AVAILABLE = True
except Exception:
    RUPTURES_AVAILABLE = False

# UI styling (Power BI style)
PRIMARY = "#F0C419"   # Power BI yellow
BG = "#07111a"
CARD = "#0d2b36"
TEXT = "#E6EEF2"

st.set_page_config(page_title="Enterprise AI Analytics (clean)", layout="wide", initial_sidebar_state="expanded")
st.markdown(f"""
    <style>
        .reportview-container {{ background-color: {BG}; color: {TEXT}; }}
        .sidebar .sidebar-content {{ background-color: #06202a; }}
        .stButton>button {{ background-color: {PRIMARY}; color: black; border: none; }}
        .stDownloadButton>button {{ background-color: {PRIMARY}; color: black; border: none; }}
        .card {{ background: {CARD}; padding: 14px; border-radius:10px; box-shadow: 0 3px 8px rgba(0,0,0,0.45); }}
        .metric-label {{ color: #b8c2c7; font-size: 13px; }}
    </style>
""", unsafe_allow_html=True)

st.title("📊 Enterprise AI Analytics — Clean v3")

# ---------------------------
# Groq initialization (optional)
# ---------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    try:
        GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    except Exception:
        GROQ_API_KEY = None

groq_client = None
if GROQ_API_KEY and GROQ_AVAILABLE:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
    except Exception:
        groq_client = None

def groq_chat(prompt: str, model: str = "llama-3.1-70b-versatile", max_tokens: int = 800) -> str:
    """Call Groq chat; returns text or error string."""
    if groq_client is None:
        return "Groq client not available. Set GROQ_API_KEY and install 'groq' to enable AI."
    try:
        resp = groq_client.chat.completions.create(
            model=model,
            messages=[{"role":"user","content":prompt}],
            max_tokens=max_tokens
        )
        # adapt to typical response structure
        try:
            return resp.choices[0].message["content"]
        except Exception:
            return str(resp)
    except Exception as e:
        return f"Groq Error: {e}"

# ---------------------------
# Sidebar navigation + toggles (KEEP toggles)
# ---------------------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/4/4a/Microsoft_Power_BI_logo.svg", width=120)
st.sidebar.markdown("## Navigation")
page = st.sidebar.radio("", [
    "Home", "Data Explorer", "Cleaning & Prep", "Auto-EDA", "Visualizations", "Drill-Down",
    "Outliers", "AI Formula Builder", "AI Data Cleaner", "Live Connectors", "Designer",
    "SQL Engine", "Export Center", "Data Quality", "TS Anomaly", "Streaming (Simulated)",
    "Autosave & Versioning"
])

st.sidebar.markdown("---")
st.sidebar.markdown("### Global settings (kept)")
ENABLE_EDA = st.sidebar.checkbox("Enable Auto-EDA (ydata_profiling)", value=False)
ENABLE_AI = st.sidebar.checkbox("Enable AI (Groq)", value=bool(groq_client))
SAFE_EXEC = st.sidebar.checkbox("Allow AI-suggested code execution (dangerous)", value=False)
CONFIRM_EXEC = st.sidebar.checkbox("Confirm execution (required to run AI code)", value=False)

# session-state init for kept features
if "df" not in st.session_state: st.session_state.df = None
if "cleaned_df" not in st.session_state: st.session_state.cleaned_df = None
if "sqlite_conn" not in st.session_state: st.session_state.sqlite_conn = None
if "snapshots" not in st.session_state: st.session_state.snapshots = []

# ---------------------------
# Helper utilities
# ---------------------------
def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    towrite = io.BytesIO()
    with pd.ExcelWriter(towrite, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False)
    towrite.seek(0)
    return towrite.read()

def download_bytes(data: bytes, filename: str, mime: str):
    st.download_button(f"Download {filename}", data=data, file_name=filename, mime=mime)

def save_snapshot(df: pd.DataFrame) -> str:
    os.makedirs("snapshots", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"snapshots/data_{ts}.parquet"
    df.to_parquet(path)
    st.session_state.snapshots.append(path)
    return path

# ---------------------------
# Pages (kept)
# ---------------------------

# Home
if page == "Home":
    st.header("🏠 Welcome")
    st.markdown("""
    Clean edition — removed Authentication, Forecasting, AutoML, AI Agent, Semantic Search,
    Notebook Mode, Feature Studio, and Deployment pages as requested.
    """)
    st.markdown("### Quick status")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("<div class='card'><b>Data</b><br>", unsafe_allow_html=True)
        if st.session_state.df is None:
            st.markdown("<span class='metric-label'>No dataset loaded</span></div>", unsafe_allow_html=True)
        else:
            d = st.session_state.df
            st.markdown(f"<span class='metric-label'>{d.shape[0]:,} rows • {d.shape[1]:,} cols</span></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='card'><b>AI</b><br>", unsafe_allow_html=True)
        st.markdown(f"<span class='metric-label'>{'Groq configured' if groq_client else 'Groq not configured'}</span></div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='card'><b>Snapshots</b><br>", unsafe_allow_html=True)
        st.markdown(f"<span class='metric-label'>{len(st.session_state.snapshots)} saved</span></div>", unsafe_allow_html=True)

# Data Explorer
elif page == "Data Explorer":
    st.header("📁 Data Explorer")
    st.markdown("Upload CSV/XLSX, load sample, or load from URL.")
    src = st.radio("Source", ["Sample","Upload","URL"], horizontal=True)
    if src == "Sample":
        dates = pd.date_range(end=pd.Timestamp.today(), periods=180)
        sample = pd.DataFrame({
            "date": dates,
            "region": np.random.choice(["North","South","East","West"], len(dates)),
            "product": np.random.choice(["A","B","C"], len(dates)),
            "units_sold": np.random.poisson(20, len(dates)),
            "price": np.random.uniform(10,100,len(dates))
        })
        sample["revenue"] = sample["units_sold"] * sample["price"]
        st.session_state.df = sample
        st.success("Sample dataset loaded.")
    elif src == "Upload":
        uploaded = st.file_uploader("Upload CSV/XLSX", type=["csv","xlsx"])
        if uploaded:
            try:
                if uploaded.name.lower().endswith(".csv"):
                    df = pd.read_csv(uploaded)
                else:
                    df = pd.read_excel(uploaded)
                st.session_state.df = df
                st.success("File loaded.")
            except Exception as e:
                st.error(f"Failed to read file: {e}")
    else:
        url = st.text_input("Enter CSV/JSON/XLSX URL")
        if st.button("Load URL"):
            try:
                r = __import__('requests').get(url); r.raise_for_status()
                ext = urlparse(url).path.split('.')[-1].lower()
                if ext in ('csv','txt'):
                    df = pd.read_csv(io.StringIO(r.text))
                elif ext == 'json':
                    df = pd.json_normalize(r.json())
                elif ext in ('xlsx','xls'):
                    df = pd.read_excel(io.BytesIO(r.content))
                else:
                    df = pd.json_normalize(r.json())
                st.session_state.df = df
                st.success("Loaded from URL.")
            except Exception as e:
                st.error(f"URL load failed: {e}")

    df = st.session_state.df
    if df is not None:
        st.subheader("Preview")
        st.dataframe(df.head(200))
        if st.button("Show dtypes"):
            st.write(df.dtypes)
        if st.button("Save snapshot"):
            p = save_snapshot(df)
            st.success(f"Saved snapshot: {p}")

# Cleaning & Prep
elif page == "Cleaning & Prep":
    st.header("🧹 Cleaning & Preparation")
    df = st.session_state.df
    if df is None:
        st.info("Load dataset first.")
    else:
        left, right = st.columns([2,1])
        with left:
            if st.button("Drop duplicates"):
                df = df.drop_duplicates().reset_index(drop=True); st.session_state.df = df; st.success("Dropped duplicates")
            if st.button("Drop all-empty rows"):
                df = df.dropna(how="all"); st.session_state.df = df; st.success("Dropped empty rows")
            fill = st.selectbox("Fill missing values", ["--","mean (numeric)","median (numeric)","mode (cat)","0","ffill"])
            if st.button("Apply fill"):
                if fill != "--":
                    for c in df.columns:
                        if df[c].isnull().any():
                            if fill == "mean (numeric)" and pd.api.types.is_numeric_dtype(df[c]): df[c].fillna(df[c].mean(), inplace=True)
                            elif fill == "median (numeric)" and pd.api.types.is_numeric_dtype(df[c]): df[c].fillna(df[c].median(), inplace=True)
                            elif fill == "mode (cat)": df[c].fillna(df[c].mode().iloc[0] if not df[c].mode().empty else "", inplace=True)
                            elif fill == "0": df[c].fillna(0, inplace=True)
                            elif fill == "ffill": df[c].fillna(method="ffill", inplace=True)
                    st.session_state.df = df; st.success("Filled missing values")
            if st.button("Save cleaned copy to session"):
                st.session_state.cleaned_df = df.copy(); st.success("Saved cleaned copy")
        with right:
            st.subheader("Column operations")
            col = st.selectbox("Select column", df.columns.tolist())
            if col:
                st.write(df[col].describe(include="all"))
                if st.button("Drop column"):
                    df = df.drop(columns=[col]); st.session_state.df = df; st.success(f"Dropped {col}")

# Auto-EDA
elif page == "Auto-EDA":
    st.header("📊 Auto-EDA / Profiling")
    df = st.session_state.df
    if df is None:
        st.info("Load dataset first.")
    else:
        if ENABLE_EDA and YDATA_AVAILABLE:
            with st.spinner("Running ydata-profiling..."):
                profile = ProfileReport(df, minimal=True)
                html = profile.to_html()
                st.components.v1.html(html, height=700, scrolling=True)
                b64 = base64.b64encode(html.encode()).decode()
                st.markdown(f'<a href="data:text/html;base64,{b64}" download="eda_report.html">📥 Download EDA (HTML)</a>', unsafe_allow_html=True)
        else:
            st.write(df.describe(include="all"))
            st.write("Missing values:")
            st.write(df.isnull().sum())

# Visualizations
elif page == "Visualizations":
    st.header("📈 Visualizations")
    df = st.session_state.df
    if df is None:
        st.info("Load dataset first.")
    else:
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical = df.select_dtypes(include=["object","category"]).columns.tolist()
        st.subheader("Single column charts")
        col = st.selectbox("Numeric column", numeric) if numeric else None
        if col:
            fig = px.line(df.reset_index(), x=df.index, y=col, title=f"{col} Trend", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
            st.plotly_chart(px.histogram(df, x=col, nbins=30, title=f"{col} distribution"), use_container_width=True)
        st.subheader("Grouped charts")
        if categorical and col:
            group = st.selectbox("Group by", ["--"] + categorical)
            if group and group != "--":
                xaxis = "date" if "date" in df.columns else df.index
                fig = px.line(df, x=xaxis, y=col, color=group, title=f"{col} by {group}")
                st.plotly_chart(fig, use_container_width=True)
        if numeric:
            corr = df[numeric].corr()
            st.plotly_chart(px.imshow(corr, text_auto=True, title="Correlation Heatmap"), use_container_width=True)

# Drill-Down
elif page == "Drill-Down":
    st.header("📊 Drill-Down Dashboard")
    df = st.session_state.df
    if df is None:
        st.info("Load dataset first.")
    else:
        cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()
        if not cat_cols:
            st.warning("No categorical columns for drill-down.")
        else:
            base = st.selectbox("Base dimension", cat_cols)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                st.warning("No numeric columns to aggregate.")
            else:
                metric = st.selectbox("Metric", numeric_cols)
                fig = px.bar(df.groupby(base)[metric].sum().reset_index(), x=base, y=metric, title=f"{metric} by {base}")
                st.plotly_chart(fig, use_container_width=True)
                sel_val = st.selectbox("Filter by value (for drill)", ["--"] + sorted(df[base].dropna().unique().tolist()))
                if sel_val and sel_val != "--":
                    filt = df[df[base] == sel_val]
                    xaxis = "date" if "date" in filt.columns else filt.index
                    st.plotly_chart(px.line(filt, x=xaxis, y=metric, title=f"{metric} over time for {sel_val}"), use_container_width=True)

# Outliers
elif page == "Outliers":
    st.header("🔍 Outlier Detection")
    df = st.session_state.df
    if df is None:
        st.info("Load dataset first.")
    else:
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric:
            st.warning("No numeric columns found.")
        else:
            col = st.selectbox("Column for outlier detection", numeric)
            method = st.selectbox("Method", ["IQR","Z-score","IsolationForest"])
            if st.button("Run detection"):
                s = df[col].dropna()
                try:
                    if method == "IQR":
                        Q1,Q3 = s.quantile(0.25), s.quantile(0.75); IQR = Q3-Q1
                        mask = (s < (Q1 - 1.5*IQR)) | (s > (Q3 + 1.5*IQR))
                        out = df.loc[mask.index[mask]]
                    elif method == "Z-score":
                        z = (s - s.mean())/s.std(); mask = z.abs() > 3; out = df.loc[mask.index[mask]]
                    else:
                        iso = IsolationForest(contamination=0.05, random_state=42)
                        preds = iso.fit_predict(s.values.reshape(-1,1))
                        out = df.iloc[np.where(preds==-1)[0]]
                    st.dataframe(out.head(200)); st.success(f"Found {len(out)} outliers")
                except Exception as e:
                    st.error(f"Error: {e}")

# AI Formula Builder (kept)
elif page == "AI Formula Builder":
    st.header("✏️ AI Formula Builder")
    st.markdown("Type formulas like: `profit_margin = (revenue - cost) / revenue`")
    formula = st.text_input("Formula")
    def apply_formula(df_local, formula_str):
        import re
        try:
            left,right = formula_str.split("=",1)
            col = left.strip()
            expr = right.strip()
            tokens = set(re.findall(r"[A-Za-z_]\w*", expr))
            for t in tokens:
                if t in df_local.columns:
                    expr = re.sub(rf"\\b{t}\\b", f"df_local['{t}']", expr)
            df_local[col] = eval(expr)
            return df_local, f"Created column {col}"
        except Exception as e:
            return df_local, f"Error: {e}"
    if st.button("Preview formula"):
        if st.session_state.df is None:
            st.error("Load dataset first.")
        else:
            preview, msg = apply_formula(st.session_state.df.copy(), formula)
            st.write(msg)
            st.dataframe(preview.head())
            if st.button("Apply formula to dataset"):
                st.session_state.df = preview
                st.success("Applied formula to dataset.")

# AI Data Cleaner (kept)
elif page == "AI Data Cleaner":
    st.header("🧹 AI Data Cleaner")
    df = st.session_state.df
    if df is None:
        st.info("Load dataset first.")
    else:
        st.write("Sample of data:")
        st.dataframe(df.head(10))
        if st.button("Ask AI for cleaning suggestions"):
            if not ENABLE_AI or groq_client is None:
                st.error("Enable AI and configure Groq to use this feature.")
            else:
                prompt = "You are a data engineer. Here are sample rows:\n" + df.head(10).to_csv(index=False) + "\nList JSON: actions with {col,issue,suggestion,pandas_code}."
                out = groq_chat(prompt)
                st.write("AI suggestions (raw):")
                st.write(out)
                st.info("Review the suggestions and choose which to apply manually or enable execution (dangerous).")

# Live Connectors (kept)
elif page == "Live Connectors":
    st.header("🌐 Live Connectors")
    st.markdown("Load from REST URL (CSV/JSON/XLSX). Database connectors can be added with credentials.")
    url = st.text_input("Enter data URL (CSV/JSON/XLSX)")
    if st.button("Load URL"):
        try:
            r = __import__('requests').get(url); r.raise_for_status()
            ext = urlparse(url).path.split('.')[-1].lower()
            if ext in ('csv','txt'):
                df = pd.read_csv(io.StringIO(r.text))
            elif ext == 'json':
                df = pd.json_normalize(r.json())
            elif ext in ('xlsx','xls'):
                df = pd.read_excel(io.BytesIO(r.content))
            else:
                df = pd.json_normalize(r.json())
            st.session_state.df = df
            st.success("Loaded data from URL.")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Load failed: {e}")

# Designer (kept)
elif page == "Designer":
    st.header("🎛️ Dashboard Designer (simple)")
    comps = ["kpi","line","bar","table","heatmap"]
    selected = st.multiselect("Pick components to place (max 6)", comps)
    slots = [None]*6
    for i, s in enumerate(selected[:6]):
        slots[i] = s
    if st.button("Save layout"):
        layout = {"slots":slots, "timestamp": datetime.now().isoformat()}
        os.makedirs("layouts", exist_ok=True)
        fname = f"layouts/layout_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(fname,"w",encoding="utf-8") as f: json.dump(layout,f)
        st.success(f"Saved {fname}")
    st.write("Current slots:", slots)
    saved = sorted(os.listdir("layouts")) if os.path.exists("layouts") else []
    st.write("Saved layouts:", saved)

# SQL Engine (kept)
elif page == "SQL Engine":
    st.header("🗄️ SQL Engine (SQLite in-memory)")
    df = st.session_state.df
    if df is None:
        st.info("Load dataset first.")
    else:
        if st.button("Load dataset into SQLite (in-memory)"):
            try:
                conn = sqlite3.connect(":memory:")
                df.to_sql("data_table", conn, index=False, if_exists="replace")
                st.session_state.sqlite_conn = conn
                st.success("Loaded into SQLite as table 'data_table'")
            except Exception as e:
                st.error(f"SQLite load error: {e}")
        if st.session_state.sqlite_conn:
            q = st.text_area("SQL query (run on data_table)", value="SELECT * FROM data_table LIMIT 100")
            if st.button("Execute SQL"):
                try:
                    res = pd.read_sql_query(q, st.session_state.sqlite_conn)
                    st.dataframe(res.head(500))
                except Exception as e:
                    st.error(f"SQL execution error: {e}")

# Export Center (kept; includes AI PDF/PPTX if enabled)
elif page == "Export Center":
    st.header("📤 Export & Reports")
    if st.session_state.cleaned_df is not None:
        st.download_button("Download cleaned CSV", data=st.session_state.cleaned_df.to_csv(index=False).encode(), file_name="cleaned.csv", mime="text/csv")
        st.download_button("Download cleaned Excel", data=df_to_excel_bytes(st.session_state.cleaned_df), file_name="cleaned.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    elif st.session_state.df is not None:
        st.download_button("Download dataset CSV", data=st.session_state.df.to_csv(index=False).encode(), file_name="dataset.csv", mime="text/csv")

    if REPORTLAB_AVAILABLE and st.session_state.df is not None:
        if st.button("Generate AI PDF Summary"):
            prompt = f"Summarize dataset columns: {', '.join(st.session_state.df.columns.tolist())}. Provide top 5 insights and 3 recommendations."
            text = groq_chat(prompt) if ENABLE_AI and groq_client else "AI disabled"
            buf = io.BytesIO(); c = canvas.Canvas(buf, pagesize=letter)
            y = 750; c.setFont("Helvetica-Bold", 14); c.drawString(30,y,"AI Generated Summary"); y -= 30; c.setFont("Helvetica",10)
            for line in text.splitlines():
                if y < 40:
                    c.showPage(); y = 750
                c.drawString(30,y,line[:110]); y -= 12
            c.save(); buf.seek(0)
            st.download_button("Download AI PDF", data=buf, file_name="ai_summary.pdf", mime="application/pdf")
    else:
        if not REPORTLAB_AVAILABLE:
            st.info("reportlab not installed — PDF export disabled.")

    if PPTX_AVAILABLE and st.session_state.df is not None:
        if st.button("Generate PPTX Report"):
            prs = Presentation(); slide = prs.slides.add_slide(prs.slide_layouts[5])
            title = slide.shapes.title; title.text = "AI Executive Summary"
            tx = slide.shapes.add_textbox(Inches(0.5), Inches(1.2), Inches(9), Inches(5)); tf = tx.text_frame; tf.text = "Generated by Enterprise Dashboard (clean)"
            out = io.BytesIO(); prs.save(out); out.seek(0)
            st.download_button("Download PPTX", data=out, file_name="ai_report.pptx", mime="application/vnd.openxmlformats-officedocument.presentationml.presentation")
    else:
        if not PPTX_AVAILABLE:
            st.info("python-pptx not installed — PPTX export disabled.")

# Data Quality (kept)
elif page == "Data Quality":
    st.header("📈 Data Quality Scoring")
    df = st.session_state.df
    if df is None:
        st.info("Load dataset first.")
    else:
        def dq_score(df_local):
            total = len(df_local) * df_local.shape[1]
            missing = df_local.isnull().sum().sum()
            completeness = 1 - missing/total if total > 0 else 0
            uniqueness = np.mean([df_local[c].nunique()/len(df_local) if len(df_local)>0 else 0 for c in df_local.columns])
            score = 100 * (0.6*completeness + 0.4*uniqueness)
            return round(score,1), {"completeness": round(completeness,3), "uniqueness": round(uniqueness,3)}
        score, meta = dq_score(df)
        st.metric("Data Quality Score", f"{score}/100")
        st.json(meta)

# TS Anomaly (kept)
elif page == "TS Anomaly":
    st.header("⏱️ Time-Series Anomaly Detection")
    df = st.session_state.df
    if df is None:
        st.info("Load dataset first.")
    else:
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric:
            st.warning("No numeric columns.")
        else:
            col = st.selectbox("Numeric column", numeric)
            if st.button("Detect anomalies (ruptures)"):
                if not RUPTURES_AVAILABLE:
                    st.error("Install 'ruptures' to enable time-series anomaly detection.")
                else:
                    series = df[col].fillna(method='ffill').values
                    algo = rpt.Pelt(model="rbf").fit(series)
                    result = algo.predict(pen=10)
                    st.write("Change points (indices):", result)
                    fig = px.line(df, y=col)
                    for r in result:
                        fig.add_vline(x=r, line_dash="dash", line_color="red")
                    st.plotly_chart(fig, use_container_width=True)

# Streaming (Simulated)
elif page == "Streaming (Simulated)":
    st.header("📡 Streaming (Simulated)")
    if st.button("Start simulated stream (50 events)"):
        placeholder = st.empty()
        for i in range(50):
            new = {"ts": pd.Timestamp.now(), "value": float(np.random.randn())}
            placeholder.write(new)
            time.sleep(0.2)
        st.success("Simulation complete.")

# Autosave & Versioning (kept)
elif page == "Autosave & Versioning":
    st.header("💾 Autosave & Versioning")
    df = st.session_state.df
    if df is None:
        st.info("Load dataset first.")
    else:
        if st.button("Save snapshot now"):
            p = save_snapshot(df)
            st.success(f"Saved snapshot: {p}")
        snaps = sorted(os.listdir("snapshots")) if os.path.exists("snapshots") else []
        st.write("Available snapshots:", snaps)
        choice = st.selectbox("Restore snapshot", ["--"] + snaps)
        if st.button("Restore snapshot"):
            if choice != "--":
                st.session_state.df = pd.read_parquet(os.path.join("snapshots", choice))
                st.success(f"Restored {choice}")

# Footer / default
st.markdown("---")
st.caption("Clean v3 — kept AI Formula Builder & AI Data Cleaner (Groq) and export features. Use toggles cautiously. Keep API keys secret.")
