# enterprise_dashboard_v2.py
# Power BI style enterprise dashboard (single-file)
# Features: Data Explorer, Cleaning, Auto-EDA, Visualizations, Outliers, Forecasting,
# AutoML, AI Agent (Groq), SQL Engine, Export, Deployment helpers.
# IMPORTANT: set GROQ_API_KEY in env or .streamlit/secrets.toml before using AI features.

import os
import io
import sqlite3
import traceback
import base64
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ML & utils
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

# Optional libs
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

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

# ---------------------------
# Styling (Power BI-ish)
# ---------------------------
PRIMARY = "#F0C419"   # Power BI yellow
BG = "#0b1720"        # dark background
CARD = "#0f2430"
TEXT = "#E6EEF2"

st.set_page_config(page_title="PowerBI-Style AI Analytics", layout="wide", initial_sidebar_state="expanded")

st.markdown(f"""
    <style>
        .reportview-container {{ background-color: {BG}; color: {TEXT}; }}
        .sidebar .sidebar-content {{ background-color: #07262f; }}
        .stButton>button {{ background-color: {PRIMARY}; color: black; border: none; }}
        .stDownloadButton>button {{ background-color: {PRIMARY}; color: black; border: none; }}
        .card {{ background: {CARD}; padding: 12px; border-radius:10px; box-shadow: 0 2px 6px rgba(0,0,0,0.4); }}
        .metric-label {{ color: #b8c2c7; font-size: 13px; }}
    </style>
""", unsafe_allow_html=True)

st.title("📊 Power BI — Style Enterprise AI Analytics (Groq)")

# ---------------------------
# Load Groq Key
# ---------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    try:
        GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    except Exception:
        GROQ_API_KEY = None

if GROQ_API_KEY and GROQ_AVAILABLE:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
    except Exception:
        groq_client = None
else:
    groq_client = None

def groq_chat(prompt: str, model: str = "llama-3.1-70b-versatile", max_tokens: int = 800):
    """Call Groq chat; returns string. Graceful error if not configured."""
    if groq_client is None:
        return "Groq not configured or 'groq' package missing."
    try:
        resp = groq_client.chat.completions.create(
            model=model,
            messages=[{"role":"user","content":prompt}],
            max_tokens=max_tokens
        )
        # adapt to response structure
        # earlier Groq SDK returns choices[0].message["content"]
        text = resp.choices[0].message["content"]
        return text
    except Exception as e:
        return f"Groq Error: {e}"

# ---------------------------
# Sidebar: navigation + settings
# ---------------------------
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/4/4a/Microsoft_Power_BI_logo.svg",
    width=140
)
st.sidebar.markdown("## Navigation")
page = st.sidebar.radio("", [
    "Home", "Data Explorer", "Cleaning & Prep", "Auto-EDA", "Visualizations",
    "Outliers", "Forecasting", "AutoML", "AI Agent", "SQL Engine", "Export Center", "Deployment"
])

st.sidebar.markdown("---")
st.sidebar.markdown("### Global settings")
ENABLE_EDA = st.sidebar.checkbox("Enable Auto-EDA (ydata_profiling)", value=False)
ENABLE_AI = st.sidebar.checkbox("Enable AI (Groq)", value=bool(groq_client))
SAFE_EXEC = st.sidebar.checkbox("Allow AI-suggested code execution", value=False)
CONFIRM_EXEC = st.sidebar.checkbox("Confirm execution (required to run AI code)", value=False)

# session state init
if "df" not in st.session_state: st.session_state.df = None
if "cleaned_df" not in st.session_state: st.session_state.cleaned_df = None
if "sqlite_conn" not in st.session_state: st.session_state.sqlite_conn = None
if "models" not in st.session_state: st.session_state.models = {}

# ---------- Helper utilities ----------
def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    """Return Excel bytes for download."""
    towrite = io.BytesIO()
    with pd.ExcelWriter(towrite, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False)
    towrite.seek(0)
    return towrite.read()

def download_bytes(data: bytes, filename: str, mime="application/octet-stream"):
    st.download_button(f"Download {filename}", data=data, file_name=filename, mime=mime)

# -------------
# Home
# -------------
if page == "Home":
    st.markdown("## Welcome")
    st.markdown("""
    This is a Power BI style enterprise analytics dashboard powered by Groq AI.
    Use the sidebar to navigate pages.  
    **Keep your GROQ_API_KEY secure** — set as environment variable or Streamlit secrets.
    """)
    st.markdown("### Quick status")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='card'><b>Data</b><br>", unsafe_allow_html=True)
        if st.session_state.df is None:
            st.markdown("<span class='metric-label'>No dataset loaded</span></div>", unsafe_allow_html=True)
        else:
            d = st.session_state.df
            st.markdown(f"<span class='metric-label'>{d.shape[0]:,} rows • {d.shape[1]:,} cols</span></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='card'><b>AI</b><br>", unsafe_allow_html=True)
        if groq_client:
            st.markdown("<span class='metric-label'>Groq configured</span></div>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='metric-label'>Groq not configured</span></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='card'><b>Export</b><br>", unsafe_allow_html=True)
        st.markdown("<span class='metric-label'>CSV / Excel / PDF / PPTX</span></div>", unsafe_allow_html=True)

# -------------
# Data Explorer
# -------------
elif page == "Data Explorer":
    st.header("📁 Data Explorer")
    st.markdown("Upload a CSV/XLSX or use the sample dataset.")
    source = st.radio("Source:", ["Upload", "Sample"], index=0, horizontal=True)
    if source == "Sample":
        dates = pd.date_range(end=pd.Timestamp.today(), periods=180, freq="D")
        sample = pd.DataFrame({
            "date": dates,
            "region": np.random.choice(["North","South","East","West"], len(dates)),
            "product": np.random.choice(["A","B","C"], len(dates)),
            "units_sold": np.random.poisson(20, len(dates)),
            "price": np.random.uniform(10,100,len(dates))
        })
        sample["revenue"] = sample["units_sold"] * sample["price"]
        st.session_state.df = sample
        st.success("Sample dataset loaded into session.")
    else:
        uploaded = st.file_uploader("Upload CSV or XLSX", type=["csv","xlsx"])
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
    df = st.session_state.df
    if df is not None:
        st.subheader("Preview")
        st.dataframe(df.head(200))
        st.subheader("Columns")
        cols = df.columns.tolist()
        st.write(cols)
        if st.button("Show dtypes"):
            st.write(df.dtypes)

# -------------
# Cleaning & Prep
# -------------
elif page == "Cleaning & Prep":
    st.header("🧹 Cleaning & Preparation")
    df = st.session_state.df
    if df is None:
        st.info("Load a dataset in Data Explorer first.")
    else:
        left, right = st.columns([2,1])
        with left:
            st.subheader("Quick actions")
            if st.button("Drop duplicates"):
                df = df.drop_duplicates().reset_index(drop=True); st.session_state.df = df; st.success("Duplicates dropped")
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
                    st.session_state.df = df
                    st.success("Missing values filled")
            if st.button("Save cleaned copy to session"):
                st.session_state.cleaned_df = df.copy(); st.success("Saved cleaned copy")
        with right:
            st.subheader("Column ops")
            col = st.selectbox("Select column", df.columns.tolist())
            if col:
                st.write(df[col].describe(include="all"))
                if st.button("Drop column"):
                    df = df.drop(columns=[col]); st.session_state.df = df; st.success(f"Dropped {col}")

# -------------
# Auto-EDA
# -------------
elif page == "Auto-EDA":
    st.header("📊 Auto-EDA / Profiling")
    df = st.session_state.df
    if df is None:
        st.info("Load a dataset first.")
    else:
        if ENABLE_EDA and YDATA_AVAILABLE:
            with st.spinner("Running profiling (ydata_profiling)..."):
                profile = ProfileReport(df, minimal=True)
                html = profile.to_html()
                st.components.v1.html(html, height=800, scrolling=True)
                b64 = base64.b64encode(html.encode()).decode()
                st.markdown(f'<a href="data:text/html;base64,{b64}" download="eda_report.html">📥 Download EDA (HTML)</a>', unsafe_allow_html=True)
        else:
            st.write("Basic summary:")
            st.write(df.describe(include="all"))
            st.write("Missing values:")
            st.write(df.isnull().sum())

# -------------
# Visualizations
# -------------
elif page == "Visualizations":
    st.header("📈 Visualizations")
    df = st.session_state.df
    if df is None:
        st.info("Load dataset first.")
    else:
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical = df.select_dtypes(include=["object","category"]).columns.tolist()
        st.subheader("Single column charts")
        col = st.selectbox("Select numeric column", numeric) if numeric else None
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
            fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
            st.plotly_chart(fig, use_container_width=True)

# -------------
# Outliers
# -------------
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
            if st.button("Detect outliers"):
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
                    st.write(out.head(200))
                    st.success(f"Found {len(out)} outliers")
                except Exception as e:
                    st.error(f"Outlier detection error: {e}")

# -------------
# Forecasting
# -------------
elif page == "Forecasting":
    st.header("🔮 Forecasting")
    df = st.session_state.df
    if df is None:
        st.info("Load dataset first.")
    else:
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric:
            st.warning("No numeric columns.")
        else:
            fc_col = st.selectbox("Numeric column to forecast", numeric)
            date_candidates = [c for c in df.columns if "date" in c.lower() or pd.api.types.is_datetime64_any_dtype(df[c])]
            date_col = st.selectbox("Date column (optional)", ["--"] + date_candidates, index=0)
            periods = st.number_input("Periods to forecast", min_value=1, max_value=365, value=30)
            if st.button("Run Forecast"):
                try:
                    if date_col != "--":
                        data = df[[date_col, fc_col]].dropna().rename(columns={date_col:"ds", fc_col:"y"})
                        data["ds"] = pd.to_datetime(data["ds"])
                    else:
                        data = df[[fc_col]].dropna().reset_index().rename(columns={"index":"ds", fc_col:"y"})
                    if PROPHET_AVAILABLE:
                        m = Prophet()
                        m.fit(data)
                        future = m.make_future_dataframe(periods=periods)
                        forecast = m.predict(future)
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=data["ds"], y=data["y"], name="history"))
                        fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], name="forecast"))
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Prophet not installed. Install 'prophet' to enable advanced forecasting.")
                except Exception as e:
                    st.error(f"Forecast error: {e}\n{traceback.format_exc()}")

# -------------
# AutoML
# -------------
elif page == "AutoML":
    st.header("🤖 AutoML (Quick Train)")
    df = st.session_state.df
    if df is None:
        st.info("Load dataset first.")
    else:
        target = st.selectbox("Target column", ["--"] + df.columns.tolist())
        if target and target != "--":
            X = df.drop(columns=[target]).select_dtypes(include=[np.number])
            y = df[target]
            if X.shape[1] == 0:
                st.warning("No numeric features — do feature engineering first.")
            else:
                test_size = st.slider("Test size", 0.1, 0.5, 0.2)
                if st.button("Train (Auto select RF)"):
                    try:
                        stratify = y if not pd.api.types.is_numeric_dtype(y) and len(y.unique()) > 1 else None
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=stratify)
                        if pd.api.types.is_numeric_dtype(y):
                            model = RandomForestRegressor(n_estimators=150, random_state=42)
                            model.fit(X_train, y_train)
                            preds = model.predict(X_test); rmse = mean_squared_error(y_test, preds, squared=False); r2 = r2_score(y_test, preds)
                            st.success(f"Regression trained. RMSE: {rmse:.3f}, R2: {r2:.3f}")
                            st.bar_chart(pd.Series(model.feature_importances_, index=X.columns).nlargest(15))
                            st.session_state.models["automl"] = ("regression", model, X.columns.tolist())
                        else:
                            model = RandomForestClassifier(n_estimators=150, random_state=42)
                            model.fit(X_train, y_train)
                            preds = model.predict(X_test); acc = accuracy_score(y_test, preds)
                            st.success(f"Classification trained. Accuracy: {acc:.3f}")
                            st.write(classification_report(y_test, preds))
                            st.session_state.models["automl"] = ("classification", model, X.columns.tolist())
                    except Exception as e:
                        st.error(f"Training error: {e}\n{traceback.format_exc()}")
        if "automl" in st.session_state.models:
            if st.button("Download trained model"):
                mtype, m, fcols = st.session_state.models["automl"]
                buf = io.BytesIO(); joblib.dump(m, buf); buf.seek(0)
                download_bytes(buf.read(), f"automl_{mtype}.joblib", mime="application/octet-stream")

# -------------
# AI Agent
# -------------
elif page == "AI Agent":
    st.header("🧠 AI Agent (Insights, Chat, SQL & Chart generation)")
    df = st.session_state.df
    if not ENABLE_AI:
        st.info("Enable AI from the sidebar to use Groq features.")
    elif groq_client is None:
        st.error("Groq client not configured. Install 'groq' and set GROQ_API_KEY.")
    else:
        st.subheader("AI Insights")
        cols = st.multiselect("Columns for AI context (empty=all)", df.columns.tolist() if df is not None else [])
        instr = st.text_area("Instructions (optional)", value="You are an expert data analyst. Provide key trends, anomalies, and 3 actionable recommendations.")
        if st.button("Generate Insights"):
            usecols = cols if cols else (df.columns.tolist() if df is not None else [])
            summary = df[usecols].describe().to_string() if df is not None else "No dataset"
            missing = df[usecols].isnull().sum().to_string() if df is not None else ""
            prompt = f"{instr}\n\nSummary statistics:\n{summary}\n\nMissing:\n{missing}\n\nAnswer in bullets."
            with st.spinner("Calling Groq..."):
                ans = groq_chat(prompt)
                st.markdown("#### AI Insights")
                st.write(ans)

        st.subheader("AI Chat")
        chat = st.text_area("Ask the agent about your data:", height=150)
        if st.button("Send Chat"):
            context = f"Dataset preview:\n{df.head(5).to_string()}\n\nColumns: {', '.join(df.columns.tolist())}\n\n" if df is not None else ""
            prompt = context + chat
            with st.spinner("Calling Groq..."):
                res = groq_chat(prompt)
                st.markdown("#### AI Response")
                st.write(res)

        st.subheader("NL → SQL")
        nl = st.text_input("Describe the query in plain English", value="Top 5 products by revenue")
        if st.button("Generate SQL"):
            prompt = f"You have a table 'data_table' with columns: {', '.join(df.columns.tolist()) if df is not None else ''}. Write SQL to: {nl}. Return only the SQL."
            out = groq_chat(prompt)
            st.code(out)
            if st.button("Run SQL (load into SQLite first)"):
                if st.session_state.sqlite_conn is None:
                    st.warning("Load dataset into SQLite on the SQL Engine page first.")
                else:
                    try:
                        res = pd.read_sql_query(out, st.session_state.sqlite_conn)
                        st.dataframe(res.head(300))
                    except Exception as e:
                        st.error(f"SQL execution error: {e}")

        st.subheader("NL → Chart code generator")
        chart_desc = st.text_input("Describe chart", value="Line chart of revenue over date grouped by product")
        if st.button("Generate Chart Code"):
            prompt = f"Write python plotly express code that given a pandas DataFrame 'df' with columns {', '.join(df.columns.tolist()) if df is not None else ''} creates the chart: {chart_desc}. Return only code and set variable 'fig'."
            code = groq_chat(prompt)
            st.code(code)
            if SAFE_EXEC and CONFIRM_EXEC:
                try:
                    local_env = {"df": df.copy(), "px": px, "pd": pd, "np": np}
                    exec(code, {}, local_env)
                    fig = local_env.get("fig")
                    if fig is not None:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Code did not produce 'fig'.")
                except Exception as e:
                    st.error(f"Execution failed: {e}")
            else:
                st.info("Enable code execution in sidebar and confirm to run generated code.")

# -------------
# SQL Engine
# -------------
elif page == "SQL Engine":
    st.header("🗄️ SQL Engine (SQLite in-memory)")
    df = st.session_state.df
    if df is None:
        st.info("Load dataset first.")
    else:
        if st.button("Load dataset into SQLite (in-memory)"):
            conn = sqlite3.connect(":memory:")
            df.to_sql("data_table", conn, index=False, if_exists="replace")
            st.session_state.sqlite_conn = conn
            st.success("Loaded into SQLite as 'data_table'")
        if st.session_state.sqlite_conn:
            q = st.text_area("SQL query (run on data_table)", value="SELECT * FROM data_table LIMIT 100")
            if st.button("Execute SQL"):
                try:
                    res = pd.read_sql_query(q, st.session_state.sqlite_conn)
                    st.dataframe(res.head(500))
                except Exception as e:
                    st.error(f"SQL error: {e}")

# -------------
# Export Center
# -------------
elif page == "Export Center":
    st.header("📤 Export & Reports")
    if st.session_state.cleaned_df is not None:
        st.download_button("Download cleaned CSV", data=st.session_state.cleaned_df.to_csv(index=False).encode(), file_name="cleaned.csv", mime="text/csv")
        st.download_button("Download cleaned Excel", data=df_to_excel_bytes(st.session_state.cleaned_df), file_name="cleaned.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    elif st.session_state.df is not None:
        st.download_button("Download dataset CSV", data=st.session_state.df.to_csv(index=False).encode(), file_name="dataset.csv", mime="text/csv")
    if REPORTLAB_AVAILABLE and st.session_state.df is not None and st.button("Generate AI PDF Summary"):
        try:
            prompt = f"Summarize dataset with columns: {', '.join(st.session_state.df.columns.tolist())}. Provide top 5 insights and 3 recommendations."
            text = groq_chat(prompt) if ENABLE_AI else "AI disabled"
            buf = io.BytesIO()
            c = canvas.Canvas(buf, pagesize=letter)
            y = 750
            c.setFont("Helvetica-Bold", 14); c.drawString(30, y, "AI Generated Summary"); y -= 30
            c.setFont("Helvetica", 10)
            for line in text.splitlines():
                if y < 40:
                    c.showPage(); y = 750
                c.drawString(30, y, line[:110]); y -= 12
            c.save(); buf.seek(0)
            st.download_button("Download PDF", data=buf, file_name="ai_summary.pdf", mime="application/pdf")
        except Exception as e:
            st.error(f"PDF generation error: {e}")

# -------------
# Deployment
# -------------
elif page == "Deployment":
    st.header("🚀 Deployment & Helpers")
    st.markdown("Generate Dockerfile and README for deployment.")
    if st.button("Create Dockerfile & README in current folder"):
        try:
            docker_text = """FROM python:3.10-slim

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8501
CMD ["streamlit", "run", "enterprise_dashboard_v2.py", "--server.port=8501", "--server.address=0.0.0.0"]
"""
            # write files
            with open("Dockerfile","w",encoding="utf-8") as f:
                f.write(docker_text)
            with open("README_DEPLOY.md","w",encoding="utf-8") as f:
                f.write("# Deploy\nRun `docker build -t enterprise-ai .` and then `docker run -p 8501:8501 enterprise-ai`")
            st.success("Wrote Dockerfile and README_DEPLOY.md to current folder.")
        except Exception as e:
            st.error(f"Write error: {e}")

st.markdown("---")
st.caption("Built in Power BI color theme. Keep your API keys secure and never run untrusted AI code in production.")
