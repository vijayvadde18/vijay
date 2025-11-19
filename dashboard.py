import os
import time
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
from openai.error import OpenAIError  # may raise different exceptions depending on sdk

st.set_page_config(page_title="AI Analytics (Online)", layout="wide")

# -----------------------
# Initialize OpenAI client
# -----------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)

if not OPENAI_API_KEY:
    st.warning("OpenAI API key not found. Please add OPENAI_API_KEY to your Streamlit secrets.")
    # still continue so user can upload and see EDA without AI
    client = None
else:
    client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------
# Helper: call OpenAI with retries
# -----------------------
def get_ai_insights(prompt, model="gpt-4o-mini", max_retries=3, backoff=2):
    if client is None:
        return "AI not configured. Add OPENAI_API_KEY to Streamlit secrets to enable AI insights."

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
            )
            # adapt depending on SDK: get text safely
            text = response.choices[0].message["content"]
            return text

        except Exception as e:
            errstr = str(e)
            # Common actionable messages
            if "401" in errstr or "invalid_api_key" in errstr.lower():
                return "Authentication error: your API key is invalid. Check your OpenAI key in Streamlit secrets."
            if "429" in errstr or "quota" in errstr.lower() or "insufficient_quota" in errstr.lower():
                return ("Quota error: your account quota may be exhausted. "
                        "Check OpenAI billing or try again later.")
            # transient -> retry
            if attempt < max_retries:
                time.sleep(backoff ** attempt)
                continue
            # final fallback
            return f"AI request failed after {max_retries} attempts. Error: {errstr}"

# -----------------------
# Sidebar / Navigation
# -----------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Upload", "Summary", "Visualizations", "AI Insights", "AI Chat"])

if "df" not in st.session_state:
    st.session_state.df = None

# -----------------------
# Upload Page
# -----------------------
if page == "Upload":
    st.title("📁 Upload CSV")
    uploaded = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.session_state.df = df
            st.success("File loaded.")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")

# -----------------------
# Summary Page
# -----------------------
elif page == "Summary":
    st.title("📊 Data Summary")
    df = st.session_state.df
    if df is None:
        st.info("Upload a file in the Upload tab first.")
    else:
        st.subheader("Preview")
        st.dataframe(df.head())
        st.subheader("Summary Statistics")
        st.write(df.describe(include="all"))
        st.subheader("Missing Values")
        st.write(df.isnull().sum())

# -----------------------
# Visualizations Page
# -----------------------
elif page == "Visualizations":
    st.title("📈 Visualizations")
    df = st.session_state.df
    if df is None:
        st.info("Upload a file in the Upload tab first.")
    else:
        numeric = df.select_dtypes(include=["number"]).columns.tolist()
        if not numeric:
            st.warning("No numeric columns for plotting.")
        else:
            chart = st.selectbox("Chart type", ["Line", "Histogram", "Correlation Heatmap"])
            if chart == "Line":
                col = st.selectbox("Select column", numeric)
                fig, ax = plt.subplots()
                ax.plot(df[col].reset_index(drop=True))
                ax.set_title(f"{col} trend")
                st.pyplot(fig)
            elif chart == "Histogram":
                col = st.selectbox("Select column", numeric)
                fig, ax = plt.subplots()
                ax.hist(df[col].dropna(), bins=30)
                ax.set_title(f"{col} distribution")
                st.pyplot(fig)
            else:
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax)
                st.pyplot(fig)

# -----------------------
# AI Insights Page
# -----------------------
elif page == "AI Insights":
    st.title("🤖 AI-Generated Insights (Online)")
    df = st.session_state.df
    if df is None:
        st.info("Upload a file in the Upload tab first.")
    else:
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        with st.expander("Customize prompt / options"):
            model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4.1-mini", "gpt-3.5-turbo"], index=0)
            include_columns = st.multiselect("Columns to include (empty = all)", df.columns.tolist())
            max_tokens = st.slider("Max tokens for response", 200, 2000, 800)

        # Build the prompt
        use_cols = include_columns if include_columns else df.columns.tolist()
        summary_stats = df[use_cols].describe().to_string()
        missing = df[use_cols].isnull().sum().to_string()

        prompt = f"""You are an expert data analyst. Analyze the dataset summary below.

Summary Statistics:
{summary_stats}

Missing values:
{missing}

Provide:
- Key trends (top 3)
- Any anomalies/outliers to investigate
- Actionable business recommendations (3)
"""

        if st.button("Generate AI Insights"):
            with st.spinner("Generating insights..."):
                res = get_ai_insights(prompt, model=model, max_retries=3)
                st.subheader("AI Insights")
                st.write(res)

# -----------------------
# AI Chat Page
# -----------------------
elif page == "AI Chat":
    st.title("💬 AI Chat (Online)")
    user_prompt = st.text_area("Ask the AI (context: dataset is loaded)", height=120)
    model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4.1-mini", "gpt-3.5-turbo"], key="chat_model")

    if st.button("Send"):
        if not user_prompt.strip():
            st.warning("Write a message first.")
        else:
            # optionally add dataset context
            df = st.session_state.df
            if df is not None:
                context = f"\n\nDataset preview:\n{df.head(5).to_string()}\n\n"
            else:
                context = ""
            full_prompt = user_prompt + context
            with st.spinner("Thinking..."):
                out = get_ai_insights(full_prompt, model=model)
                st.write(out)

