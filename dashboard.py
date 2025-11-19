import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai

st.set_page_config(page_title="AI Analytics (Gemini Online)", layout="wide")

# -----------------------
# Initialize Gemini Client
# -----------------------
GEMINI_API_KEY = "AIzaSyCNb1fPdEAwG3qQsPjmrdPDFJrdZ8rWXz0"

if not GEMINI_API_KEY:
    st.warning("Gemini API Key not found. Please add GEMINI_API_KEY to Streamlit secrets.")
else:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-pro-latest")

    
    

def get_gemini_response(prompt):
    """Helper to call Gemini with safe error handling."""
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini Error: {str(e)}"


# -----------------------
# Sidebar Navigation
# -----------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Upload", "Summary", "Visualizations", "AI Insights", "AI Chat"])

if "df" not in st.session_state:
    st.session_state.df = None

# -----------------------
# Upload
# -----------------------
if page == "Upload":
    st.title("📁 Upload CSV")
    uploaded = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.session_state.df = df
        st.success("File loaded!")
        st.dataframe(df.head())

# -----------------------
# Summary
# -----------------------
elif page == "Summary":
    st.title("📊 Data Summary")
    df = st.session_state.df
    if df is None:
        st.info("Upload a file first.")
    else:
        st.subheader("Preview")
        st.dataframe(df.head())

        st.subheader("Summary Statistics")
        st.write(df.describe(include="all"))

        st.subheader("Missing Values")
        st.write(df.isnull().sum())

# -----------------------
# Visualizations
# -----------------------
elif page == "Visualizations":
    st.title("📈 Visualizations")
    df = st.session_state.df
    if df is None:
        st.info("Upload a file first.")
    else:
        numeric = df.select_dtypes(include=["number"]).columns.tolist()
        if not numeric:
            st.warning("No numeric columns available.")
        else:
            chart = st.selectbox("Chart type", ["Line", "Histogram", "Correlation Heatmap"])

            if chart == "Line":
                col = st.selectbox("Column", numeric)
                fig, ax = plt.subplots()
                ax.plot(df[col])
                ax.set_title(f"{col} Trend")
                st.pyplot(fig)

            elif chart == "Histogram":
                col = st.selectbox("Column", numeric)
                fig, ax = plt.subplots()
                ax.hist(df[col], bins=30)
                ax.set_title(f"{col} Distribution")
                st.pyplot(fig)

            else:
                fig, ax = plt.subplots(figsize=(9, 6))
                sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
                st.pyplot(fig)

# -----------------------
# AI Insights
# -----------------------
elif page == "AI Insights":
    st.title("🤖 AI Insights (Gemini)")

    df = st.session_state.df
    if df is None:
        st.info("Upload a file first.")
    else:
        prompt = f"""
        You are a senior data analyst.

        Dataset summary:
        {df.describe().to_string()}

        Missing values:
        {df.isnull().sum().to_string()}

        Provide insights in this format:
        - Key trends
        - Patterns
        - Outliers/anomalies
        - Actionable recommendations
        """

        if st.button("Generate Insights"):
            with st.spinner("Thinking..."):
                result = get_gemini_response(prompt)
                st.subheader("Insights")
                st.write(result)

# -----------------------
# AI Chat
# -----------------------
elif page == "AI Chat":
    st.title("💬 AI Chat (Gemini)")

    query = st.text_area("Ask something about your data:")
    if st.button("Send Question"):
        df = st.session_state.df
        context = f"\nDataset Preview:\n{df.head().to_string()}\n\n" if df is not None else ""

        full_prompt = context + query

        with st.spinner("Thinking..."):
            res = get_gemini_response(full_prompt)

        st.subheader("Response")
        st.write(res)


