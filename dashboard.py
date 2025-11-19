import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
import json
import io

st.set_page_config(page_title="Offline AI Analytics Dashboard", layout="wide")

# -------------------------------
# OLLAMA CHAT FUNCTION (OFFLINE)
# -------------------------------
def ollama_chat(prompt, model="llama3.1"):
    """Send prompt to local Ollama model and return the response text."""
    try:
        process = subprocess.Popen(
            ["ollama", "run", model],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        prompt = prompt + "\n"
        output, error = process.communicate(prompt.encode("utf-8"))

        output = output.decode("utf-8", errors="ignore")
        error = error.decode("utf-8", errors="ignore")

        if error.strip():
            return f"Ollama Error:\n{error}"

        if output.strip() == "":
            return "⚠️ No AI output. Check if the model is installed."

        return output

    except Exception as e:
        return f"Error running Ollama: {e}"



# -------------------------------
# SIDEBAR NAVIGATION
# -------------------------------
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Go to:", ["Dataset Upload", "Data Summary", "Visualizations", "AI Insights", "AI Chat"])



# -------------------------------
# SESSION STATE
# -------------------------------
if "df" not in st.session_state:
    st.session_state.df = None



# -------------------------------
# PAGE 1: DATA UPLOAD
# -------------------------------
if page == "Dataset Upload":
    st.title("📁 Upload Your Dataset (CSV)")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df

        st.success("File uploaded successfully!")
        st.dataframe(df.head())



# -------------------------------
# PAGE 2: DATA SUMMARY
# -------------------------------
elif page == "Data Summary":
    st.title("📘 Data Summary")

    df = st.session_state.df
    if df is None:
        st.warning("Upload a dataset first!")
    else:
        st.subheader("🔹 Dataset Preview")
        st.dataframe(df.head())

        st.subheader("🔹 Summary Statistics")
        st.write(df.describe())

        st.subheader("🔹 Missing Values")
        st.write(df.isnull().sum())



# -------------------------------
# PAGE 3: VISUALIZATIONS
# -------------------------------
elif page == "Visualizations":
    st.title("📈 Data Visualizations")

    df = st.session_state.df
    if df is None:
        st.warning("Upload a dataset first!")
    else:
        st.subheader("Select column for visualization")

        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

        chart_type = st.selectbox("Choose a chart type", ["Line Chart", "Histogram", "Correlation Heatmap"])

        if chart_type == "Line Chart":
            col = st.selectbox("Select numeric column", numeric_cols)
            fig, ax = plt.subplots()
            ax.plot(df[col])
            ax.set_title(f"{col} Trend")
            st.pyplot(fig)

        elif chart_type == "Histogram":
            col = st.selectbox("Select numeric column", numeric_cols)
            fig, ax = plt.subplots()
            ax.hist(df[col], bins=20)
            ax.set_title(f"{col} Distribution")
            st.pyplot(fig)

        elif chart_type == "Correlation Heatmap":
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
            st.pyplot(fig)



# -------------------------------
# PAGE 4: AI INSIGHTS (Offline)
# -------------------------------
elif page == "AI Insights":
    st.title("🤖 AI Insights (Offline – Ollama)")

    df = st.session_state.df
    if df is None:
        st.warning("Upload a dataset first!")
    else:
        prompt = f"""
        You are an expert data analyst.

        Dataset summary:
        {df.describe().to_string()}

        Missing values:
        {df.isnull().sum().to_string()}

        Provide analysis in this format:
        - Key trends
        - Outliers or anomalies
        - Business/operational recommendations
        """

        with st.spinner("AI analyzing data…"):
            result = ollama_chat(prompt)

        st.subheader("📌 AI Generated Insights")
        st.write(result)



# -------------------------------
# PAGE 5: AI CHAT (Offline)
# -------------------------------
elif page == "AI Chat":
    st.title("💬 Offline AI Chat (LLaMA + Ollama)")

    user_input = st.text_area("Ask the AI anything:")

    if st.button("Send") and user_input.strip() != "":
        with st.spinner("Thinking…"):
            result = ollama_chat(user_input)

        st.write(result)
