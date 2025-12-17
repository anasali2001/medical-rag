import streamlit as st
import pandas as pd
from pathlib import Path
import os
os.environ["OPENROUTER_API_KEY"] = st.secrets["openrouter"]["api_key"]
SUPABASE_URL = st.secrets["supabase"]["url"]
SUPABASE_KEY = st.secrets["supabase"]["key"]
import re

# Import components
from rag_cli import make_chain
from db_client import insert_lab_results, fetch_all_results
from analytics_engine import generate_trend_chart
# We will use existing ingest functions but call them manually for CSV text
from ingest import chunk_docs, build_vectorstore
from langchain_core.documents import Document

st.set_page_config(page_title="Personal Medical Insights", layout="wide")


MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
}


def clean_value(val):
    """
    Cleans dirty CSV values like '97/26', '?', or empty strings.
    """
    if pd.isna(val) or val == "" or str(val).strip() in ["?", "nan", "None"]:
        return None

    val_str = str(val).strip()

    # If formatted like "97/26", take the first number (97)
    if "/" in val_str:
        val_str = val_str.split("/")[0]

    # Regex to extract the first valid float number found
    match = re.search(r"(\d+(\.\d+)?)", val_str)
    if match:
        try:
            return float(match.group(1))
        except:
            return None
    return None


import io  # <--- ADD THIS IMPORT AT THE TOP


def process_csv_file(uploaded_file):
    filename = uploaded_file.name

    # 1. Determine default year
    year_match = re.search(r"20\d{2}", filename)
    default_year = int(year_match.group(0)) if year_match else 2024

    # 2. Pre-process to fix trailing commas
    try:
        content = uploaded_file.getvalue().decode("utf-8")
        lines = content.splitlines()
        if not lines: return [], []

        header_line = lines[0]
        expected_cols = len(header_line.split(','))

        clean_lines = [header_line]
        for line in lines[1:]:
            if not line.strip(): continue
            parts = line.split(',')
            clean_parts = parts[:expected_cols]
            while len(clean_parts) < expected_cols:
                clean_parts.append("")
            clean_lines.append(",".join(clean_parts))

        df = pd.read_csv(io.StringIO("\n".join(clean_lines)))
    except Exception as e:
        st.error(f"Error processing {filename}: {e}")
        return [], []

    df.columns = [str(c).strip() for c in df.columns]
    id_col = df.columns[0]

    db_records = []
    rag_docs = []

    # 3. Iterate Rows
    for _, row in df.iterrows():
        test_name = row[id_col]
        if pd.isna(test_name) or str(test_name).strip() == "":
            continue

        # --- NEW: Collect all values for this row to make ONE smart summary ---
        row_summary_parts = []

        for col_name in df.columns[1:]:
            val = row[col_name]
            clean_val = clean_value(val)

            if clean_val is not None:
                # Date Logic
                col_lower = col_name.lower()
                target_year = default_year
                target_month = None

                year_suffix_match = re.search(r"([a-z]{3})(\d{2})", col_lower)
                if year_suffix_match:
                    month_str = year_suffix_match.group(1)
                    target_year = 2000 + int(year_suffix_match.group(2))
                    if month_str in MONTH_MAP: target_month = MONTH_MAP[month_str]
                else:
                    clean_col = re.sub(r"[^a-zA-Z]", "", col_lower)
                    if clean_col in MONTH_MAP: target_month = MONTH_MAP[clean_col]

                if target_month:
                    date_str = f"{target_year}-{target_month:02d}-01"

                    # Add to Database (For Tab 1 Visualization)
                    db_records.append({
                        "test_name": test_name,
                        "value": clean_val,
                        "unit": None,
                        "report_date": date_str,
                        "status": "Unknown"
                    })

                    # Add to text summary list (For Tab 2 AI)
                    month_name = [k for k, v in MONTH_MAP.items() if v == target_month][0].title()
                    row_summary_parts.append(f"{month_name} {target_year}: {clean_val}")

        # --- Create ONE Document per Test per File ---
        if row_summary_parts:
            full_text_summary = f"Lab Test: {test_name}. Source File: {filename}. Data points: " + ", ".join(
                row_summary_parts) + "."
            rag_docs.append(Document(page_content=full_text_summary, metadata={"source": filename, "test": test_name}))

    return db_records, rag_docs





def main():
    st.title("🏥 Personal Health Data Hub")

    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "💬 AI Doctor (RAG)", "📥 Upload CSVs"])

    # ---------------- TAB 1: DASHBOARD ----------------
    with tab1:
        st.header("Health Trends")
        data = fetch_all_results()

        if not data:
            st.info("Database empty. Please upload 2024.csv / 2025.csv in Tab 3.")
        else:
            df = pd.DataFrame(data)
            all_tests = df['test_name'].unique()
            selected_test = st.selectbox("Select Lab Metric:", all_tests)

            col1, col2 = st.columns([3, 1])
            with col1:
                # Generate Chart
                fig, pred_val = generate_trend_chart(df, selected_test)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Analysis")
                subset = df[df['test_name'] == selected_test].sort_values("report_date")
                if not subset.empty:
                    latest = subset.iloc[-1]
                    st.metric(f"Latest ({latest['report_date']})", f"{latest['value']}")
                    if pred_val:
                        st.metric("Predicted (30d)", f"{pred_val:.2f}")

    # ---------------- TAB 2: RAG Q&A ----------------
    with tab2:
        st.header("Ask your Data")
        try:
            chain = make_chain()
            rag_ready = True
        except Exception:
            rag_ready = False
            st.warning("AI not ready. Please ingest CSVs first.")

        query = st.text_input("Question:", placeholder="Compare my Hb between 2024 and 2025")

        if st.button("Ask AI") and rag_ready and query:
            with st.spinner("Thinking..."):
                res = chain.invoke(query)
                st.write(res)

    # ---------------- TAB 3: CSV UPLOAD ----------------
        # ---------------- TAB 3: CSV UPLOAD ----------------
    with tab3:
        st.header("Upload 2024/2025 CSVs")
        uploaded_files = st.file_uploader("Upload CSV files", type="csv", accept_multiple_files=True)

        if st.button("Process & Ingest"):
            if not uploaded_files:
                st.warning("Select files first.")
            else:
                all_db_records = []
                all_rag_docs = []

                progress = st.progress(0)
                status_text = st.empty()

                for i, file in enumerate(uploaded_files):
                    status_text.text(f"📂 Processing {file.name}...")

                        # Parse CSV
                    recs, docs = process_csv_file(file)

                    if recs:
                        all_db_records.extend(recs)
                        all_rag_docs.extend(docs)
                    else:
                        st.warning(f"No valid data found in {file.name}")

                    # 1. Insert into Supabase
                if all_db_records:
                    status_text.text(f"💾 Inserting {len(all_db_records)} records to Database...")
                    try:
                        insert_lab_results(all_db_records)
                        st.success(f"Successfully saved {len(all_db_records)} results.")
                    except Exception as e:
                        st.error(f"Database Error: {e}")

                    # 2. Update Vector DB for AI
                if all_rag_docs:
                    status_text.text(f"🧠 Teaching AI {len(all_rag_docs)} data points...")
                    chunks = chunk_docs(all_rag_docs)
                    build_vectorstore(chunks)

                progress.progress(100)
                status_text.text("Done!")
        st.success("Done! Go to Dashboard or AI Chat.")


if __name__ == "__main__":
    main()
