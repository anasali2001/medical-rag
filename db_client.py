import streamlit as st
from supabase import create_client

# Initialize connection
@st.cache_resource
def init_supabase():
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    return create_client(url, key)

def insert_lab_results(data_list):
    supabase = init_supabase()
    # Supabase allows bulk inserts
    response = supabase.table("lab_results").insert(data_list).execute()
    return response

def fetch_all_results():
    supabase = init_supabase()
    response = supabase.table("lab_results").select("*").order("report_date").execute()
    return response.data