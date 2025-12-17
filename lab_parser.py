import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st


def parse_pdf_text_to_json(pdf_text):
    """
    Uses LLM to extract structured data from raw PDF text.
    """

    # Using a cheaper/faster model for extraction (e.g., Gemini Flash or Kimi via OpenRouter)
    llm = ChatOpenAI(
        model="google/gemini-2.0-flash-001",  # Good for extraction
        api_key=st.secrets["openrouter"]["api_key"],
        base_url="https://openrouter.ai/api/v1",
        temperature=0.0
    )

    prompt_text = """
    You are a medical data extraction assistant. 
    Analyze the following OCR text from a lab report.
    Extract the Test Name, Value, Unit, and calculate the Status based on reference ranges found in text.

    Return ONLY a valid JSON list. Schema:
    [
      {{"test_name": "Hemoglobin", "value": 13.5, "unit": "g/dL", "report_date": "YYYY-MM-DD", "status": "Normal"}},
      ...
    ]

    Rules:
    1. If date is missing in a row, use the report date found in the header.
    2. Convert all numeric values to floats.
    3. Status must be 'Normal', 'Abnormal', or 'Unknown'.

    Input Text:
    {text}
    """

    prompt = ChatPromptTemplate.from_template(prompt_text)
    chain = prompt | llm | StrOutputParser()

    try:
        # We slice text to avoid token limits if PDF is huge
        response_str = chain.invoke({"text": pdf_text[:15000]})
        # Clean up Markdown code blocks if LLM adds them
        clean_json = response_str.replace("```json", "").replace("```", "")
        return json.loads(clean_json)
    except Exception as e:
        print(f"Parsing Error: {e}")
        return []