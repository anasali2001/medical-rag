# 🏥 Personal Health Data Hub (PMIA)

An AI-powered medical insights platform that transforms unstructured lab reports into structured data, visual trends, and predictive health insights.

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)
![Supabase](https://img.shields.io/badge/Supabase-3ECF8E?style=flat&logo=supabase&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-121212?style=flat&logo=chainlink&logoColor=white)

## 🌟 Project Vision
Personal medical records are often trapped in static PDFs or disparate CSVs. **PMIA** provides a secure, centralized hub where users can track long-term health trends, receive AI-powered explanations of their metrics, and forecast future values using Machine Learning.

## ✨ Features
- **📊 Interactive Dashboard:** Track historical trends for any lab metric (Hb, Cholesterol, Vitamin D, etc.) using high-performance Plotly charts.
- **💬 AI Medical Assistant (RAG):** Ask natural language questions like *"How has my Hemoglobin changed since last year?"* using Retrieval-Augmented Generation.
- **📈 Predictive Analytics:** Uses `scikit-learn` Linear Regression to forecast health metrics 30 days into the future.
- **📥 Smart Ingestion:** Handles dirty CSV data, fixes formatting issues, and maps dates automatically.
- **💾 Robust Persistence:** Integrated with Supabase (PostgreSQL) for reliable cloud storage, replacing ephemeral local files.

## 🛠️ Tech Stack
| Layer | Technology |
| :--- | :--- |
| **Frontend** | [Streamlit](https://streamlit.io/) |
| **Database** | [Supabase](https://supabase.com/) (PostgreSQL) |
| **Vector Store** | [ChromaDB](https://www.trychroma.com/) |
| **LLM Engine** | [OpenRouter](https://openrouter.ai/) (Kimi-k2 / Gemini 1.5 Flash) |
| **Orchestration**| [LangChain](https://www.langchain.com/) |
| **ML/Analytics** | [Scikit-learn](https://scikit-learn.org/), Pandas, Plotly |

## 📐 Architecture
1. **Ingestion:** User uploads CSV/PDF files.
2. **Structuring:** Python logic cleans values; LLM Agents (via OpenRouter) extract entities.
3. **Storage:** Structured data is pushed to Supabase; Unstructured text is vectorized into ChromaDB.
4. **Analysis:** Scikit-learn processes historical points to generate a regression line.
5. **Retrieval:** LangChain retrieves relevant context from ChromaDB to answer user queries via the Chat interface.

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- A [Supabase](https://supabase.com/) project
- An [OpenRouter](https://openrouter.ai/) API Key

### Installation
1. **Clone the repository:**
   ```bash
   git repo clone [YOUR_USERNAME]/medical-rag
   cd medical-rag
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure Secrets:**
   Create a `.streamlit/secrets.toml` file:
   ```toml
   [supabase]
   url = "YOUR_SUPABASE_URL"
   key = "YOUR_SUPABASE_SERVICE_ROLE_KEY"

   [openrouter]
   api_key = "YOUR_OPENROUTER_API_KEY"
   ```

### Database Setup
Run the following SQL in your Supabase SQL Editor to create the necessary table:
```sql
CREATE TABLE public.lab_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    test_name VARCHAR(100) NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    unit VARCHAR(20),
    report_date DATE NOT NULL,
    status VARCHAR(20) DEFAULT 'Unknown',
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### Running the App
```bash
streamlit run app.py
```

## 📝 Roadmap
- [x] CSV Multi-file Ingestion
- [x] Linear Regression for 30-day forecasting
- [x] RAG Chatbot Integration
- [ ] PDF OCR Support (via Gemini 2.0 Flash)
- [ ] Multi-user Authentication
- [ ] Color-coded status alerts (Normal/Abnormal)

## ⚠️ Disclaimer
**This application is for informational purposes only.** It does not provide medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.


