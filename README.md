# Document & Web Scrapers
<br>
A small collection of Python scripts and Streamlit apps that scrape or ingest documents, create embeddings, store/retrieve them with FAISS, and generate answers and summaries using an Ollama LLM. This README explains what each script does, how to install and run the project, common fixes, and suggested improvements.
<br>
# Features
<br>
PDF → text extraction and Streamlit UI for uploading PDFs

Website scraping with BeautifulSoup + Streamlit summarizer

FAISS vector store for semantic retrieval (with Hugging Face sentence-transformer embeddings)

Console and Streamlit voice assistants (speech recognition + TTS) using Ollama as LLM

Small demo pipelines for:<pre><code>scrape -> embed -> store -> retrieve -> answer</code></pre>
<br>
# Repository layout
<br>
<pre><code>
README.md
requirements.txt
.env.example
scripts/
├─ streamlit_doc_reader.py # PDF upload -> store -> summary
├─ streamlit_web_summarizer.py # scrape URL -> summarize
├─ streamlit_faiss_scraper.py # scrape -> embed -> FAISS -> QA
├─ voice_assistant_console.py # console voice assistant
└─ voice_assistant_streamlit.py # Streamlit voice assistant UI
</code></pre>
<br>
# Prerequisites
<br>
Python 3.8+

pip (or pipx / venv) for installing dependencies

(Optional) faiss-gpu if you want GPU acceleration — otherwise use faiss-cpu

Ollama running locally or accessible from the machine if using langchain_ollama (see notes below)
<br>

# Quick install
<br>
Create and activate a venv:
<pre><code>
python -m venv .venv
source .venv/bin/activate # Windows: .venv\Scripts\activate
</code></pre>
Install dependencies:
<pre><code>
pip install -r requirements.txt
</code></pre>

# How to run
Streamlit PDF reader
<pre><code>
streamlit run scripts/streamlit_doc_reader.py
</code></pre>
Upload a PDF in the UI and the app will extract text, store embeddings (FAISS), and produce a summary.

Streamlit web summarizer (scrapes a URL)
<pre><code>
streamlit run scripts/streamlit_web_summarizer.py
</code></pre>
enter a URL and the page will be scraped and summarized.

Streamlit FAISS scraper + QA
<pre><code>
streamlit run scripts/streamlit_faiss_scraper.py
</code></pre>
This app scrapes a URL, indexes chunks into FAISS, and provides a simple Q&A box that searches the index and asks the LLM for an answer.

Console voice assistant
<pre><code>
  python scripts/voice_assistant_console.py
</code></pre>
This listens via microphone (Google Speech API used via speech_recognition) and replies using pyttsx3 TTS.

Streamlit voice assistant
<pre><code>
  streamlit run scripts/voice_assistant_streamlit.py
</code></pre>
Streamlit version shows the chat history and triggers audio capture in the server environment (be careful — microphone access is server-side).
