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

