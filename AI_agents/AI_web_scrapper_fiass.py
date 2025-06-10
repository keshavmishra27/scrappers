import requests
from bs4 import BeautifulSoup
import streamlit as sl
import faiss #facebook ai similarity search library
import numpy as np
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain.schema import document

llm=OllamaLLM(model="tinyllama")

#load huggingface embeddings
embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")#replace with your preferred model

#initialize FAISS vector  database
index=faiss.IndexFlatL2(384)#vector dimension for minilmLM-L6-v2
vector_store={}

#function to scrap website
def scrap_web(url):
    try:
        sl.write(f"scrapping data from {url}")
        headers={"User-Agent": "Mozilla/5.0"}
        response=requests.get(url, headers=headers) 

        if response.status_code!=200:
            return "failed to fetch {url}!!!"
        
        #extract text from the page
        soup=BeautifulSoup(response.text, 'html.parser')
        para=soup.find_all('p')
        text=' '.join([p.get_text() for p in para])

        return text[:5000]
    
    except Exception as e:
        return f"Error occurred while scraping {url}: {str(e)}"
    
#function to store data in FAISS
def storage(text,url):
    global index, vector_store
    sl.write(f"Storing data in FAISS for {url}")

    #split text into chunks
    splitter=CharacterTextSplitter(chunk_size=500,chunk_overlap=100)
    chunks=splitter.split_text(text)

    #convert text to embeddings
    vectors=embeddings.embed_documents(chunks)
    vectors=np.array(vectors,dtype=np.float32)

    #store in FAISS
    index.add(vectors)
    vector_store[len(vector_store)]=(url, chunks)

    return "Data stored successfully in FAISS"

#function to retreive relevant chunks and answer questions
def retrive_and_ans(query):
    global index, vector_store
    sl.write(f"Retrieving relevant chunks for {query}")

    #convert query into embessings
    query_vector=np.array(embeddings.embed_query(query), dtype=np.float32).reshape(1, -1)

    #search FAISS
    D,I=index.search(query_vector, k=3)#reteieve 3 most relevant chunks
    contexts=""
    for idx in I[0]:
        if idx in vector_store:
            contexts+="".join(vector_store[idx][1])+"\n\n"

        if not contexts:
            return "No relevant chunks found"
        
        #ask AI to generate answers
        return llm.invoke(f"based on the following context, answer the question: {query}", input=contexts)
    
#streamlit web UI
sl.title("Web Scraper & AI Question Answering")
sl.write("Enter the URL of the website to scrape data")
url=sl.text_input("Enter the URL of the website to scrape data", "")
if url:
    text=scrap_web(url)
    if "failed" or "Error" in text:
        sl.write(text)
    else:
        store_msg=storage(text, url)

#user input for QNA
query=sl.text_input("Enter the question you want to answer", "")
if query:
    ans=retrive_and_ans(query)
    sl.subheader("Answer")
    sl.write(f"Answer: {ans}")
