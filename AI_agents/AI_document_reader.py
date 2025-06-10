import streamlit as sl
import faiss
import numpy as np
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain.schema import Document


llm=OllamaLLM(model_name="tinyllama")

# Load embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#initialize FAISS vector database
Index=faiss.IndexFlatL2(384)#vector dimension for minilm
vector_store={}
summmary_texts=[]

#function to extract text from pdf
def extract_txt_from_pdf(uploaded_file):
    pdf_reader=PyPdf2.PdfFileReader(uploaded_file)
    txt=""
    for page in pdf_reader.pages:
        txt+=page.extractText()+"\n"
        return txt
    
#function to store txt in FAISS
def store_FAISS(txt,filename):
    global vector_store,Index
    sl.write_text("Storing documents {filename} in FAISS...")

    #split txt into chunks
    splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
    txts=splitter.split_text(txt)

    #convert txt into embedings
    vectors=embeddings.embed_documents(txts)
    vectors=np.array(vectors, dtype=np.float32)#convert to numpy array

    #store in FAISS
    vector_store[len(vector_store)]=(filename,txts)
    return "Stored documents {filename} in FAISS successfully."

#function to generate AI summary
def AI_summary(txt):
    global summmary_texts
    sl.write_text("Generating AI summary...")
    summmary_texts=llm.invoke(f"Summarize the following text:\n\n {txt}[3000]")
    return summmary_texts

#function to retrieve relevant chunks and answer questions 
def retrieve_and_ans(query):
    global vector_store,Index

    #covert query into embedding
    query_vector=np.array(embeddings.embed_query(query), dtype=np.float32).reshape(1,-1)

    #search in FAISS
    distances, indices=Index.search(query_vector,k=3)
    context=""
    for i in indices[0]:
        if i in vector_store:
            context+="".join(vector_store[i][1])+"\n\n"

        if not context:
            return ("No relevant documents found.")
        
        #ask AI to generate ans
        return llm.invoke(f"Given the context: {context} \n, answer the question: {query} \n")
    
#function to allow file download
def download_file():
    if summmary_texts:
        sl.download_button(label="Download Summary",
                           data=summmary_texts, 
                           file_name="summary.txt",
                           mime="text/plain")
    
#streamlib web ui
sl.title("Document Retrieval System")
sl.write("Upload a PDF document and ask questions on basis of it and get AI generated summary of uploaded document")

#file uploader for pdf
upload_file=sl.file_uploader("Choose a PDF file", type=["pdf"])
if upload_file:
    txt=extract_txt_from_pdf(upload_file)
    store_msg=store_FAISS(txt, upload_file.name)
    sl.write(store_msg)

    #geenerate AI summmary
    summary=AI_summary(txt)
    sl.subheader("AI generated Summary:")
    sl.write(summary)

    #enable file download for summary
    download_file()

#user input for QNA
query=sl.text_input("Ask a question")
if query:
    ans=retrieve_and_ans(query)
    sl.subheader("Answer:")
    sl.write(ans)
