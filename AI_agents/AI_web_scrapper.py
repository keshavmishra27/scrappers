import requests
from bs4  import BeautifulSoup
import streamlit as sl
from langchain_ollama import OllamaLLM

llm=OllamaLLM(model="tinyllama")

#fuction to scrap a webstite
def scrap_web(url):
    try:
        sl.write("Scraping data from: "+url)
        headers={'User-Agent': 'Mozilla/5.0'}
        response=requests.get(url, headers=headers)

        if response.status_code!=200:
            return "Failed to fetch the webpage"
        
        #extracting text from the webpage
        soup=BeautifulSoup(response.text, 'html.parser')# BeautifulSoup object
        para=soup.find_all("p")
        msg="".join([p.get_text() for p in para])

        return msg[:2000]#limiting the output to 2000 characters to stop overloading of AI model
    
    except Exception as e:
        return str(e)
    
#function for summarizing content using AI
def summary(content):
    sl.write("Summarizing the content...")
    return llm.invoke("Summarize the following text:\n {content}[1000]")

#streamlit web AI
sl.title("Web Scraper and Summarizer")
sl.write("This web app allows you to scrape content from a website and summarize it using AI.")

#user input 
url=sl.text_input("Enter the URL of the website")

if url:
    content=scrap_web(url)
    if "Failed" in content or"Error" in content:
        sl.write(content)

    else:
        summary_text=summary(content)
        sl.subheader("Summary:")
        sl.write(summary_text)