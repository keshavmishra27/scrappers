import streamlit as sl
import speech_recognition as sr
import pyttsx3
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

llm=OllamaLLM(model='tinyllama')

if "chat_history" not in sl.session_state:
    sl.session_state.chat_history = ChatMessageHistory()

engine=pyttsx3.init()
engine.setProperty("rate",160)

recognizer=sr.Recognizer()

def speak(msg):
    engine.say(msg)
    engine.runAndWait()

def listen():
    with sr.Microphone() as src:
        sl.write("Say something...")
        recognizer.adjust_for_ambient_noise(src)
        audio=recognizer.listen(src)

    try:
        query=recognizer.recognize_google(audio)
        sl.write("You said: " + query)
        return query.lower()

    except sr.UnknownValueError:
        sl.write("Could not understand audio")
        return ""

    except sr.RequestError:
        sl.write("sppech recognition unavailable")
        return ""
    
#define AI chat prompt
prompt=PromptTemplate(input_variables=["chat_history","question"],
                      template="previous conversation:{chat_history}\n user: {question}\n AI")

#function for AI reesponses
def run_chain(question):
    chat_history_msg="\n".join([f"{msg.type.capitalize()}:{msg.content}" for msg in sl.session_state.chat_history.messages])
    response=llm.invoke(prompt.format(chat_history=chat_history_msg, question=question))

    sl.session_state.chat_history.add_user_message(question)
    sl.session_state.chat_history.add_ai_message(response)


    return response

#streamlit web_ui
sl.title("AI voice assistant Chatbot")
sl.write("click the btn below to start talking with chatbot")

#btn to record voice input
if sl.button("Start listening"):
    user_query=listen()
    if user_query:
        ai_response=run_chain(user_query)
        sl.write(user_query)
        sl.write(ai_response)
        speak(ai_response)

#display full chat history
sl.subheader("Full chat history:")
for msg in sl.session_state.chat_history.messages:
    sl.write(f"{msg.type.capitalize()}: {msg.content}")



