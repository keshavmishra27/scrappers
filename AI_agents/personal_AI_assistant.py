import speech_recognition as sr
import pyttsx3
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

#load AI model
llm=OllamaLLM(model='tinyllama')

#initailize memory( langchain v1+)
chat_history=ChatMessageHistory()

#initailize speech recognition
engine=pyttsx3.init()#initialize pyttsx3 engine
engine.setProperty('rate', 150) # set speaking speed

#speech recognition
recogniozer=sr.Recognizer()

#function to speak
def speak(text):
    engine.say(text)
    engine.runAndWait()

#function to listen
def listen():
    with sr.Microphone() as source:
        print("Listening...")
        recogniozer.adjust_for_ambient_noise(source)
        audio=recogniozer.listen(source)

    try:
        query=recogniozer.recognize_google(audio)
        print("You said: ",query)
        return query.lower()
    
    except sr.UnknownValueError:
        print("Could not understand audio")
        return None
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition API; {0}".format(e))
        return None
    
#AI chat prompt
prompt=PromptTemplate(input_variables=["chat_history","questions"],
                      template="previous converastion:{chat_history}\n Uer:{questions}\n AI:" )

#functoin to process AI response
def run_chain(question,chat_history=chat_history) -> str:
    #retrieve past chat manually
    chat_history="\n".join([f"{msg.type.capitalize()}:{msg.content}" for msg in chat_history.messages])

    #run AI response generation
    response=llm.invoke(prompt.format(chat_history=chat_history,questions=question))

    #store new user and new respone in AI
    chat_history.add_user_message(question)#add new user message
    chat_history.add_ai_message(response)#add new AI message

    return response

#main loop
print("hello! I am your AI assistant! How can I help you today")
while True:
    query=listen()
    if "stop" or "exit" in query:
        speak("Goodbye! Have a great day!")

    if query:
        response = run_chain(query)
        print(f"\n AI response: {response}")
        speak(response)
