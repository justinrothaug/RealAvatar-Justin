from openai import OpenAI
import streamlit as st
import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_community.callbacks import get_openai_callback
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
# Importing Eleven Labs
from elevenlabs.client import ElevenLabs
from elevenlabs import play
# Importing Speech Recognition
import speech_recognition as sr
from streamlit_player import st_player
from streamlit_mic_recorder import mic_recorder, speech_to_text
import time
import os
from langchain.vectorstores.pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
st.set_page_config(page_title="Justin AI")

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]    
client = OpenAI(api_key= st.secrets["openai_key"])
chat = ChatOpenAI(openai_api_key=st.secrets["openai_key"])
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"] 
os.environ['PINECONE_API_KEY'] = st.secrets["PINECONE_API_KEY"] 

# Define your custom prompt template
template = """You are Justin, a 40 year old from the Bay Area who is funny and charming.
You are given the following extracted parts of a long document and a question. 
Provide a short conversational answer and follow-up question using the extracted parts of the document.
The answer should be less than 140 characters, formatted in one complete paragraph.
Do not use the following words: Answer, Question, Context.
Question: {question}
=========
{context}
=========
"""
QA_PROMPT = PromptTemplate(template=template, input_variables=[
                           "question", "context"])



def get_chatassistant_chain():
    #loader = CSVLoader(file_path="RAG-Justin2.csv", encoding="utf8")
    #documents = loader.load()
    #text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    #texts = text_splitter.split_documents(documents)
    
    embeddings_model = OpenAIEmbeddings(openai_api_key=st.secrets["openai_key"])
    #vectorstore = FAISS.from_documents(texts, embeddings_model)
    vectorstore = PineconeVectorStore(index_name="justinai", embedding=embeddings_model)
    llm = ChatOpenAI(model="ft:gpt-3.5-turbo-0125:personal::92WRQSTH", temperature=1)
    memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    chain=ConversationalRetrievalChain.from_llm(llm=ChatOpenAI(), retriever=vectorstore.as_retriever(), memory=memory,combine_docs_chain_kwargs={"prompt": QA_PROMPT})
    return chain

chain = get_chatassistant_chain()

assistant_logo = 'https://pbs.twimg.com/profile_images/733174243714682880/oyG30NEH_400x400.jpg'

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "ft:gpt-3.5-turbo-0125:personal::92WRQSTH"

# check for messages in session and create if not exists
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello, I'm Justin! 👋"}
    ]
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

#Record Audio Input and convert to Text
state = st.session_state
if 'text_received' not in state:
    state.text_received = []
text = speech_to_text(language='en', use_container_width=True, just_once=True, key='STT')
if text:
    state.text_received.append(text)
    user_prompt = text
  
if user_prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant", avatar=assistant_logo):
        message_placeholder = st.empty()
        response = chain.invoke({"question": user_prompt})
        message_placeholder.markdown(response['answer'])
        print (chain)

        #ElevelLabs API Call and Return
        #text = str(response['answer'])
        #audio = client2.generate(text=text,voice="Justin",model="eleven_multilingual_v2")
        #play(audio)
        #stream(audio)
    st.session_state.messages.append({"role": "assistant", "content": response['answer']})
