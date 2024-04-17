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
from elevenlabs import play, save
# Importing Speech Recognition
import speech_recognition as sr
import time
from streamlit_player import st_player
import os
from langchain.vectorstores.pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
import csv
from langchain.docstore.document import Document 

from streamlit_mic_recorder import mic_recorder, speech_to_text
import base64
import array

#Add Keys
CLAUDE_API_KEY= os.environ['CLAUDE_API_KEY']
api_key= os.environ['CLAUDE_API_KEY']
PINECONE_API_KEY= os.environ['PINECONE_API_KEY']
REPLICATE_API_TOKEN= os.environ['REPLICATE_API_TOKEN']
OPENAI_API_KEY= os.environ["OPENAI_API_KEY"]
client= OpenAI(api_key= os.environ["OPENAI_API_KEY"])
chat= ChatOpenAI(openai_api_key= os.environ["OPENAI_API_KEY"])
ELEVEN_LABS_API_KEY= os.environ["ELEVEN_LABS_API_KEY"]
client2= ElevenLabs(api_key= os.environ["ELEVEN_LABS_API_KEY"])


# Define your custom prompt template
template = """You are Justin, a 40 year old from the Bay Area who is funny and charming.
You are given the following extracted parts of a long document and a question. 
Provide a short conversational answer and follow-up question using the extracted parts of the document. 
The answer should be less than 140 characters, formatted in one complete paragraph. Do not ask more than one question. Ask a maximum of one question.
Do not use the following words: Answer, Question, Context.
Question: {question}
=========
{context}
=========
"""
QA_PROMPT = PromptTemplate(template=template, input_variables=[
                           "question", "context"])

# Define the columns we want to embed vs which ones we want in metadata


# LLM Chain
def get_chatassistant_chain():
    #columns_to_embed = ["Question","Context"]
    #columns_to_metadata = ["Tags"]
    #documents = []
    #with open("C:\\Users\\HP\\Desktop\\JI\\RAG-Justin2.csv", newline="", encoding='utf-8-sig') as csvfile:
        #csv_reader = csv.DictReader(csvfile)
        #for i, row in enumerate(csv_reader):
            #to_metadata = {col: row[col] for col in columns_to_metadata if col in row}
            #values_to_embed = {k: row[k] for k in columns_to_embed if k in row}
            #to_embed = "\n".join(f"{k.strip()}: {v.strip()}" for k, v in values_to_embed.items())
            #newDoc = Document(page_content=to_embed, metadata=to_metadata)
            #documents.append(newDoc)
       
    #loader = CSVLoader(file_path="C:\\Users\\HP\\Desktop\\JI\\RAG-Justin2.csv", encoding="utf8")
    #documents = loader.load()

    #text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separator = "\n",length_function=len)
    #texts = text_splitter.split_documents(documents)
    
    embeddings_model = OpenAIEmbeddings()
    #vectorstore = FAISS.from_documents(texts, embeddings_model)
    vectorstore = PineconeVectorStore(index_name="justinai", embedding=embeddings_model)
    #llm = ChatOpenAI(model="ft:gpt-3.5-turbo-0125:personal::92WRQSTH", temperature=1)
    llm = ChatOpenAI(model="gpt-4-turbo", temperature=1)

    memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    chain=ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory,combine_docs_chain_kwargs={"prompt": QA_PROMPT})
    return chain

chain = get_chatassistant_chain()






video_html = """
<video controls autoplay="true" muted="true" loop="true">
<source 
            src="https://ugc-idle.s3-us-west-2.amazonaws.com/est_e103ea6195a27be25c6379fa1e36a4a9.mp4" 
            type="video/mp4" />
</video>"""


# Sidebar to select LLM
with st.sidebar:   
    st.markdown("# Chat Options")
    # model names - https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo
    #model = st.selectbox('What model would you like to use?',('gpt-4-turbo','claude-3-opus-20240229', 'llama-2-70b-chat', 'ft:gpt-3.5-turbo-0125'))
    st.markdown(video_html, unsafe_allow_html=True)
    
    text = speech_to_text(language='en', use_container_width=True, just_once=True, key='STT')
    state = st.session_state
    if 'text_received' not in state:
        state.text_received = []
   


assistant_logo = 'https://media.licdn.com/dms/image/C5603AQEsY2cRFiJCLg/profile-displayphoto-shrink_200_200/0/1517054132693?e=2147483647&v=beta&t=KeDZ8nO3IuEdVvbgrz-xCgnkauK4DISvQZfPsF0O_dQ'
# check for messages in session and create if not exists
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello, I'm Justin! 👋"}
    ]
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])




 
if text:
    state.text_received.append(text)
    user_prompt = text

    with st.chat_message("user"):
        st.markdown(user_prompt)
    with st.chat_message("assistant", avatar=assistant_logo):
        message_placeholder = st.empty()
        response = chain.invoke({"question": user_prompt})
        message_placeholder.markdown(response['answer'])

        #ElevelLabs API Call and Return
        text = str(response['answer'])
        audio = client2.generate(text=text,voice="Justin",model="eleven_turbo_v2")

        # Create single bytes object from the returned generator.
        data = b"".join(audio)

        ##send data to audio tag in HTML
        audio_base64 = base64.b64encode(data).decode('utf-8')
        audio_tag = f'<audio autoplay="true" src="data:audio/wav;base64,{audio_base64}">'     
        st.markdown(audio_tag, unsafe_allow_html=True)
        

    st.session_state.messages.append({"role": "assistant", "content": response['answer']})
    


    # Text Search Instead
#user_prompt = st.chat_input()
if user_prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    with st.chat_message("user"):
        st.markdown(user_prompt)
    with st.chat_message("assistant", avatar=assistant_logo):
        message_placeholder = st.empty()
        response = chain.invoke({"question": user_prompt})
        message_placeholder.markdown(response['answer'])

        #ElevelLabs API Call and Return
        text = str(response['answer'])
        audio = client2.generate(text=text, voice="Justin", model="eleven_turbo_v2")
        

        # Create single bytes object from the returned generator.
        data = b"".join(audio)

        ##send data to audio tag in HTML
        audio_base64 = base64.b64encode(data).decode('utf-8')
        audio_tag = f'<audio autoplay="true" src="data:audio/wav;base64,{audio_base64}">'     
        st.markdown(audio_tag, unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": response['answer']})
