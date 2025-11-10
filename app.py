import os
from flask import Flask, render_template, jsonify, request
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.prompts import MessagesPlaceholder # The MessagesPlaceholder is part of the prompt along with user input and user message
from langchain_classic.chains import create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage

from dotenv import load_dotenv

#Custome Module
from src.prompt import *
from src.helper import download_hugging_face_embeddings

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

#HuggingFace Model
embeddings = download_hugging_face_embeddings()

#Chat history to save chats 
chat_history = []

# Retriever to retrive the vectors
index_name = "medicalbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

#ChatBot
chatModel = ChatOpenAI(model="gpt-4o", temperature=0.4, max_tokens=500)

prompt_search_query = ChatPromptTemplate.from_messages([
MessagesPlaceholder(variable_name="chat_history"),
("user","{input}"),
("user","Given the above conversation, generate a search query to look up to get information relevant to the conversation")
])

# Retriever Chain
retriever_chain = create_history_aware_retriever(chatModel, retriever, prompt_search_query)

# Prompt To Get Response From LLM Based on Chat History
prompt_get_answer = ChatPromptTemplate.from_messages([
("system", system_prompt ),
MessagesPlaceholder(variable_name="chat_history"),
("user","{input}"),
])

## Document Chain
document_chain=create_stuff_documents_chain(chatModel,prompt_get_answer)

# Conversational Retrieval Chain
retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    user_input = msg
    print(input)
    response = retrieval_chain.invoke({
    "chat_history":chat_history,
    "input":user_input
    })

    if response:
        user_message = HumanMessage(content=user_input)
        assistant_message = AIMessage(content=response['answer'])
        chat_history.append(user_message)
        chat_history.append(assistant_message)
        print("Response : ", response["answer"])
    return str(response["answer"])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)