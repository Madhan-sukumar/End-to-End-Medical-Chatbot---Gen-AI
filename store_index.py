"""
Executed only for very first time to create the index in Pinecone
and store the embeddings in the Pinecone
"""

from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore #Embed Each chunk and upsert the embeddings into Pincone Index

#Custom Module
from src.helper import load_pdf_file, text_splits, download_hugging_face_embeddings

load_dotenv()

#API KEYS
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY

#Loading the data
extracted_data = load_pdf_file(data='Data/')

#text chunking
text_chunks = text_splits(extracted_data)

#downloading the embedding model
embeddings = download_hugging_face_embeddings()

# Creating the index for very first time
index_name = "medicalbot"
pc = Pinecone(api_key=PINECONE_API_KEY)
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        vector_type="dense",
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ),
        deletion_protection="disabled",
        tags={
            "environment": "development"
        }
    )

#Embed Each chunk and upsert the embeddings into Pincone Index
docsearch = PineconeVectorStore.from_documents(
    documents= text_chunks, #chunked text documents
    index_name = index_name, #index name in Pinecone
    embedding= embeddings #embeddings
)