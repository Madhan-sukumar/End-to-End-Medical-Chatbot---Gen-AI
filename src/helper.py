from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters  import RecursiveCharacterTextSplitter  
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

#Extract Data from the PDF File
def load_pdf_file(data):
    loader = DirectoryLoader(
        data,
        glob = '*.pdf',
        loader_cls=PyPDFLoader
    )

    documents = loader.load()
    return documents

#split the Data into Text Chunks
def text_splits(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap = 20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

#downloading embedding model from hugging face
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs = {'device': 'cpu'})
    return embeddings

