from langchain_qdrant import RetrievalMode
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_qdrant import QdrantVectorStore
import os
import re
import PyPDF2
from qdrant_client import QdrantClient

# Load environment variables
load_dotenv()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_CLUSTER_URL = os.getenv("QDRANT_CLUSTER_URL")

qdrant_client = QdrantClient(

    url=QDRANT_CLUSTER_URL,
    api_key=QDRANT_API_KEY

)

# Initialize the embedding model
model_name = "BAAI/bge-small-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        count = 0
        for page in reader.pages:
            if count >= 200:
                break
            if page.extract_text():
                text += page.extract_text()
            else: 
                text += ""
            count += 1
    return text

def split_text_into_chunks(text, max_length=500):
    
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)
        if current_length + sentence_length > max_length:
            chunks.append(" ".join(current))
            current = []
            current_length = 0
        current.append(sentence)
        current_length += sentence_length

    if current:
        chunks.append(" ".join(current))
    
    return chunks

# Load PDF and split text into chunks
pdf_path = "tax.pdf"
text = extract_text_from_pdf(pdf_path)
chunks = split_text_into_chunks(text, max_length=1500)

qdrant = QdrantVectorStore.from_texts(
    chunks,
    embeddings,
    location=":memory:",  # Local mode with in-memory storage only
    collection_name="tax-v1",
)

found_docs = qdrant.similarity_search("tax evasion")

print(found_docs)