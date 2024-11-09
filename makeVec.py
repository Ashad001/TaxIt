# from langchain_qdrant import RetrievalMode
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.document_loaders import PyPDFLoader
from dotenv import load_dotenv
# from langchain.embeddings import HuggingFaceBgeEmbeddings
import pickle
import os
import re
import PyPDF2
import chromadb
import sys

# __import__("pysqlite3")
# sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")


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

with open('embeddings.pkl', 'rb') as f:
    embeddings = pickle.load(f)  


client = chromadb.Client()

collection = client.create_collection(name="tax_embeddings")

ids = [f"doc_1_chunk_{i}" for i in range(len(chunks))]

collection.add(
    documents=chunks,
    embeddings=embeddings,
    ids=ids

)


context = collection.query(
    query_texts=['What is property tax'],
    n_results=3  
)

print(context)
