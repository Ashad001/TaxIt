import PyPDF2
import re
from langchain_huggingface import HuggingFaceEmbeddings
import faiss
import numpy as np
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import uuid
import os

load_dotenv()


QDRANT_API_KEY = os.getenv("API")
QDRANT_CLUSTER_URL = os.getenv("QDRANT_CLUSTER_URL")


qdrant_client = QdrantClient(

    url=QDRANT_CLUSTER_URL,
    api_key=QDRANT_API_KEY

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

def store_embeddings(chunks, embeddings, client):
    points = []
    for i, embedding in enumerate(embeddings):
        point_id = str(uuid.uuid4())  
        points.append({
            "id": point_id,
            "vector": embedding.tolist(),
            "payload": {"text": chunks[i]} 
        })
    
    client.upsert(
        collection_name='tax',
        points=points
    )


def querying(index, query, model, chunk_dict, top_k=5):
    
    query_embedding = model.encode([query], convert_to_tensor=False)
    distances, indices = index.search(np.array(query_embedding).astype('float32'), top_k)
    
    results = []
    for i in indices[0]:
        if i in chunk_dict:
            results.append(chunk_dict[i])
    return results



pdf_path = "tax.pdf"
extracted_text = extract_text_from_pdf(pdf_path)
print(extracted_text[:10])
chunks = split_text_into_chunks(extracted_text,450)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print(embeddings[0].shape)
# embeddings_np = np.array(embeddings)

qdrant_client.recreate_collection(
    collection_name='tax',
    vector_size=384,  #vec size acc to embeddings lagana
    distance="Cosine"  
)

index, chunk_dict = vecDB(chunks, embeddings_np)




 