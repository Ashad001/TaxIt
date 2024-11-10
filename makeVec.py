import os
import re
import pickle
import PyPDF2
import chromadb
from tax_agent import TaxAgentModule, TaxAgent

class PDFTextExtractor:
    def __init__(self, pdf_path="tax.pdf", max_pages=200):
        self.pdf_path = pdf_path
        self.max_pages = max_pages

    def extract_text(self):
        text = ""
        with open(self.pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for count, page in enumerate(reader.pages):
                if count >= self.max_pages:
                    break
                page_text = page.extract_text() or ""
                text += page_text
        return text

class TextChunker:
    def __init__(self, max_length=1500):
        self.max_length = max_length

    def split_into_chunks(self, text):
        sentences = re.split(r'(?<=[.!?]) +', text)
        chunks = []
        current = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)
            if current_length + sentence_length > self.max_length:
                chunks.append(" ".join(current))
                current = []
                current_length = 0
            current.append(sentence)
            current_length += sentence_length

        if current:
            chunks.append(" ".join(current))
        
        return chunks

class EmbeddingManager():
    def __init__(self, embedding_path='embeddings.pkl'):
        self.embedding_path = embedding_path
        self.embeddings = self.load_embeddings()

    def load_embeddings(self):
        with open(self.embedding_path, 'rb') as f:
            embeddings = pickle.load(f)
        return embeddings

class ChromaDBHandler:
    def __init__(self, collection_name="tax_embeddings"):
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(name=collection_name)

    def add_documents(self, documents, embeddings, ids):
        print("--- Adding documents to the collection ---")
        try:
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                ids=ids
            )
        except Exception as e:
            print(str(e))
        print("--- Documents added successfully ---")

    def query(self, query_texts, n_results=3):
        return self.collection.query(
            query_texts=query_texts,
            n_results=n_results
        )

if __name__ == "__main__":
    pdf_extractor = PDFTextExtractor()
    text = pdf_extractor.extract_text()

    chunker = TextChunker(max_length=1500)
    chunks = chunker.split_into_chunks(text)

    embedding_manager = EmbeddingManager('embeddings.pkl')
    embeddings = embedding_manager.embeddings

    db_handler = ChromaDBHandler()
    ids = [f"doc_1_chunk_{i}" for i in range(len(chunks))]
    db_handler.add_documents(documents=chunks, embeddings=embeddings, ids=ids)

    query = "What is property tax?"
    # Querying the collection
    response = db_handler.query(query_texts=[query], n_results=1)
    context = []
    for result in response['documents']:
        context.append(result[0])
    context = " ".join(context)
    print(context)
    tax_agent = TaxAgentModule()
    
    # Running the TaxAgentModule
    response = tax_agent.run(question=query, context=context)
    
    print("English response:", response[0])
    print("Urdu response:", response[1])