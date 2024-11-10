import os
import re
import pickle
import PyPDF2
import chromadb
import streamlit as st
from chromadb.errors import UniqueConstraintError
from dspy import InputField, OutputField, Signature, Module, ChainOfThought
import dspy
import os
from dotenv import load_dotenv

load_dotenv()

lm = dspy.GROQ(model="llama-3.2-11b-vision-preview", api_key=os.getenv("GROQ_API_KEY"),  max_tokens=4096)

dspy.settings.configure(lm=lm)


chromadb.api.client.SharedSystemClient.clear_system_cache()

# if "start" not in st.session_state:
#     chromadb.api.client.SharedSystemClient.clear_system_cache()
#     st.session_state.start = True

class PDFTextExtractor:
    def __init__(self , max_pages=200):
      
        self.max_pages = max_pages

    def extract_text(self, pdf_path):
        text = ""
        with open(pdf_path, 'rb') as file:
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
        
        try:
            self.collection = self.client.create_collection(name=collection_name)
        except UniqueConstraintError:
            print(f"Collection {collection_name} already exists. Using the existing collection.")
            self.collection = self.client.get_collection(name=collection_name)

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

    def query(self, query_texts, n_results=2):
        
        return self.collection.query(
            query_texts=query_texts,
            n_results=n_results
        )
        

# def query_database(db_client, question, n_results=2):
#     response = db_client.query(query_texts=[question], n_results=n_results)
#     return " ".join(result[0] for result in response['documents'])

class TaxAgent(Signature):
    """
    You are an expert Pakistani Tax Agent with comprehensive knowledge of tax regulations and financial reporting. 
    Your role is to provide accurate, and professional answers based on the Pakistan's Tax Ordinance Context provided. 
    When responding, structure your answer in clear, step-by-step explanations to ensure clarity and completeness. 
    Make sure to follow the given keywords - tags, these will help you to generate pointers for each specific tag.
    """
    question = InputField(type=str, desc="A detailed question related to tax data that needs answering.")
    context = InputField(type=str, desc="Detailed tax data or context to support the answer.")
    Tags = InputField(type=str, desc = "These are the keywords you should revolve your pointers around and relate to the query.")
    answer_eng = OutputField(type=str, desc="The professional answer in English, structured step-by-step as requested.")

class TagGenerator(Signature):
   
    """
    You are an expert Prompt Engineer with comprehensive knowledge of how keyword specific tags work in LLMs 
    Your role is to provide accurate, and clear KEYWORDS from the given Pakistan's Tax Ordinance context and a native user's query  
    When responding, make sure you make a list of tags, with heading as 'Keywords to answer on:' and that these words help
    LLM to generate comprehensive answers, related to the query.
    """
    question = InputField(type=str, desc="A question related to tax laws and data asked by a native user.")
    context = InputField(type=str, desc="Detailed Pakistan's Tax Ordinance Context to support the answer.")
    tags = OutputField(type=str, desc="The keyword specific tag list relating to the query, generated in English.")

class English2Urdu(Signature):
    """
    Translate the provided English text into professional and accurate Urdu.
    ---
    Input:
    - English text
    
    Output:
    - Urdu translation of the text, maintaining the original meaning, tone, and professionalism.
    """
    english_text = InputField(type=str, desc="English text to be translated.")
    urdu_text = OutputField(type=str, desc="Urdu translation of the text with accurate meaning and tone. Give only the urdu translation not the original")
    
class TaxAgentModule(Module):
    def __init__(self):
        super().__init__()
        self.tagGenerator = ChainOfThought(TagGenerator)
        self.tax_agent = ChainOfThought(TaxAgent)
        self.language_converter = ChainOfThought(English2Urdu)

    def run(self, question: str, context: str):
        print("Running TaxAgentModule...")

        # Run the tax agent to get the professional tax summary

        resp = self.tagGenerator(question=question, context=context)

        agent_response = self.tax_agent(question=question, context=context, Tags=resp.tags)
        
        # Translate the English summary into Urdu
        urdu_response = self.language_converter(english_text=agent_response.answer_eng)
        
        # Return both English and Urdu answers
        return agent_response.answer_eng, urdu_response.urdu_text


if __name__ == "__main__":
    
    
    pdf_extractor = PDFTextExtractor()
    text1 = pdf_extractor.extract_text(pdf_path='tax.pdf')
    
    text2 = pdf_extractor.extract_text(pdf_path='tax2.pdf')
    text = 'Text Slabs and Computing Tax Rate and Amount: '+ text2 + 'Tax Laws and ordinance:' + text1
    chunker = TextChunker(max_length=1500)
    chunks = chunker.split_into_chunks(text)

    embedding_manager = EmbeddingManager('embeddings.pkl')
    embeddings = embedding_manager.embeddings

    db_handler = ChromaDBHandler()
    ids = [f"doc_1_chunk_{i}" for i in range(len(chunks))]
    db_handler.add_documents(documents=chunks, embeddings=embeddings, ids=ids)

    query = "What is an exemption in tax and how do you get one ?"
    # Querying the collection
    response = db_handler.query(query_texts=[query], n_results=2)
    context = []
    for result in response['documents']:
        context.append(result[0])
    context = " ".join(context)
   
    # print(context)
    tax_agent = TaxAgentModule()
    
    # Running the TaxAgentModule
    
    
    # response = tax_agent.run(question=query, context=context)
    # print("English response:", response[0])
    # print("Urdu response:", response[1])
    
    st.title("Smart Tax Assistant - Pakistan")
    st.title("Be free from your Tax Worries")
    st.write("Ask any tax-related questions and get answers based on Pakistan's Tax laws/ordinance and up-to-date computational slabs. ")

    # User input
    question = st.text_input("Enter your question:", "")

    # Process question and display answer
    if st.button("Get Answer") and question:
        context = []
        response =  db_handler.query(query_texts=[question])
        for result in response['documents']:
            context.append(result[0])
        context = " ".join(context)
        if context:
            response = tax_agent.run(question=question, context=context)
            st.write("**Answer (English):**", response[0])
            st.write("**Answer (Urdu):**", response[1])
        else:
            st.write("No relevant information found in the database.")