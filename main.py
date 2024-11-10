from dspy import InputField, OutputField, Signature, Module, ChainOfThought
from makeVec import *


class TaxAgent(Signature):
    """
    You are a professional Tax Agent. You have been hired by a company to summarize their tax data.
    You'll get a question from the user, along with a context to answer from, and you will give a meaningful answer in a stepwise manner in both English and Urdu.
    """
    question = InputField(type=str, desc="The question you want to ask.")
    context = InputField(type=str, desc="The context to answer the question from.")
    answer_eng = OutputField(type=str, desc="The answer to the question.")
    answer_urdu = OutputField(type=str, desc="The answer to the question in Urdu.")


class ContextHandler:
    def __init__(self) -> None:
        pdf_path = "tax.pdf"  
        pdf_extractor = PDFTextExtractor(pdf_path)
        text = pdf_extractor.extract_text()
        
        print("Extracted text from PDF.")
        

        chunker = TextChunker(max_length=1500)
        chunks = chunker.split_into_chunks(text)

        print("Text split into chunks.")

        embedding_manager = EmbeddingManager('embeddings.pkl')
        embeddings = embedding_manager.embeddings

        print("Embeddings ")
        self.db_handler = ChromaDBHandler("tax-collection")
        
        print("DB handler")
        print(len(chunks))
        ids = [f"doc_chunk_{i}" for i in range(len(chunks))]
        print("Chunks length:", len(chunks))
        print("Embeddings length:", len(embeddings))
        print("IDs length:", len(ids))

        try:
            self.db_handler.add_documents(documents=chunks, embeddings=embeddings, ids=ids)
            print("Documents added successfully.")
        except Exception as e:
            print("Error while adding documents:", e)

        print("docs") #! THIS AINT RUNNING; PROBLEM WITH UPPER LINE

    def query(self, question: str):
        print("Executing query in ContextHandler...")
        results = self.db_handler.query(query_texts=[question], n_results=2)
        print("Query completed.")
        return results


class TaxAgentModule(Module):
    def __init__(self, context_handler: ContextHandler):
        super().__init__()
        self.tax_agent = ChainOfThought(TaxAgent)
        self.db_handler = context_handler.db_handler  # Link the shared db_handler

    def run(self, question: str):
        print("Running TaxAgentModule...")
        results = self.db_handler.query(query_texts=[question], n_results=2)
        
        # Assume results[0]['documents'] contains the best-matching text
        if results and 'documents' in results[0]:
            context = results[0]['documents'][0]
            print("Context obtained:", context)
            
            agent_response = self.tax_agent(question=question, context=str(context))
            return agent_response.answer_eng, agent_response.answer_urdu
        else:
            print("No results found.")
            return "No answer found.", "کوئی جواب نہیں ملا"


# Usage
context_handler = ContextHandler()
agent = TaxAgentModule(context_handler)
query = "What is property tax?"

print("Initiating query and agent run...")
eng_answer, urdu_answer = agent.run(query)
print(f"English Answer: {eng_answer}")
print("----")
print(f"Urdu Answer: {urdu_answer}")
