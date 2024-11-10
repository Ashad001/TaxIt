from dspy import InputField, OutputField, Signature, Module, ChainOfThought
import dspy
import os
from dotenv import load_dotenv
load_dotenv()

lm = dspy.GROQ(model="llama-3.2-11b-vision-preview", api_key=os.getenv("GROQ_API_KEY"),  max_tokens=4096)

dspy.settings.configure(lm=lm)

class TaxAgent(Signature):
    """
    You are an expert Tax Agent with comprehensive knowledge of tax regulations and financial reporting. 
    Your role is to provide concise, accurate, and professional answers based on the specific tax data provided. 
    When responding, structure your answer in clear, step-by-step explanations to ensure clarity and completeness.
    """
    question = InputField(type=str, desc="A detailed question related to tax data that needs answering.")
    context = InputField(type=str, desc="Detailed tax data or context to support the answer.")
    answer_eng = OutputField(type=str, desc="The professional answer in English, structured step-by-step as requested.")

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
    urdu_text = OutputField(type=str, desc="Urdu translation of the text with accurate meaning and tone.")
    
class TaxAgentModule(Module):
    def __init__(self):
        super().__init__()
        self.tax_agent = ChainOfThought(TaxAgent)
        self.language_converter = ChainOfThought(English2Urdu)

    def run(self, question: str, context: str):
        print("Running TaxAgentModule...")

        # Run the tax agent to get the professional tax summary
        agent_response = self.tax_agent(question=question, context=context)
        
        # Translate the English summary into Urdu
        urdu_response = self.language_converter(english_text=agent_response.answer_eng)
        
        # Return both English and Urdu answers
        return agent_response.answer_eng, urdu_response.urdu_text
