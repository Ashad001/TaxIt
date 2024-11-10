from dspy import InputField, OutputField, Signature, Module, ChainOfThought
import dspy
import os
from dotenv import load_dotenv
load_dotenv()

lm = dspy.GROQ(model="llama-3.2-11b-vision-preview", api_key=os.getenv("GROQ_API_KEY"))
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

class LanguageConverter(Signature):
    """
    You are a skilled Language Converter, fluent in both English and Urdu. 
    Your role is to accurately translate tax summaries from English to Urdu, ensuring that the meaning and nuances are retained. 
    Your translation should be formal and professional to match the tone of tax documentation.
    """
    summary = InputField(type=str, desc="The tax-related summary in English that needs translating to Urdu.")
    answer_urdu = OutputField(type=str, desc="The professionally translated summary in Urdu, retaining the original meaning and tone.")
   
class TaxAgentModule(Module):
    def __init__(self):
        super().__init__()
        self.tax_agent = ChainOfThought(TaxAgent)
        self.language_converter = ChainOfThought(LanguageConverter)

    def run(self, question: str, context: str):
        print("Running TaxAgentModule...")

        # Run the tax agent to get the professional tax summary
        agent_response = self.tax_agent(question=question, context=context)
        
        # Translate the English summary into Urdu
        urdu_response = self.language_converter(summary=agent_response.answer_eng)
        
        # Return both English and Urdu answers
        return agent_response.answer_eng, urdu_response.answer_urdu
