from dspy import InputField, OutputField, Signature, Module, ChainOfThought
import dspy
import os
from dotenv import load_dotenv
load_dotenv()

lm = dspy.GROQ(model="llama-3.2-11b-vision-preview", api_key=os.getenv("GROQ_API_KEY"),  max_tokens=4096)

dspy.settings.configure(lm=lm)

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
    urdu_text = OutputField(type=str, desc="Urdu translation of the text with accurate meaning and tone.")
    
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
