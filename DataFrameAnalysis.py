
import logging
import sys
import os
from IPython.display import Markdown, display
from llama_index.llms.ollama import Ollama
from llama_index.llms.lmstudio import LMStudio

from dotenv import load_dotenv
import pandas as pd
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.core import Settings


from llama_index.llms.openai import OpenAI

load_dotenv('.env_4_SC')
logging.basicConfig(stream=sys.stdout, level=os.environ['LOG_LEVEL'])
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

def messages_to_prompt_phi3(messages):
    prompt = ""
    system_found = False
    for message in messages:
        if message.role == "system":
            prompt += f"<|system|>\n{message.content}<|end|>\n"
            system_found = True
        elif message.role == "user":
            prompt += f"<|user|>\n{message.content}<|end|>\n"
        elif message.role == "assistant":
            prompt += f"<|assistant|>\n{message.content}<|end|>\n"
        else:
            prompt += f"<|user|>\n{message.content}<|end|>\n"

    # trailing prompt
    prompt += "<|assistant|>\n"

    if not system_found:
        prompt = (
            "<|system|>\nYou are a helpful AI assistant.<|end|>\n" + prompt
        )

    return prompt

class DataFrameAnalysis:
    def __init__(self, docpath):
        self.docpath = docpath
        if bool(os.environ['USE_LMSTUDIO'] == 'True'):
            self.llm = LMStudio(
                                base_url=os.environ["LLAMACPP_URL"], 
                                model_name=os.environ["LLAMACPP_MODEL"],  
                                temperature=0.5,      
                                max_new_tokens=200,
                                verbose=True,
                                is_chat_model=True,
                                request_timeout=600.0
                            )
        elif bool(os.environ['USE_OLLAMA'] == 'True'):
            self.llm = Ollama(model=os.environ["OLLAMA_MODEL"],
                               request_timeout=600.0,
                               base_url=os.environ["OLLAMA_URL"],
                                additional_kwargs={"num_predict": 200})
        else:
            self.llm = Settings.llm   
    
    def load_data(self):
        self.df = pd.read_csv(self.docpath)

        pd.set_option("display.max_columns", int(os.environ['Max_Output_Columns']))
        pd.set_option("display.max_rows", int(os.environ["Max_Output_Rows"]))

        self.query_engine = PandasQueryEngine(llm=self.llm,df=self.df, verbose=True,synthesize_response=True)

    def query_data(self, query):
        response = self.query_engine.query(query)
        yield str(response)

def test_csv():
    docpath = 'storage/test.csv'
    dfa = DataFrameAnalysis(docpath)
    dfa.load_data()
    query = 'Give summary of the data wit completed sentences'
    response = dfa.query_data(query)
    partialMessage = ''
    for text in response:
        partialMessage += text
    
    print(partialMessage) 

if __name__ == '__main__':
    test_csv()

