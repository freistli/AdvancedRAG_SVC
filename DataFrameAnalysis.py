
import logging
import sys
import os
from IPython.display import Markdown, display


from dotenv import load_dotenv
import pandas as pd
from llama_index.experimental.query_engine import PandasQueryEngine

load_dotenv('.env_4_SC')
logging.basicConfig(stream=sys.stdout, level=os.environ['LOG_LEVEL'])
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

class DataFrameAnalysis:
    def __init__(self, docpath):
        self.docpath = docpath
    
    def load_data(self):
        self.df = pd.read_csv(self.docpath)
        self.query_engine = PandasQueryEngine(df=self.df, verbose=True,synthesize_response=True)

    def query_data(self, query):
        response = self.query_engine.query(query)
        yield str(response)

def test_csv():
    docpath = 'storage/test.csv'
    dfa = DataFrameAnalysis(docpath)
    dfa.load_data()
    query = 'how many "SMS Sent" happened?'
    response = dfa.query_data(query)
    partialMessage = ''
    for text in response:
        partialMessage += text

    print(partialMessage) 

if __name__ == '__main__':
    test_csv()

