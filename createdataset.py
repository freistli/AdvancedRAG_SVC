
import base64
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time
import re

from dotenv import load_dotenv
from fastapi.responses import RedirectResponse
from langchain.text_splitter import MarkdownHeaderTextSplitter,MarkdownTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.callbacks import BaseCallbackHandler
from langchain_text_splitters import TextSplitter, CharacterTextSplitter
from llama_index.core import Document
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from queue import SimpleQueue
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings, AzureOpenAI
from typing import Any, Union, Dict, List
from langchain.schema import LLMResult
from threading import Thread
from azure.ai.documentintelligence.models import AnalyzeResult, AnalyzeDocumentRequest
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from fastapi import FastAPI
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.core import VectorStoreIndex, KnowledgeGraphIndex, load_index_from_storage, load_indices_from_storage, ComposableGraph, ServiceContext
from llama_index.core import Settings
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core.storage import StorageContext
import tempfile
import logging
from Environment import *

load_dotenv('.env')
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

tmpdirname = tempfile.gettempdir()
ruleFilePath = ".//rules//rules_original.pdf"

logging.info('Temporary directory ' + tmpdirname)

persist_dir=tmpdirname+"/index_cache"

os.makedirs(persist_dir, exist_ok=True)

persist_dir=persist_dir+"/"+Path(ruleFilePath).stem

os.makedirs(persist_dir, exist_ok=True)

logging.info(persist_dir)

trainFilePath = ".//rules//rules_train.pdf"

train_persist_dir=tmpdirname+"/index_cache"

os.makedirs(train_persist_dir, exist_ok=True)

train_persist_dir=train_persist_dir+"/"+Path(trainFilePath).stem

os.makedirs(train_persist_dir, exist_ok=True)

logging.info(train_persist_dir)

train_index = None
rules_index = None
composeGraph = None



load_dotenv('.env')

endpoint = os.environ['DOC_AI_BASE']
key = os.environ['DOC_AI_KEY']

"""
# Initiate Azure AI Document Intelligence to load the document. You can either specify file_path or url_path to load the document.
loader = AzureAIDocumentIntelligenceLoader(file_path=".//rules//English.docx", api_key=key, api_endpoint=endpoint,
                                           api_model="prebuilt-layout")
docs = loader.load()
"""

import tiktoken
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
token_counter = TokenCountingHandler(
    tokenizer=tiktoken.encoding_for_model(os.environ['AZURE_OPENAI_Deployment']).encode
)

#langchain_openai
client = AzureChatOpenAI(
    openai_api_version="2023-12-01-preview",
    azure_deployment=os.environ['AZURE_OPENAI_Deployment'],
    streaming=True)


llama_index_llm = AzureChatOpenAI(azure_deployment=os.environ['AZURE_OPENAI_Deployment'],
                           azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
                           api_key=os.environ['AZURE_OPENAI_API_KEY'])


llama_index_embed_model = LangchainEmbedding(AzureOpenAIEmbeddings(chunk_size=1000,
                                                             model=Embedding_Mode,
                                                             deployment=Embedding_Mode,
                                                             openai_api_key=os.environ['AZURE_OPENAI_API_KEY']))
 
Settings.llm = llama_index_llm
Settings.embed_model = llama_index_embed_model
Settings.node_parser = SemanticSplitterNodeParser(buffer_size=1, breakpoint_percentile_threshold=95,embed_model=llama_index_embed_model)
Settings.callback_manager = CallbackManager([token_counter])


docClient = DocumentIntelligenceClient(endpoint, AzureKeyCredential(key))

#setup the storage context

graph_store = SimpleGraphStore()

use_storage = True
batch_size = 10

Draft = ".//rules//train.txt"

with open(Draft, 'r') as file:
        file_content = file.read()

        textSplitter = CharacterTextSplitter( chunk_size=200, separator="====", is_separator_regex=False)

        to_be_proofread_content_list = textSplitter.split_text(file_content)        

print("Number of chunks: ", len(to_be_proofread_content_list))
for i in range(len(to_be_proofread_content_list)):
    print("Chunk " + str(i) +"\r\n")
    clean_paragraph = re.sub(r'\n\s*\n', '\n', to_be_proofread_content_list[i])
    print("\r\n"+clean_paragraph)
    to_be_proofread_content_list[i] = clean_paragraph



# The system message
system_message = {
    "role": "system",
    "content": "You are an experienced Japanese Proofreader. You are tasked with proofreading the following text. Please correct any errors you find and provide a reason for each correction."
}

# Convert each string in the array to a set of messages
json_objects = ""
i = 0
for string in to_be_proofread_content_list:
    print("Count: " + str(i)) 
    # Split the string into the original text, the correction, and the reason
    original_text, correction, reason = string.split("\n")

    # Create the user and assistant messages
    user_message = {"role": "user", "content": original_text}
    assistant_message = {"role": "assistant", "content": correction + "\n" + reason}

    # Create the JSON object
    json_object = {"messages": [system_message, user_message, assistant_message]}

    json_string = json.dumps(json_object,ensure_ascii=True,indent=0).replace('\n', '').replace('\r', '').replace('\t', '')

    # Add the JSON object to the list
    json_objects = json_objects + json_string + "\n"
    i += 1

# Write the JSON objects to a file
with open("proofreading_dataset_v2.jsonl", "w") as file:
    file.write(json_objects)