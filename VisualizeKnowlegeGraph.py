
import base64
import json
import os
from pathlib import Path
import sys

from dotenv import load_dotenv
from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter,MarkdownTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.callbacks import BaseCallbackHandler
from langchain_experimental.text_splitter import SemanticChunker
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
import networkx as nx
from pyvis.network import Network

from matplotlib import pyplot as plt

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

tmpdirname = tempfile.gettempdir()
ruleFilePath = ".//rules//rules_original.pdf"

logging.info('Temporary directory ' + tmpdirname)

persist_dir=tmpdirname+"/proofreading_rules"

os.makedirs(persist_dir, exist_ok=True)

persist_dir=persist_dir+"/"+Path(ruleFilePath).stem

os.makedirs(persist_dir, exist_ok=True)

logging.info(persist_dir)

trainFilePath = ".//rules//rules_train.pdf"

train_persist_dir=tmpdirname+"/proofreading_rules"

os.makedirs(train_persist_dir, exist_ok=True)

train_persist_dir=train_persist_dir+"/"+Path(trainFilePath).stem

os.makedirs(train_persist_dir, exist_ok=True)

logging.info(train_persist_dir)

load_dotenv('.env_4_VKG')

endpoint = os.environ['DOC_AI_BASE']
key = os.environ['DOC_AI_KEY']

llama_index_llm = AzureChatOpenAI(azure_deployment=os.environ['AZURE_OPENAI_Deployment'],
                           azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
                           api_key=os.environ['AZURE_OPENAI_API_KEY'])


llama_index_embed_model = LangchainEmbedding(AzureOpenAIEmbeddings(chunk_size=1000,
                                                             model="text-embedding-ada-002",
                                                             deployment="text-embedding-ada-002",
                                                             openai_api_key=os.environ['AZURE_OPENAI_API_KEY']))
 
Settings.llm = llama_index_llm
Settings.embed_model = llama_index_embed_model
Settings.node_parser = SemanticSplitterNodeParser(buffer_size=1, breakpoint_percentile_threshold=95,embed_model=llama_index_embed_model)

graph_store = SimpleGraphStore()

use_storage = True
batch_size = 10

if os.path.exists(train_persist_dir+"/docstore.json") and use_storage:
    train_storage_context = StorageContext.from_defaults(persist_dir=train_persist_dir)
    train_index = load_index_from_storage(train_storage_context)
    logging.info("Loaded train index from storage")
    g = train_index.get_networkx_graph()
    net = Network(notebook=True, cdn_resources="in_line", directed=True, height="1200px")
    net.from_nx(g)
    net.show("train.html")


if os.path.exists(persist_dir+"/docstore.json") and use_storage:
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    rules_index = load_index_from_storage(storage_context)
    logging.info("Loaded rules index from storage")
    g = rules_index.get_networkx_graph()
    net = Network(notebook=True, cdn_resources="in_line")
    net.from_nx(g)
    net.show("rules.html")