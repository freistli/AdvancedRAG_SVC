
import base64
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time
import Common
from dotenv import load_dotenv
from fastapi.responses import HTMLResponse, RedirectResponse
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
from llama_index.core.postprocessor.optimizer import SentenceEmbeddingOptimizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import KnowledgeGraphRAGRetriever
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core.callbacks import LlamaDebugHandler

from pyvis.network import Network
from uvicorn import run
import multiprocessing
import uvicorn
from BlockScripts import *


load_dotenv('.env_4_SC')

Embedding_Mode = os.environ['AZURE_OPENAI_EMBEDDING_Deployment']

Predict_Concurrency = int(os.environ['Predict_Concurrency'])
Build_Concurrency = int(os.environ['Build_Concurrency'])
Max_Queue_Size = int(os.environ['Max_Queue_Size'])

Individual_Chat = bool(os.environ['Individual_Chat'] == 'True')

Proofread_Tab = bool(os.environ['Proofread_Tab'] == 'True')
AdvProofread_Tab = bool(os.environ['AdvProofread_Tab'] == 'True')
ProofreadAddin_Tab = bool(os.environ['ProofreadAddin_Tab'] == 'True')
BuildRRIndex_Tab = bool(os.environ['BuildRRIndex_Tab'] == 'True')
AdvChatBot_Tab = bool(os.environ['AdvChatBot_Tab'] == 'True')
BuildKGIndex_Tab = bool(os.environ['BuildKGIndex_Tab'] == 'True')
BuildGraphRAGIndex_Tab= bool(os.environ['BuildGraphRAGIndex_Tab'] == 'True')
BuildSummaryIndex_Tab= bool(os.environ['BuildSummaryIndex_Tab'] == 'True')
CSVQueryEngine_Tab= bool(os.environ['CSVQueryEngine_Tab'] == 'True')
BuildAzureIndex_Tab= bool(os.environ['BuildAzureIndex_Tab'] == 'True')

AdvChatBot_API = bool(os.environ['AdvChatBot_API'] == 'True')
Proofread_API = bool(os.environ['Proofread_API'] == 'True')
AdvProofread_API = bool(os.environ['AdvProofread_API'] == 'True')
ProofreadAddin_API = bool(os.environ['ProofreadAddin_API'] == 'True')
BuildRRIndex_API = bool(os.environ['BuildRRIndex_API'] == 'True')
BuildKGIndex_API = bool(os.environ['BuildKGIndex_API'] == 'True')
BuildGraphRAGIndex_API = bool(os.environ['BuildGraphRAGIndex_API'] == 'True')
BuildSummaryIndex_API = bool(os.environ['BuildSummaryIndex_API'] == 'True')
CSVQueryEngine_API = bool(os.environ['CSVQueryEngine_API'] == 'True')
BuildAzureIndex_API = bool(os.environ['BuildAzureIndex_API'] == 'True')


AZURE_AI_SEARCH = "Azure AI Search"
MS_GRAPHRAG_LOCAL = "MS GraghRAG Local"
MS_GRAPHRAG_GLOBAL = "MS GraghRAG Global"
KNOWLEDGE_GRAPH = "Knowledge Graph"
RECURSIVE_RETRIEVER = "Recursive Retriever"
SUMMARY_INDEX = "Tree Mode Summary"
CSV_AI_ANALYSIS = "CSV Query Engine"

CHAT_MODE_LABLE = "Chat Mode"
AZURE_AI_SEARCH_LABLE = "Azure AI Search Index"
MS_GRAPHRAG_LABLE = "MS GraghRAG Index"
KNOWLEDGE_GRAPH_LABLE = "Knowledge Graph Index"
RECURSIVE_RETRIEVER_LABLE = "Recursive Retriever Index"
SUMMARY_INDEX_LABLE = "Tree Mode Summary Index"
CSV_AI_ANALYSIS_LABLE = "CSV Query Engine"
PROOFREAD_LABLE = "Proofread"
PROOFREAD_ADDIN_LABLE = "Proofread Addin"
ADVANCED_PROOFREAD_LABLE = "Advanced Proofread"


modelName = "Azure OpenAI "+ os.environ['AZURE_OPENAI_Deployment']
if bool(os.environ['USE_LMSTUDIO'] == 'True'):
    modelName += ", LM-STUDIO "+ os.environ['LLAMACPP_MODEL']
elif bool(os.environ['USE_OLLAMA'] == 'True'):
    modelName += ", OLLAMA "+ os.environ['OLLAMA_MODEL']


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
graph_store = None
query_engine = None
rulesPath_query_engine = None
rulesPath_index = None
current_fine_Tune = None
current_graph_path = None
proofread_chunk = 500

doc_endpoint = os.environ['DOC_AI_BASE']
doc_key = os.environ['DOC_AI_KEY']


use_storage = True
batch_size = 10



prompt = ""
systemMessage = SystemMessage(
    content = os.environ['Proofread_System_Message']
)
message = HumanMessage(
    content=prompt
)

default_system_message = os.environ['Default_System_Message']

