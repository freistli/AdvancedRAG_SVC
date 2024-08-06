'''
Functionality: 
    1. Proofreading by Azure OpenAI LLM
    2. Build Knowledge Graph Index
    3. View and Download Knowledge Graph Index
    4. Advanced Proofreading by Azure OpenAI LLM
    5. Download Proofread Result
    6. Choose Selected Content to upload
Author: Freist Li
Date: 2024-05
Version: 1.0
'''

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
import tiktoken
import DataFrameAnalysis
from GradioView import advchat_bot_view, proofread_addin_view
from IndexGenerator import IndexGenerator, StreamingGradioCallbackHandler
from AzureSearchIndexGenerator import AzureAISearchIndexGenerator
from RecursiveRetrieverIndexGenerator import RecursiveRetrieverIndexGenerator
from SummmaryIndexGenerator import SummaryIndexGenerator
from GraphRagIndexGenerator import GraphRagIndexGenerator
import networkx as nx
from pyvis.network import Network
from uvicorn import run
import multiprocessing
import uvicorn
import gradio as gr 
from Environment import *
from GradioView import *


load_dotenv('.env_4_SC')

optimizer = SentenceEmbeddingOptimizer(
    percentile_cutoff=0.5,
    threshold_cutoff=0.7,
    context_before=1,
    context_after=1
)

logging.basicConfig(stream=sys.stdout, level=os.environ['LOG_LEVEL'])
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))   


app = FastAPI()

@app.get("/",response_class=HTMLResponse)
def read_root():
    return RedirectResponse(url="/main")

@app.get("/main/info",response_class=HTMLResponse)
def read_main():
    html_content = f"Thanks for using Advanced RAG Service Studio"
    return html_content

with gr.Blocks(title=f"Advanced RAG Service Studio",analytics_enabled=False, css="footer{display:none !important}", js=js,theme=gr.themes.Default(spacing_size="sm", radius_size="none", primary_hue="blue")).queue(default_concurrency_limit=int(os.environ['Predict_Concurrency']),max_size=Max_Queue_Size) as custom_theme_Main:
        
            if AdvChatBot_Tab is True:
                with gr.Tab(CHAT_MODE_LABLE) as tab:
                    advchat_bot_view()            

            if BuildAzureIndex_Tab is True:
                with gr.Tab(AZURE_AI_SEARCH_LABLE) as tab:
                    azuresearch_view()           

            if BuildKGIndex_Tab is True:
                with gr.Tab(KNOWLEDGE_GRAPH_LABLE) as tab:
                    knowledgegraph_view() 

            if BuildGraphRAGIndex_Tab is True:
                with gr.Tab(MS_GRAPHRAG_LABLE) as tab:
                    graphrag_view()    

            if CSVQueryEngine_Tab is True:
                with gr.Tab(CSV_AI_ANALYSIS_LABLE) as tab:
                    csvqueryengine_view()

            if BuildSummaryIndex_Tab is True:
                with gr.Tab(SUMMARY_INDEX_LABLE) as tab:
                    summaryindex_view()

            if BuildRRIndex_Tab is True:
                with gr.Tab(RECURSIVE_RETRIEVER_LABLE) as tab:
                    recursive_retriever_view()

            if Proofread_Tab is True:
                with gr.Tab(PROOFREAD_LABLE) as tab:
                    proofread_view()

            if AdvProofread_Tab is True:
                with gr.Tab(ADVANCED_PROOFREAD_LABLE) as tab:
                    adv_proofread_view()

            

            with gr.Tab("View Setting") as tab:
                darkmode_button()
                

app = gr.mount_gradio_app(app, custom_theme_Main, path="/main")

if ProofreadAddin_API is True:
    proofread_addin_view(app)    

if AdvChatBot_API is True:
    advchat_bot_view(app)

if CSVQueryEngine_API is True:
    csvqueryengine_view(app)    

if __name__ == '__main__':
    multiprocessing.freeze_support()  # For Windows support
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ["PORT"]), reload=False, workers=1)