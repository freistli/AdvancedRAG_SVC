
import base64
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time

from dotenv import load_dotenv
from fastapi.responses import RedirectResponse
from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter,MarkdownTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.callbacks import BaseCallbackHandler
from langchain_experimental.text_splitter import SemanticChunker
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

train_index = None
rules_index = None
composeGraph = None


job_done = object() # signals the processing is done

q = SimpleQueue()
class StreamingGradioCallbackHandler(BaseCallbackHandler):
    def __init__(self, q: SimpleQueue):
        self.q = q

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts running. Clean the queue."""
        while not self.q.empty():
            try:
                self.q.get(block=False)
            except:
                continue

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        self.q.put(token)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""
        self.q.put(job_done)

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when LLM errors."""
        self.q.put(job_done)

load_dotenv('.env_4_SC')

endpoint = os.environ['DOC_AI_BASE']
key = os.environ['DOC_AI_KEY']

import tiktoken
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
token_counter = TokenCountingHandler(
    tokenizer=tiktoken.encoding_for_model('gpt-4').encode
)

#langchain_openai
client = AzureChatOpenAI(
    openai_api_version="2024-02-15-preview",
    azure_deployment=os.environ['AZURE_OPENAI_Deployment'],
    streaming=True,
    callbacks=[StreamingGradioCallbackHandler(q)]
)

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
Settings.callback_manager = CallbackManager([token_counter])


docClient = DocumentIntelligenceClient(endpoint, AzureKeyCredential(key))

#setup the storage context

graph_store = SimpleGraphStore()

use_storage = False
batch_size = 10


def GenerateKGIndex(ruleFilePath, persist_dir, graph_store, use_storage, batch_size):
    # Check if the index is already created and stored in the persist directory
    if os.path.exists(persist_dir+"/docstore.json") and use_storage:
        storage_context = StorageContext.from_defaults(graph_store=graph_store,persist_dir=persist_dir)
        rules_index = load_index_from_storage(storage_context)
    else:  
        layoutJson = persist_dir + "/"+Path(ruleFilePath).stem+".json"

        # Load the document and analyze the layout from online service
        if not os.path.exists(layoutJson):
            with open(ruleFilePath, 'rb') as file:
                file_content = file.read()

            layoutDocs = docClient.begin_analyze_document(
                "prebuilt-layout",
                analyze_request=AnalyzeDocumentRequest(bytes_source=file_content),
                output_content_format="markdown"
            )        
            docs_string = layoutDocs.result().content
            with open(layoutJson, 'w') as json_file:
                json.dump(layoutDocs.result().content, json_file)

        # Load the document and analyze the layout from local file
        else:
            with open(layoutJson) as json_file:
                docs_string = json.load(json_file)  


        splitter = MarkdownTextSplitter.from_tiktoken_encoder(chunk_size=300)
        rules_content_list = splitter.split_text(docs_string)  
        #chunk the original content
        #splits = text_splitter.split_text(docs_string)
        print(rules_content_list)

        docs = []

        for i in range(len(rules_content_list)):
            doc = Document(text=rules_content_list[i],id_=str(i))
            docs.append(doc)
        
        if os.path.exists(persist_dir+"/docstore.json"):
            storage_context = StorageContext.from_defaults(graph_store=graph_store,persist_dir=persist_dir)
        else:
            storage_context = StorageContext.from_defaults(graph_store=graph_store)
        
        # Create the index from the documents
        nodes = Settings.node_parser.get_nodes_from_documents(docs)

        rules_index = KnowledgeGraphIndex(nodes=nodes[0:10],
                                        max_triplets_per_chunk=5,
                        storage_context=storage_context,
                        include_embeddings=True,
                        show_progress=True,
                        index_id="rules_index")
                        
        
        startFrom = 10

        for i in range(startFrom, len(nodes), batch_size):
            logging.warning(f"Processing batch {i} to {i+batch_size}, total {len(nodes)} nodes")
            
            batch_nodes = nodes[i:i+batch_size]

            logging.info(str(batch_nodes))

            max_retries = 5
            attempts = 0
            success = False

            while attempts < max_retries and not success:
                try:
                    rules_index.insert_nodes(nodes=batch_nodes)                
                    success = True
                except Exception as e:
                    attempts += 1
                    logging.error(f"Failed to create index, retrying {attempts} of {max_retries}")
                    logging.error(str(e))

            logging.warning(f"Persisting batch {i} to {i+batch_size}, total {len(nodes)} nodes")
            rules_index.storage_context.persist(persist_dir=persist_dir)
        
        rules_index.storage_context.persist(persist_dir=persist_dir)
        return rules_index
    
rules_index = GenerateKGIndex(ruleFilePath, persist_dir, graph_store, use_storage, batch_size)