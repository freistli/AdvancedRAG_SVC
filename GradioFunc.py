
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

class FineTune:
    def __init__(self,  max_entities: int = 5, max_synonyms: int = 5,graph_traversal_depth: int = 2, max_knowledge_sequence: int = 30):    
        self.graph_traversal_depth = graph_traversal_depth
        self.max_entities = max_entities
        self.max_synonyms = max_synonyms
        self.max_knowledge_sequence = max_knowledge_sequence

    def __eq__(self, other):
        return self.graph_traversal_depth == other.graph_traversal_depth and \
                    self.max_entities == other.max_entities and \
                    self.max_synonyms == other.max_synonyms and \
                    self.max_knowledge_sequence == other.max_knowledge_sequence    

#langchain_openai
client = AzureChatOpenAI(
    openai_api_version="2024-02-15-preview",
    azure_deployment=os.environ['AZURE_OPENAI_Deployment'],
    streaming=True,
    callbacks=[StreamingGradioCallbackHandler()]
)

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
llama_debug = LlamaDebugHandler(print_trace_on_end=True)
token_counter = TokenCountingHandler(
    tokenizer=tiktoken.encoding_for_model('gpt-35-turbo').encode
)
Settings.callback_manager = CallbackManager([token_counter,llama_debug])


docClient = DocumentIntelligenceClient(doc_endpoint, AzureKeyCredential(doc_key))

def print_like_dislike(x: gr.LikeData):
    print(x.index, x.value, x.liked)


def stream_predict(message, history):
    prompt = ""
    message = HumanMessage(
        content=prompt
    )
    response = client.stream([systemMessage, message])
    partial_message = ""
    for chunk in response:
        partial_message = partial_message + chunk.dict()['content']
        yield partial_message

def DebugLlama():
    event_pairs = llama_debug.get_llm_inputs_outputs()
    try:
        print('\n\n=========================================\n\n')
        print(event_pairs)          
    except:
        print("Error")
    llama_debug.flush_event_logs()

def compose_query(Graph, QueryRules, content, fine_tune=None, content_prefix=os.environ['Content_Prefix']):

    global rules_index
    global train_index
    global composeGraph
    global graph_store
    global query_engine
    global rulesPath_query_engine
    global rulesPath_index
    global current_fine_Tune
    global current_graph_path

    rules_storage_dir=".//rules//storage"+"/"+Path(ruleFilePath).stem
    train_storage_dir=".//rules//storage"+"/"+Path(trainFilePath).stem

    Settings.llm = IndexGenerator(ruleFilePath).llm
  
    if Graph == "compose": 
        logging.info("compose")
        if composeGraph is None:

            if rules_index is None:
                if os.path.exists(rules_storage_dir+"/docstore.json") and use_storage:
                    storage_context = StorageContext.from_defaults(graph_store=graph_store,persist_dir=rules_storage_dir)
                    rules_index = load_index_from_storage(storage_context)

            if train_index is None:
                if os.path.exists(train_storage_dir+"/docstore.json") and use_storage:
                    train_storage_context = StorageContext.from_defaults(graph_store=graph_store,persist_dir=train_storage_dir)
                    train_index = load_index_from_storage(train_storage_context)


        service_context = ServiceContext.from_defaults(
                llm=Settings.llm,
                embed_model=Settings.embed_model,
                node_parser=Settings.node_parser
            ) 
        
        storage_context = StorageContext.from_defaults(graph_store=graph_store,persist_dir=rules_storage_dir)

        composeGraph = ComposableGraph.from_indices(
                    KnowledgeGraphIndex,
                    [rules_index,train_index],
                    index_summaries=['rules knowledge graph index','train knowledge graph index'],
                    service_context=service_context,
                    storage_context=storage_context,
                    include_embeddings=True)
        
        response_stream = composeGraph.as_query_engine(streaming=True,verbose=True,max_entities=10,
            graph_traversal_depth=3, similarity_top_k=5).query(QueryRules + f"\r\n {content_prefix}  \r\n "+content)

    elif Graph == "rules":
        logging.info("rules")
        if current_graph_path is None or current_graph_path != Graph:
                if os.path.exists(rules_storage_dir+"/docstore.json") and use_storage:
                    #storage_context = StorageContext.from_defaults(graph_store=graph_store,persist_dir=persist_dir)

                    graph_store = SimpleGraphStore().from_persist_dir(rules_storage_dir)                    
                    storage_context = StorageContext.from_defaults(graph_store=graph_store,persist_dir=rules_storage_dir)
                    rules_index = load_index_from_storage(storage_context)

        if current_fine_Tune is None or current_fine_Tune != fine_tune:

            print("Fine Tune changed")

            storage_context = StorageContext.from_defaults(graph_store=graph_store,persist_dir=rules_storage_dir)
           
            
            temp_retriever = rules_index.as_retriever(
                #entity_extract_template="",
                #synonym_expand_template="",
                graph_traversal_depth=fine_tune.graph_traversal_depth,
                max_entities=fine_tune.max_entities,
                max_synonyms=fine_tune.max_synonyms,
                max_knowledge_sequence=fine_tune.max_knowledge_sequence,
                verbose=True )

            query_engine = RetrieverQueryEngine.from_args(
                #graph_rag_retriever,
                llm=Settings.llm,
                retriever=temp_retriever,
                streaming=True,
                verbose=True
            )

            current_fine_Tune = fine_tune
                
            current_graph_path = Graph
        #response_stream = rules_index.as_query_engine(streaming=True,optimizer=optimizer,storage_context=storage_context,graph_traversal_depth=3, verbose=True, similarity_top_k=5).query(QueryRules + "\r\n 以下の文章を校正してください:  \r\n "+content)
        response_stream = query_engine.query(QueryRules + f"\r\n {content_prefix}  \r\n "+content) 

        DebugLlama()
        
    elif Graph == "train":
        logging.info("train")
        if train_index is None:
                if os.path.exists(train_storage_dir+"/docstore.json") and use_storage:
                    #train_storage_context = StorageContext.from_defaults(graph_store=graph_store,persist_dir=train_persist_dir)
                    train_storage_context = StorageContext.from_defaults(persist_dir=train_storage_dir)
                    train_index = load_index_from_storage(train_storage_context)
        response_stream = train_index.as_query_engine(streaming=True,graph_traversal_depth=3, verbose=True).query(QueryRules + "\r\n 以下の文章を校正してください:  \r\n "+content)
    
    else:
        logging.info(Graph)
        if current_graph_path is None or current_graph_path != Graph:
             rules_storage_dir = Graph
             if os.path.exists(rules_storage_dir+"/docstore.json") and use_storage:
                graph_store = SimpleGraphStore().from_persist_dir(rules_storage_dir)          
                storage_context = StorageContext.from_defaults(graph_store=graph_store,persist_dir=rules_storage_dir)
                rulesPath_index = load_index_from_storage(storage_context)

                storage_context = StorageContext.from_defaults(graph_store=graph_store,persist_dir=rules_storage_dir)
                
                temp_retriever = rulesPath_index.as_retriever(
                    #entity_extract_template="",
                    #synonym_expand_template="",
                    graph_traversal_depth=fine_tune.graph_traversal_depth,
                    max_entities=fine_tune.max_entities,
                    max_synonyms=fine_tune.max_synonyms,
                    max_knowledge_sequence=fine_tune.max_knowledge_sequence,
                    verbose=True )

                rulesPath_query_engine = RetrieverQueryEngine.from_args(
                    #graph_rag_retriever,
                    llm=Settings.llm,
                    retriever=temp_retriever,
                    streaming=True,
                    verbose=True
                )
                current_graph_path = Graph

        response_stream = rulesPath_query_engine.query(QueryRules + f"\r\n {content_prefix}  \r\n "+content) 

        DebugLlama()    

    partial_message = ""
    for text in response_stream.response_gen:
        partial_message = partial_message + text
        # do something with text as they arrive.
        yield partial_message
    DebugLlama()

def proof_read_addin(Content: str = "", streaming: bool = True):
    result = proof_read("rules", systemMessage.content, Content, "", 5, 5, 2, 30, None, True)

    if streaming == True:
        for text in result:
            yield text
    else:
        response = ""
        for text in result:   
            if len(text) > len(response):  # Check if the length of text is longer than the current response
                response = text         
            pass
        yield response 
    

def proof_read (Graph, QueryRules, Content: str = "" ,Draft: str = "", max_entities: int = 5, max_synonyms: int = 5,graph_traversal_depth: int = 2, max_knowledge_sequence: int = 30,request: gr.Request = None , addin: bool = False, content_prefix="以下の文章を校正してください: "):
    if request:
        print("Request headers dictionary:", request.headers)
        print("IP address:", request.client.host)
        print("Query parameters:", dict(request.query_params))

    begin = time.time()

    token_counter.reset_counts()

    if str(QueryRules).strip() == "":
        QueryRules = systemMessage.content

    if str(Graph).strip() == "":
        Graph = "rules"

    
    # Get the current time
    now = datetime.now()

    # Format the current time as a string, including milliseconds
    filename = now.strftime("%Y%m%d%H%M%S%f")
    
    partial_message = ""

    if str(Content).strip() != "":
        logging.info(Content)

        #textSplitter = CharacterTextSplitter(chunk_size=350, separator="\n", is_separator_regex=False)
        #to_be_proofread_content_list = textSplitter.split_text(Content)

        textSplitter = MarkdownTextSplitter.from_tiktoken_encoder(chunk_size = proofread_chunk)
        to_be_proofread_content_list = textSplitter.split_text(Content)  
        
    elif str(Draft).strip() != "":
        print(Draft)
        isText = False
        common = Common.Common(Draft)
        isText = common.isTextFile()

        if isText is not True:
            with open(Draft, 'rb') as file:
                file_content = file.read()

            to_be_proofread_content = docClient.begin_analyze_document(
                "prebuilt-layout",
                analyze_request=AnalyzeDocumentRequest(bytes_source=file_content),
                output_content_format="markdown"
            )

            docs_string = to_be_proofread_content.result().content
        else:
            with open(Draft, 'r') as file:
                docs_string = file.read()

        #textSplitter = CharacterTextSplitter(chunk_size=350, separator="\n", is_separator_regex=False)
        #to_be_proofread_content_list = textSplitter.split_text(docs_string)        

        textSplitter = MarkdownTextSplitter.from_tiktoken_encoder(chunk_size = proofread_chunk)
        to_be_proofread_content_list = textSplitter.split_text(docs_string)  

    fine_tune = FineTune(max_entities, max_synonyms,graph_traversal_depth,max_knowledge_sequence)
    proofreadResultFilePath = os.path.join(persist_dir+"/"+filename + '_proofread_result.md')
    currentProofRead = ""
    with open(proofreadResultFilePath, 'a') as f:
        for i in range(len(to_be_proofread_content_list)):             
            status = "\n\nTime elapsed: " +str(time.time() - begin)+ "\n\nPart: "+str(i+1) + " of " + str(len(to_be_proofread_content_list))+" :"
            try:                        
                logging.info(status)
                logging.info("processing content: " + to_be_proofread_content_list[i])
                
                llm_previous = Settings.llm

                response = compose_query(Graph, QueryRules, to_be_proofread_content_list[i],fine_tune,content_prefix)
                
                for text in response:
                    status = "\n\nTime elapsed: " +str(time.time() - begin)+ "\n\nPart: "+str(i+1) + " of " + str(len(to_be_proofread_content_list))+" :"
                    currentProofRead = status + "\n\n" + text 
                    if addin:
                        yield currentProofRead
                    else:
                        yield currentProofRead, proofreadResultFilePath
                partial_message = currentProofRead
                f.write(partial_message)
                f.flush()
                if addin:
                        yield currentProofRead
                else:
                        yield currentProofRead, proofreadResultFilePath

                logging.info(partial_message)
            except Exception as e:
                logging.error("Error: " + str(e))
                f.write(status)
                f.write("Error: " + str(e))
                f.flush()
                if addin:
                    yield "Error: " + str(e)
                else:
                    yield "Error: " + str(e),proofreadResultFilePath                
            finally:
                Settings.llm = llm_previous
                

    print(
        "Embedding Tokens: ",
        token_counter.total_embedding_token_count,
        "\n",
        "LLM Prompt Tokens: ",
        token_counter.prompt_llm_token_count,
        "\n",
        "LLM Completion Tokens: ",
        token_counter.completion_llm_token_count,
        "\n",
        "Total LLM Token Count: ",
        token_counter.total_llm_token_count,
        "\n",
    )

    end = time.time()

    logging.info("Time elapsed: " + str(end - begin) + " seconds")

    if (currentProofRead.strip() != ""):
        if addin:
            yield currentProofRead
        else:
            yield currentProofRead,proofreadResultFilePath


def build_azure_index(indexName, filePath):
    azureAISearchIndexGenerator = AzureAISearchIndexGenerator(filePath, indexName, Path(filePath).stem)
    result = azureAISearchIndexGenerator.GenerateIndex() 
    for status in result:
        yield status

def build_RecursiveRetriever_index(filePath,nodeRefer):
    recursiveRetrieverIndexGenerator = RecursiveRetrieverIndexGenerator (filePath, "","", nodeRefer)
    result = recursiveRetrieverIndexGenerator.GenerateOrLoadIndex(nodeRefer=nodeRefer)
    for msg, zipfile in result:
        yield msg, zipfile

def build_Summary_index(filePath):
    summaryIndexGenerator = SummaryIndexGenerator (filePath, "","")
    result = summaryIndexGenerator.GenerateOrLoadIndex()
    for msg, zipfile in result:
        yield msg, zipfile

def build_GraghRAG_index(filePath,storageName,indexName):
    graphRagIndexGenerator = GraphRagIndexGenerator (filePath, storageName,indexName)
    result = graphRagIndexGenerator.upload_files(os.path.dirname(filePath),storageName,overwrite=False)
    yield result.text
    result = graphRagIndexGenerator.build_index_default(storageName,indexName)
    yield result.text
    percent_complete = 0
    while (percent_complete < 100):    
            response = graphRagIndexGenerator.index_status(indexName)
            if response.ok:
                status_data = response.json()
                percent_complete = status_data.get('percent_complete', 0)
                progress = status_data.get('progress', 0)
                if percent_complete >= 100:
                    yield "Indexing is 100% complete."
                else:
                    yield f"{datetime.now()}: {percent_complete}% --> {progress}"
            else:
                yield f"Failed to get index status.\nStatus: {response.text}"
            time.sleep(5)
                
def view_graph(persist_dir):
    if os.path.exists(persist_dir+"/docstore.json"):
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        rules_index = load_index_from_storage(storage_context)
        status = "Loaded rules index from storage"
        logging.info(status)
        yield status, "NotReady"
        g = rules_index.get_networkx_graph()
        net = Network(notebook=True, cdn_resources="in_line")
        net.from_nx(g)
        filename = Path(persist_dir).stem+"_graph.html"
        filePath = tmpdirname + "/" + filename
        net.save_graph(filePath)     
        status = "Graph view created: "+filePath
        logging.info(status)
        yield status, filePath    
    else:
        return "No graph found","NotReady"
    
def KnowledgeGraphIndexSearch(indexName, systemMessage, content):
    result = proof_read(indexName, systemMessage, content, "", 5, 5, 2, 30, None, True, "Please help to answer: ")
    downloadproofreadbutton = gr.DownloadButton(label="Download Proofread Result")
    for text in result:
       yield text
 
def chat_bot(message, history, indexType, indexName, systemMessage, streaming: bool = True):
    history_openai_format = []
    history_openai_format.append({"role": "system", "content": systemMessage})
    for human, assistant in history:
            history_openai_format.append({"role": "user", "content": human })
            history_openai_format.append({"role": "assistant", "content":assistant})
    history_openai_format.append({"role": "user", "content": message})

    response = None  

    if indexType == AZURE_AI_SEARCH:
        azureAISearchIndexGenerator = AzureAISearchIndexGenerator(docPath="", indexName=indexName, idPrefix="")
        azureAISearchIndexGenerator.LoadIndex()
        response = azureAISearchIndexGenerator.HybridSearch(str(history_openai_format))
        
    elif indexType == KNOWLEDGE_GRAPH:
        response = KnowledgeGraphIndexSearch(indexName, str(history_openai_format), message)
        
    elif indexType == MS_GRAPHRAG_GLOBAL:
        graphRagIndexGenerator = GraphRagIndexGenerator("","",indexName)
        result,context = graphRagIndexGenerator.test_global_search(str(history_openai_format))
        yield result
    elif indexType == MS_GRAPHRAG_LOCAL:
        graphRagIndexGenerator = GraphRagIndexGenerator("","",indexName)
        result,context  = graphRagIndexGenerator.test_local_search(str(history_openai_format))
        yield result
    elif indexType == RECURSIVE_RETRIEVER:
        recursiveRetrieverIndexGenerator = RecursiveRetrieverIndexGenerator (docPath="", indexName="", idPrefix="")
        result = recursiveRetrieverIndexGenerator.LoadIndex(indexFolder=indexName)
        for text in result:
            pass
        response = recursiveRetrieverIndexGenerator.RecursiveRetrieverSearch(str(history_openai_format))
        
    elif indexType == SUMMARY_INDEX:
        summaryIndexGenerator = SummaryIndexGenerator (docPath="", indexName="", idPrefix="")
        result = summaryIndexGenerator.LoadIndex(indexFolder=indexName)
        for text in result:
            pass
        response = summaryIndexGenerator.SummaryRetrieverSearch(str(history_openai_format))  

    elif indexType == CSV_AI_ANALYSIS:
        dfa = DataFrameAnalysis.DataFrameAnalysis(indexName)
        dfa.load_data()
        status = "Data loaded successfully: " + indexName
        partial_message = ""
        print(status)
        response = dfa.query_data(str(history_openai_format))       
          

    if response is not None:
        if streaming == True:
            for text in response:
                yield text
        else:
            finalResponse = ""
            for text in response:   
                if len(text) > len(finalResponse):  # Check if the length of text is longer than the current response
                    finalResponse = text         
                pass
            yield finalResponse 

def CSV_Load(file):
    dfa = DataFrameAnalysis.DataFrameAnalysis(file)
    dfa.load_data()
    status = "Data loaded successfully: " + file
    partial_message = ""
    print(status)
    partial_message += status
    yield file,partial_message


def build_index(ruleFilePath):
    indexGenerator = IndexGenerator(ruleFilePath)
 
    producer = Thread(target=indexGenerator.GenerateIndex)
    producer.start()    
    results  = indexGenerator.ReportRunningStatus()    
    for status in results:
        yield status, indexGenerator.zipFile

    producer.join()

def updateFileName(file):
    return file