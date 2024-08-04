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


load_dotenv('.env_4_SC')

optimizer = SentenceEmbeddingOptimizer(
    percentile_cutoff=0.5,
    threshold_cutoff=0.7,
    context_before=1,
    context_after=1
)

logging.basicConfig(stream=sys.stdout, level=os.environ['LOG_LEVEL'])
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
graph_store = None
query_engine = None
rulesPath_query_engine = None
rulesPath_index = None
current_fine_Tune = None
current_graph_path = None
proofread_chunk = 500
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


AZURE_AI_SEARCH = "Azure AI Search"
MS_GRAPHRAG_LOCAL = "MS GraghRAG Local"
MS_GRAPHRAG_GLOBAL = "MS GraghRAG Global"
KNOWLEDGE_GRAPH = "Knowledge Graph"
RECURSIVE_RETRIEVER = "Recursive Retriever"
SUMMARY_INDEX = "Tree Mode Summary"
CSV_AI_ANALYSIS = "CSV Query Engine"



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

endpoint = os.environ['DOC_AI_BASE']
key = os.environ['DOC_AI_KEY']

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
                                                             model="text-embedding-ada-002",
                                                             deployment="text-embedding-ada-002",
                                                             openai_api_key=os.environ['AZURE_OPENAI_API_KEY']))


 
Settings.llm = llama_index_llm
Settings.embed_model = llama_index_embed_model
Settings.node_parser = SemanticSplitterNodeParser(buffer_size=1, breakpoint_percentile_threshold=95,embed_model=llama_index_embed_model)
llama_debug = LlamaDebugHandler(print_trace_on_end=True)
token_counter = TokenCountingHandler(
    tokenizer=tiktoken.encoding_for_model('gpt-35-turbo').encode
)
Settings.callback_manager = CallbackManager([token_counter,llama_debug])


docClient = DocumentIntelligenceClient(endpoint, AzureKeyCredential(key))

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

import gradio as gr 
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
        #if rules_index is None:
        if query_engine is None:
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
                llm=IndexGenerator(ruleFilePath).llm,
                graph_traversal_depth=fine_tune.graph_traversal_depth,
                max_entities=fine_tune.max_entities,
                max_synonyms=fine_tune.max_synonyms,
                max_knowledge_sequence=fine_tune.max_knowledge_sequence,
                verbose=True )

            query_engine = RetrieverQueryEngine.from_args(
                #graph_rag_retriever,
                llm=IndexGenerator(ruleFilePath).llm,
                retriever=temp_retriever,
                streaming=True,
                verbose=True
            )

            current_fine_Tune = fine_tune

                

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
                    #storage_context = StorageContext.from_defaults(graph_store=graph_store,persist_dir=persist_dir)
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
                    llm=IndexGenerator(ruleFilePath).llm,
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
                continue

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

    if addin:
        yield currentProofRead
    else:
        yield currentProofRead,proofreadResultFilePath


js = "custom.js"

css = """
footer{display:none !important}
[role="tab"]
{
   font-family: 'Arial', sans-serif;
   font-size: 20px;
   font-weight: bold;
}
"""

app = FastAPI()
@app.get("/",response_class=HTMLResponse)
def read_main():
    with open('UI.html', 'r') as file:
        html_content = file.read()    
    return html_content



def build_index(ruleFilePath):
    indexGenerator = IndexGenerator(ruleFilePath)
 
    producer = Thread(target=indexGenerator.GenerateIndex)
    producer.start()    
    results  = indexGenerator.ReportRunningStatus()    
    for status in results:
        yield status, indexGenerator.zipFile

    producer.join()

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

def print_like_dislike(x: gr.LikeData):
    print(x.index, x.value, x.liked)


def CSV_Load(file):
    dfa = DataFrameAnalysis.DataFrameAnalysis(file)
    dfa.load_data()
    status = "Data loaded successfully: " + file
    partial_message = ""
    print(status)
    partial_message += status
    yield file,partial_message


modelName = "Azure OpenAI "+ os.environ['AZURE_OPENAI_Deployment']
if bool(os.environ['USE_LMSTUDIO'] == 'True'):
    modelName += ", LM-STUDIO "+ os.environ['LLAMACPP_MODEL']
elif bool(os.environ['USE_OLLAMA'] == 'True'):
    modelName += ", OLLAMA "+ os.environ['OLLAMA_MODEL']

if AdvChatBot_Tab is True:
    if Individual_Chat is False:
        with gr.Blocks(title=f"Chat with {modelName}",analytics_enabled=False, css="footer{display:none !important}", js=js,theme=gr.themes.Default(spacing_size="sm", radius_size="none", primary_hue="blue")).queue(default_concurrency_limit=int(os.environ['Predict_Concurrency']),max_size=Max_Queue_Size) as custom_theme_ChatBot:
            #interface = gr.Interface(fn=proof_read, inputs=["file"],outputs="markdown",css="footer{display:none !important}",allow_flagging="never")
            with gr.Row():    
                with gr.Column(scale=1):
                    with gr.Accordion("Chatbot Configuration", open=True):
                        checkbox_Stream = gr.Checkbox(label="Streaming", value=True)
                        radtio_ptions = gr.Radio([AZURE_AI_SEARCH,MS_GRAPHRAG_LOCAL,MS_GRAPHRAG_GLOBAL,KNOWLEDGE_GRAPH,RECURSIVE_RETRIEVER, SUMMARY_INDEX,CSV_AI_ANALYSIS], label="Index Type", value="Azure AI Search")
                        textbox_index = gr.Textbox("azuresearch_0", label="Search Index Name, can be index folders or Azure AI Search Index Name")
                with gr.Column(scale=3): 
                    chatbot = gr.Chatbot(likeable=True,
                                                show_share_button=False, 
                                                show_copy_button=True, 
                                                bubble_full_width = False,
                                                render=True
                                                )
                    chatbot.like(print_like_dislike,None, None)       
                    textbox_systemMessage = gr.Textbox(default_system_message, label="System Message",visible=True, lines=9)            
                    interface = gr.ChatInterface(fn=chat_bot,
                                    chatbot=chatbot,
                                    additional_inputs=[radtio_ptions, textbox_index, textbox_systemMessage, checkbox_Stream], 
                                    submit_btn="Chat",                             
                                    examples = [["provide summary for the document"],["give me insights of the document"]])  
                    interface.queue(default_concurrency_limit=Build_Concurrency,max_size=Max_Queue_Size)  

        app = gr.mount_gradio_app(app, custom_theme_ChatBot, path="/advchatbot")
    else:
        with gr.Blocks(title=f"Chat with {modelName}",analytics_enabled=False, css="footer{display:none !important}", js=js,theme=gr.themes.Default(spacing_size="sm", radius_size="none", primary_hue="blue")).queue(default_concurrency_limit=int(os.environ['Predict_Concurrency']),max_size=Max_Queue_Size) as custom_theme_ChatBot:
            gr.Label("Chat Mode is now configured into Each Index Tab")
        app = gr.mount_gradio_app(app, custom_theme_ChatBot, path="/advchatbot")
    

if AdvProofread_Tab is True:
    with gr.Blocks(title=f"Advanced Proofreading by {modelName}",analytics_enabled=False, css="footer{display:none !important}", js=js,theme=gr.themes.Default(spacing_size="sm", radius_size="none", primary_hue="blue")).queue(default_concurrency_limit=int(os.environ['Predict_Concurrency']),max_size=Max_Queue_Size) as custom_theme:
        #interface = gr.Interface(fn=proof_read, inputs=["file"],outputs="markdown",css="footer{display:none !important}",allow_flagging="never")
        max_entities = gr.Slider(label="Max Entities", value=5, minimum=1, maximum=10, step=1)
        max_synonyms = gr.Slider(label="Max Synonyms", value=5, minimum=1, maximum=10, step=1)
        graph_traversal_depth = gr.Slider(label="Graph Traversal Depth", value=2, minimum=1, maximum=10, step=1)
        max_knowledge_sequence = gr.Slider(label="Max Knowledge Sequence", value=30, minimum=1, maximum=50, step=1)
        texbox_Rules = gr.Textbox(lines=1, label="Knowledge Graph of Proofreading Rules (rules, train, compose)", value="rules")
        textbox_QueryRules = gr.Textbox(lines=10, label="Preset Query Prompt", value=systemMessage.content)
        textbox_Content_Empty = gr.Textbox(lines=10)
        downloadproofreadbutton = gr.DownloadButton(label="Download Proofread Result")
        interface = gr.Interface(fn=proof_read, inputs=[texbox_Rules, 
                                                        textbox_QueryRules, 
                                                        textbox_Content_Empty,                                                    
                                                        "file",
                                                        max_entities,
                                                        max_synonyms,
                                                        graph_traversal_depth,
                                                        max_knowledge_sequence,
                                                        ], outputs=["markdown",downloadproofreadbutton],allow_flagging="never",analytics_enabled=False)
        interface.queue(default_concurrency_limit=Build_Concurrency,max_size=Max_Queue_Size) 

    app = gr.mount_gradio_app(app, custom_theme, path="/advproofread")

if Proofread_Tab is True:
    with gr.Blocks(title=f"Proofreading by {modelName}",analytics_enabled=False, css="footer{display:none !important}", js=js,theme=gr.themes.Default(spacing_size="sm", radius_size="none", primary_hue="blue")).queue(default_concurrency_limit=Predict_Concurrency,max_size=Max_Queue_Size) as custom_theme_base:
        #interface = gr.Interface(fn=proof_read, inputs=["file"],outputs="markdown",css="footer{display:none !important}",allow_flagging="never")
        texbox_Rules = gr.Textbox(lines=1, label="Knowledge Graph of Proofreading Rules (rules, train, compose)", value="rules")
        textbox_QueryRules = gr.Textbox(lines=10, label="Preset Query Prompt", value=systemMessage.content)
        textbox_Content = gr.Textbox(lines=10, elem_id="ProofreadContent", label="Content to be Proofread", value=os.environ['Sample_Content'])
        interface = gr.Interface(fn=proof_read, inputs=[texbox_Rules, 
                                                        textbox_QueryRules,
                                                        textbox_Content], outputs=["markdown",downloadproofreadbutton],allow_flagging="never",analytics_enabled=False)
        interface.queue(default_concurrency_limit=Build_Concurrency,max_size=Max_Queue_Size) 

    app = gr.mount_gradio_app(app, custom_theme_base, path="/proofread")

if ProofreadAddin_Tab is True:
    with gr.Blocks(title=f"Proofreading by {modelName}",analytics_enabled=False, css="footer{display:none !important}", js=js,theme=gr.themes.Default(spacing_size="sm", radius_size="none", primary_hue="blue")).queue(default_concurrency_limit=Predict_Concurrency,max_size=Max_Queue_Size) as custom_theme_addin:
        #interface = gr.Interface(fn=proof_read, inputs=["file"],outputs="markdown",css="footer{display:none !important}",allow_flagging="never")
        button = gr.Button("Choose Selected Content",elem_id="ChooseSelectedContent")
        StreamingCheckBox = gr.Checkbox(label="Streaming", value=True)
        textbox_Content = gr.Textbox(lines=10, elem_id="ProofreadContent", label="Content to be Proofread", value=os.environ['Sample_Content'])
        interface = gr.Interface(fn=proof_read_addin, inputs=[textbox_Content,StreamingCheckBox], outputs=["markdown"],allow_flagging="never",analytics_enabled=False)
        interface.queue(default_concurrency_limit=Build_Concurrency,max_size=Max_Queue_Size) 

    app = gr.mount_gradio_app(app, custom_theme_addin, path="/proofreadaddin")


if BuildAzureIndex_Tab is True:
    with gr.Blocks(title="Build and Run Index on Azure AI Search",analytics_enabled=False, css=css, js=js,theme=gr.themes.Default(spacing_size="sm", radius_size="none", primary_hue="blue")).queue(default_concurrency_limit=Build_Concurrency,max_size=Max_Queue_Size) as custom_theme_AzureSearchV2:
        #interface = gr.Interface(fn=proof_read, inputs=["file"],outputs="markdown",css="footer{display:none !important}",allow_flagging="never")
        input_indexname_azuresearch = gr.Textbox("azuresearch_0",lines=1)  
        with gr.Tab("Build Index"):                         
                    input_file_azuresearch = gr.File()
                    output_markdown_azuresearch = gr.Markdown()
                    interface=gr.Interface(fn=build_azure_index, inputs=[input_indexname_azuresearch,input_file_azuresearch], outputs=[output_markdown_azuresearch],
                                allow_flagging="never",
                                analytics_enabled=False,
                                submit_btn="Build Index")
                    interface.queue(default_concurrency_limit=Build_Concurrency,max_size=Max_Queue_Size) 

        if Individual_Chat is True: 
                with gr.Tab("Chat Mode"):
                            radtio_ptions_azuresearch = gr.Radio([AZURE_AI_SEARCH], label="Index Type", value="Azure AI Search", visible=False)
                            with gr.Accordion("Chat Settings", open=False):            
                                textbox_systemMessage_azuresearch = gr.Textbox(default_system_message, label="System Message",visible=True, lines=5)
                            checkbox_Stream_azuresearch = gr.Checkbox(label="Streaming", value=True)
                            interface=gr.ChatInterface(fn=chat_bot,
                                                chatbot= gr.Chatbot(likeable=False,
                                                        show_share_button=False, 
                                                        show_copy_button=True, 
                                                        bubble_full_width = False,
                                                        render=True
                                                        ),
                                                additional_inputs=[radtio_ptions_azuresearch, input_indexname_azuresearch, textbox_systemMessage_azuresearch, checkbox_Stream_azuresearch], 
                                                submit_btn="Chat",                                
                                                examples = [["provide summary for the document"],["give me insights of the document"]])
                            interface.queue(default_concurrency_limit=Build_Concurrency,max_size=Max_Queue_Size) 

        app = gr.mount_gradio_app(app, custom_theme_AzureSearchV2, path="/buildrunazureindex")


if BuildSummaryIndex_Tab is True:
    with gr.Blocks(title="Build and Run Summary Index",analytics_enabled=False, css=css, js=js,theme=gr.themes.Default(spacing_size="sm", radius_size="none", primary_hue="blue")).queue(default_concurrency_limit=Build_Concurrency,max_size=Max_Queue_Size) as custom_theme_summaryIndexV2:
        input_indexname_summary = gr.Textbox(label="Index Name",placeholder= "Summary index folder path generated by Index building",lines=1)   
        with gr.Tab("Build Index"):
                downloadbutton_summary = gr.DownloadButton(label="Download Index")
                interface = gr.Interface(fn=build_Summary_index, inputs=["file"], outputs=["markdown",downloadbutton_summary],allow_flagging="never",analytics_enabled=False)
                interface.queue(default_concurrency_limit=Build_Concurrency,max_size=Max_Queue_Size) 

        if Individual_Chat is True: 
            with gr.Tab("Chat Mode"):   
                    radtio_ptions_summary = gr.Radio([SUMMARY_INDEX], label="Index Type", value=SUMMARY_INDEX, visible=False)   
                    with gr.Accordion("Chat Settings", open=False):           
                        textbox_systemMessage_summary = gr.Textbox(default_system_message, label="System Message",visible=True, lines=5)
                    checkbox_Stream_summary = gr.Checkbox(label="Streaming", value=True)
                    interface=gr.ChatInterface(fn=chat_bot,
                                        chatbot= gr.Chatbot(likeable=False,
                                                show_share_button=False, 
                                                show_copy_button=True, 
                                                bubble_full_width = False,
                                                render=True
                                                ),
                                        additional_inputs=[radtio_ptions_summary, input_indexname_summary, textbox_systemMessage_summary, checkbox_Stream_summary], 
                                        submit_btn="Chat",                                    
                                        concurrency_limit="default" ,                             
                                        examples = [["provide summary for the document"],["give me insights of the document"]])
                    interface.queue(default_concurrency_limit=Build_Concurrency,max_size=Max_Queue_Size)  

        app = gr.mount_gradio_app(app, custom_theme_summaryIndexV2, path="/buildrunsummaryindex")

if BuildRRIndex_Tab is True:
    with gr.Blocks(title="Build and Run Recursive Retriever Index",analytics_enabled=False, css=css, js=js,theme=gr.themes.Default(spacing_size="sm", radius_size="none", primary_hue="blue")).queue(default_concurrency_limit=Build_Concurrency,max_size=Max_Queue_Size) as custom_theme_rrIndexV2:
        input_indexname_rr = gr.Textbox(label="Index Name",placeholder= "Recursive Retriever Index folder path generated by Index building",lines=1)   
        with gr.Tab("Build Index"): 
                downloadbutton_rr = gr.DownloadButton(label="Download Index")
                NodeReferenceCheckBox_rr = gr.Checkbox(label="Node Reference", value=False)
                interface = gr.Interface(fn=build_RecursiveRetriever_index, inputs=["file",NodeReferenceCheckBox_rr], outputs=["markdown",downloadbutton_rr],allow_flagging="never",analytics_enabled=False)
                interface.queue(default_concurrency_limit=Build_Concurrency,max_size=Max_Queue_Size) 

        if Individual_Chat is True:
            with gr.Tab("Chat Mode"):
                    radtio_options_rr = gr.Radio([RECURSIVE_RETRIEVER], label="Index Type", value=RECURSIVE_RETRIEVER, visible=False)
                    with gr.Accordion("Chat Settings", open=False):              
                        textbox_systemMessage_rr = gr.Textbox(default_system_message, label="System Message",visible=True, lines=5)
                    checkbox_Stream_rr = gr.Checkbox(label="Streaming", value=True)
                    interface = gr.ChatInterface(fn=chat_bot,
                                        chatbot= gr.Chatbot(likeable=False,
                                                show_share_button=False, 
                                                show_copy_button=True, 
                                                bubble_full_width = False,
                                                render=True
                                                ),
                                        additional_inputs=[radtio_options_rr, input_indexname_rr, textbox_systemMessage_rr, checkbox_Stream_rr], 
                                        submit_btn="Chat",                                
                                        examples = [["provide summary for the document"],["give me insights of the document"]])
                    interface.queue(default_concurrency_limit=Build_Concurrency,max_size=Max_Queue_Size)    


    app = gr.mount_gradio_app(app, custom_theme_rrIndexV2, path="/buildrunrrindex")

if BuildKGIndex_Tab is True:
    with gr.Blocks(title="Build and Run Knowledge Graph Index",analytics_enabled=False, css=css, js=js,theme=gr.themes.Default(spacing_size="sm", radius_size="none", primary_hue="blue")).queue(default_concurrency_limit=Build_Concurrency,max_size=Max_Queue_Size) as custom_theme_indexV2:
        #interface = gr.Interface(fn=proof_read, inputs=["file"],outputs="markdown",css="footer{display:none !important}",allow_flagging="never")
        input_indexname_kg = gr.Textbox(label="Index Name",placeholder= "Knowledge Graph Index folder path generated by Index building",lines=1)   
        with gr.Tab("Build Index"):             
                    downloadbutton_kg = gr.DownloadButton(label="Download Index")
                    interface = gr.Interface(fn=build_index, inputs=["file"], outputs=["markdown",downloadbutton_kg],allow_flagging="never",analytics_enabled=False)
                    interface.queue(default_concurrency_limit=Build_Concurrency,max_size=Max_Queue_Size) 
        with gr.Tab("View Index"):
                    downloadgraphviewbutton = gr.DownloadButton(label="Download Graph View")
                    interface = gr.Interface(fn=view_graph, inputs=[input_indexname_kg], outputs=["markdown",downloadgraphviewbutton],allow_flagging="never",analytics_enabled=False)
                    interface.queue(default_concurrency_limit=Build_Concurrency,max_size=Max_Queue_Size) 
        if Individual_Chat is True:
            with gr.Tab("Chat Mode"):
                    radtio_options_kg = gr.Radio([KNOWLEDGE_GRAPH], label="Index Type", value=KNOWLEDGE_GRAPH, visible=False)
                    with gr.Accordion("Chat Settings", open=False):              
                        textbox_systemMessage_kg = gr.Textbox(default_system_message, label="System Message",visible=True, lines=5)
                    checkbox_Stream_kg = gr.Checkbox(label="Streaming", value=True)
                    interface = gr.ChatInterface(fn=chat_bot,
                                        chatbot= gr.Chatbot(likeable=False,
                                                show_share_button=False, 
                                                show_copy_button=True, 
                                                bubble_full_width = False,
                                                render=True
                                                ),
                                        additional_inputs=[radtio_options_kg, input_indexname_kg, textbox_systemMessage_kg, checkbox_Stream_kg], 
                                        submit_btn="Chat",                                
                                        examples = [["provide summary for the document"],["give me insights of the document"]])  
                    interface.queue(default_concurrency_limit=Build_Concurrency,max_size=Max_Queue_Size) 

    app = gr.mount_gradio_app(app, custom_theme_indexV2, path="/buildrunragindex")

if BuildGraphRAGIndex_Tab is True:
    with gr.Blocks(title="Build and Run MS GraphRAG Index",analytics_enabled=False, css=css, js=js,theme=gr.themes.Default(spacing_size="sm", radius_size="none", primary_hue="blue")).queue(default_concurrency_limit=Build_Concurrency,max_size=Max_Queue_Size) as custom_theme_GraphRAGIndexV2:
        textbox_GraphRAGIndex_msrag =  gr.Textbox(lines=1, label="MS GraphRAG Index Name", value="msgrag01")
        with gr.Tab("Build Index"): 
                textbox_GraphRAGStorageName_msrag =  gr.Textbox(lines=1, label="MS GraphRAG Storage Name", value="msgrag01")            
                interface = gr.Interface(fn=build_GraghRAG_index, inputs=["file",textbox_GraphRAGStorageName_msrag, textbox_GraphRAGIndex_msrag], outputs=["markdown"],allow_flagging="never",analytics_enabled=False)
                interface.queue(default_concurrency_limit=Build_Concurrency,max_size=Max_Queue_Size) 
        if Individual_Chat is True:
            with gr.Tab("Chat Mode"):
                    radtio_options_msrag = gr.Radio([MS_GRAPHRAG_LOCAL,MS_GRAPHRAG_GLOBAL], label="Index Type", value=MS_GRAPHRAG_LOCAL)
                    with gr.Accordion("Chat Settings", open=False):              
                        textbox_systemMessage_msrag = gr.Textbox(default_system_message, label="System Message",visible=True, lines=5)
                    checkbox_Stream_msrag = gr.Checkbox(label="Streaming", value=True)
                    interface = gr.ChatInterface(fn=chat_bot,
                                        chatbot= gr.Chatbot(likeable=False,
                                                show_share_button=False, 
                                                show_copy_button=True, 
                                                bubble_full_width = False,
                                                render=True
                                                ),
                                        additional_inputs=[radtio_options_msrag, textbox_GraphRAGIndex_msrag, textbox_systemMessage_msrag, checkbox_Stream_msrag], 
                                        submit_btn="Chat",                              
                                        examples = [["provide summary for the document"],["give me insights of the document"]])
                    interface.queue(default_concurrency_limit=Build_Concurrency,max_size=Max_Queue_Size)   

    app = gr.mount_gradio_app(app, custom_theme_GraphRAGIndexV2, path="/buildrunGraphRAGindex")

def updateFileName(file):
     return file

file_upload_csv = gr.File(label="Upload CSV File",file_types=["csv"])
textbox_csv_file =  gr.Textbox(lines=1, label="CSV File Path") 

if CSVQueryEngine_Tab is True:
    with gr.Blocks(title="CSV Query Engine",analytics_enabled=False, css=css, js=js,theme=gr.themes.Default(spacing_size="sm", radius_size="none", primary_hue="blue")).queue(default_concurrency_limit=Build_Concurrency,max_size=Max_Queue_Size) as custom_theme_CSVAnalysis:
        textbox_csv_chat_file =  gr.Textbox(lines=1, label="CSV File Path for Chat Mode") 
        with gr.Tab("Upload CSV File"):            
                interface = gr.Interface(fn=CSV_Load, inputs=[file_upload_csv], outputs=[textbox_csv_file,"markdown"],allow_flagging="never",analytics_enabled=False)
                interface.queue(default_concurrency_limit=Build_Concurrency,max_size=Max_Queue_Size) 
                textbox_csv_file.change(updateFileName,textbox_csv_file ,textbox_csv_chat_file)
        with gr.Tab("Chat Mode"): 
                with gr.Accordion("Chat Settings", open=False):              
                        textbox_systemMessage_csv = gr.Textbox(default_system_message, label="System Message",visible=True, lines=5)
                radtio_ptions_csv = gr.Radio([CSV_AI_ANALYSIS], label="Index Type", value=CSV_AI_ANALYSIS, visible=False)           
                checkbox_Stream_csv = gr.Checkbox(label="Streaming", value=True, visible=False)
                interface = gr.ChatInterface(fn=chat_bot,
                            chatbot= gr.Chatbot(likeable=False,
                                                show_share_button=False, 
                                                show_copy_button=True, 
                                                bubble_full_width = False,
                                                render=True
                                                ),
                            additional_inputs=[radtio_ptions_csv, textbox_csv_chat_file, textbox_systemMessage_csv, checkbox_Stream_csv], 
                            submit_btn="Chat",                              
                            examples = [["how many records does it have"],["give me insights of the document"]])
                interface.queue(default_concurrency_limit=Build_Concurrency,max_size=Max_Queue_Size)   

    app = gr.mount_gradio_app(app, custom_theme_CSVAnalysis, path="/ChatWithCSV")

if __name__ == '__main__':
    multiprocessing.freeze_support()  # For Windows support
    uvicorn.run(app, host="0.0.0.0", port=os.environ["PORT"], reload=False, workers=1)