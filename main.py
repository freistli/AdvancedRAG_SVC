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
from IPython.display import display, HTML
from dotenv import load_dotenv
from fastapi.responses import HTMLResponse, RedirectResponse
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
from llama_index.core.postprocessor.optimizer import SentenceEmbeddingOptimizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import KnowledgeGraphRAGRetriever
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core.callbacks import LlamaDebugHandler
import tiktoken
from IndexGenerator import IndexGenerator, StreamingGradioCallbackHandler
from AzureSearchIndexGenerator import AzureAISearchIndexGenerator
from RecursiveRetrieverIndexGenerator import RecursiveRetrieverIndexGenerator
from SummmaryIndexGenerator import SummaryIndexGenerator
import networkx as nx
from pyvis.network import Network

optimizer = SentenceEmbeddingOptimizer(
    percentile_cutoff=0.5,
    threshold_cutoff=0.7,
    context_before=1,
    context_after=1
)

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
graph_store = None
query_engine = None
rulesPath_query_engine = None
rulesPath_index = None
current_fine_Tune = None
current_graph_path = None
proofread_chunk = 500


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


load_dotenv('.env_4_SC')

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
    content = "Criticize the proofread content, especially for wrong words. Only use 当社の用字・用語の基準,  送り仮名の付け方, 現代仮名遣い,  接続詞の使い方 ，外来語の書き方，公正競争規約により使用を禁止されている語  製品の取扱説明書等において使用することはできない, 常用漢字表に記載されていない読み方, and 誤字 proofread rules, don't use other rules those are not in the retrieved documents.                Pay attention to some known issues:もっとも, または->又は, 「ただし」という接続詞は原則として仮名で表記するため,「又は」という接続詞は原則として漢字で表記するため。また、「又は」は、最後の語句に“など”、「等(とう)」又は「その他」を付けてはならない, 優位性を意味する語.               Firstly show 原文, use bold text to point out every incorrect issue, and then give 校正理由, respond in Japanese. Finally give 修正後の文章, use bold text for modified text. If everything is correct, tell no issues, and don't provide 校正理由 or 修正後の文章."
)
message = HumanMessage(
    content=prompt
)



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

def compose_query(Graph, QueryRules, content, fine_tune=None, content_prefix="以下の文章を校正してください: "):

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
                graph_traversal_depth=fine_tune.graph_traversal_depth,
                max_entities=fine_tune.max_entities,
                max_synonyms=fine_tune.max_synonyms,
                max_knowledge_sequence=fine_tune.max_knowledge_sequence,
                verbose=True )

            query_engine = RetrieverQueryEngine.from_args(
                #graph_rag_retriever,
                temp_retriever,
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
                    temp_retriever,
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

def proof_read_addin(Content: str = ""):
    result = proof_read("rules", systemMessage.content, Content, "", 5, 5, 2, 30, None, True)
    for text in result:
       yield text
    

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
        with open(Draft, 'rb') as file:
            file_content = file.read()

        to_be_proofread_content = docClient.begin_analyze_document(
            "prebuilt-layout",
            analyze_request=AnalyzeDocumentRequest(bytes_source=file_content),
            output_content_format="markdown"
        )

        docs_string = to_be_proofread_content.result().content

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
    for text in result:
       yield text
 
def chat_bot(message, history, indexType, indexName, systemMessage):
    history_openai_format = []
    history_openai_format.append({"role": "system", "content": systemMessage})
    for human, assistant in history:
            history_openai_format.append({"role": "user", "content": human })
            history_openai_format.append({"role": "assistant", "content":assistant})
    history_openai_format.append({"role": "user", "content": message})

    if indexType == "Azure AI Search":
        azureAISearchIndexGenerator = AzureAISearchIndexGenerator(docPath="", indexName=indexName, idPrefix="")
        azureAISearchIndexGenerator.LoadIndex()
        response = azureAISearchIndexGenerator.HybridSearch(str(history_openai_format))
        for text in response:
            yield text
    elif indexType == "Knowledge Graph":
        response = KnowledgeGraphIndexSearch(indexName, str(history_openai_format), message)
        for text in response:
            yield text
    elif indexType == "Recursive Retriever":
        recursiveRetrieverIndexGenerator = RecursiveRetrieverIndexGenerator (docPath="", indexName="", idPrefix="")
        result = recursiveRetrieverIndexGenerator.LoadIndex(indexFolder=indexName)
        for text in result:
            pass
        response = recursiveRetrieverIndexGenerator.RecursiveRetrieverSearch(str(history_openai_format))
        for text in response:
            yield text
    elif indexType == "Summary Index":
        summaryIndexGenerator = SummaryIndexGenerator (docPath="", indexName="", idPrefix="")
        result = summaryIndexGenerator.LoadIndex(indexFolder=indexName)
        for text in result:
            pass
        response = summaryIndexGenerator.SummaryRetrieverSearch(str(history_openai_format))
        for text in response:
            yield text
       
def print_like_dislike(x: gr.LikeData):
    print(x.index, x.value, x.liked)

max_entities = gr.Slider(label="Max Entities", value=5, minimum=1, maximum=10, step=1)
max_synonyms = gr.Slider(label="Max Synonyms", value=5, minimum=1, maximum=10, step=1)
graph_traversal_depth = gr.Slider(label="Graph Traversal Depth", value=2, minimum=1, maximum=10, step=1)
max_knowledge_sequence = gr.Slider(label="Max Knowledge Sequence", value=30, minimum=1, maximum=50, step=1)

texbox_Rules = gr.Textbox(lines=1, label="Knowledge Graph of Proofreading Rules (rules, train, compose)", value="rules")
textbox_QueryRules = gr.Textbox(lines=10, label="Preset Query Prompt", value=systemMessage.content)
textbox_Content = gr.Textbox(lines=10, elem_id="ProofreadContent", label="Content to be Proofread", value="今回は半導体製造装置セクターの最近の動きを分析します。このセクターが成長性のあるセクターであるという意見は変えません。また、後工程（テスタ、ダイサなど）は2023年4-6月期、前工程（ウェハプロセス装置）は7-9月期または 10-12月期等 で大底を打ち、その後は回復、再成長に向かうと思われます。但し 、足元ではいくつか問題も出ています。")
textbox_Content_Empty = gr.Textbox(lines=10)
textbox_AzureSearchIndex =  gr.Textbox(lines=1, label="Azure AI Search Index Name", value="azuresearch_0")

downloadbutton = gr.DownloadButton(label="Download Index")
downloadgraphbutton = gr.DownloadButton(label="Download Graph View")
downloadproofreadbutton = gr.DownloadButton(label="Download Proofread Result")
downloadRRbutton = gr.DownloadButton(label="Download Index")
downloadSummarybutton = gr.DownloadButton(label="Download Index")

NodeReferenceCheckBox = gr.Checkbox(label="Node Reference", value=False)


modelName = "Azure OpenAI GPT-4o"

with gr.Blocks(title=f"Advanced Proofreading by {modelName}",analytics_enabled=False, css="footer{display:none !important}", js=js,theme=gr.themes.Default(spacing_size="sm", radius_size="none", primary_hue="blue")).queue(default_concurrency_limit=3,max_size=20) as custom_theme:
    #interface = gr.Interface(fn=proof_read, inputs=["file"],outputs="markdown",css="footer{display:none !important}",allow_flagging="never")
    interface = gr.Interface(fn=proof_read, inputs=[texbox_Rules, 
                                                    textbox_QueryRules, 
                                                    textbox_Content_Empty,                                                    
                                                    "file",
                                                    max_entities,
                                                    max_synonyms,
                                                    graph_traversal_depth,
                                                    max_knowledge_sequence,
                                                    ], outputs=["markdown",downloadproofreadbutton],allow_flagging="never",analytics_enabled=False)

app = gr.mount_gradio_app(app, custom_theme, path="/advproofread")


with gr.Blocks(title=f"Proofreading by {modelName}",analytics_enabled=False, css="footer{display:none !important}", js=js,theme=gr.themes.Default(spacing_size="sm", radius_size="none", primary_hue="blue")).queue(default_concurrency_limit=3,max_size=20) as custom_theme_base:
    #interface = gr.Interface(fn=proof_read, inputs=["file"],outputs="markdown",css="footer{display:none !important}",allow_flagging="never")
    interface = gr.Interface(fn=proof_read, inputs=[texbox_Rules, 
                                                    textbox_QueryRules,
                                                    textbox_Content], outputs=["markdown",downloadproofreadbutton],allow_flagging="never",analytics_enabled=False)



app = gr.mount_gradio_app(app, custom_theme_base, path="/proofread")

with gr.Blocks(title=f"Proofreading by {modelName}",analytics_enabled=False, css="footer{display:none !important}", js=js,theme=gr.themes.Default(spacing_size="sm", radius_size="none", primary_hue="blue")).queue(default_concurrency_limit=3,max_size=20) as custom_theme_addin:
    #interface = gr.Interface(fn=proof_read, inputs=["file"],outputs="markdown",css="footer{display:none !important}",allow_flagging="never")
    button = gr.Button("Choose Selected Content",elem_id="ChooseSelectedContent")
    interface = gr.Interface(fn=proof_read_addin, inputs=[textbox_Content], outputs=["markdown"],allow_flagging="never",analytics_enabled=False)


app = gr.mount_gradio_app(app, custom_theme_addin, path="/proofreadaddin")

with gr.Blocks(title="Build Knowledge Graph Index",analytics_enabled=False, css="footer{display:none !important}", js=js,theme=gr.themes.Default(spacing_size="sm", radius_size="none", primary_hue="blue")).queue(default_concurrency_limit=3,max_size=20) as custom_theme_index:
    #interface = gr.Interface(fn=proof_read, inputs=["file"],outputs="markdown",css="footer{display:none !important}",allow_flagging="never")
    interface = gr.Interface(fn=build_index, inputs=["file"], outputs=["markdown",downloadbutton],allow_flagging="never",analytics_enabled=False)


app = gr.mount_gradio_app(app, custom_theme_index, path="/buildragindex")

with gr.Blocks(title="View Knowledge Graph Index",analytics_enabled=False, css="footer{display:none !important}", js=js,theme=gr.themes.Default(spacing_size="sm", radius_size="none", primary_hue="blue")).queue(default_concurrency_limit=3,max_size=20) as custom_theme_viewgraph:
    #interface = gr.Interface(fn=proof_read, inputs=["file"],outputs="markdown",css="footer{display:none !important}",allow_flagging="never")
    interface = gr.Interface(fn=view_graph, inputs=["text"], outputs=["markdown",downloadgraphbutton],allow_flagging="never",analytics_enabled=False)


app = gr.mount_gradio_app(app, custom_theme_viewgraph, path="/viewgraph")

with gr.Blocks(title="Build Index on Azure AI Search",analytics_enabled=False, css="footer{display:none !important}", js=js,theme=gr.themes.Default(spacing_size="sm", radius_size="none", primary_hue="blue")).queue(default_concurrency_limit=3,max_size=20) as custom_theme_AzureSearch:
    #interface = gr.Interface(fn=proof_read, inputs=["file"],outputs="markdown",css="footer{display:none !important}",allow_flagging="never")
    interface = gr.Interface(fn=build_azure_index, inputs=[textbox_AzureSearchIndex,"file"], outputs=["markdown"],allow_flagging="never",analytics_enabled=False)


app = gr.mount_gradio_app(app, custom_theme_AzureSearch, path="/buildazureindex")


with gr.Blocks(title="Build Recursive Retriever Index",analytics_enabled=False, css="footer{display:none !important}", js=js,theme=gr.themes.Default(spacing_size="sm", radius_size="none", primary_hue="blue")).queue(default_concurrency_limit=3,max_size=20) as custom_theme_rrIndex:
    #interface = gr.Interface(fn=proof_read, inputs=["file"],outputs="markdown",css="footer{display:none !important}",allow_flagging="never")
    interface = gr.Interface(fn=build_RecursiveRetriever_index, inputs=["file",NodeReferenceCheckBox], outputs=["markdown",downloadRRbutton],allow_flagging="never",analytics_enabled=False)


app = gr.mount_gradio_app(app, custom_theme_rrIndex, path="/buildrrindex")


with gr.Blocks(title="Build Summary Index",analytics_enabled=False, css="footer{display:none !important}", js=js,theme=gr.themes.Default(spacing_size="sm", radius_size="none", primary_hue="blue")).queue(default_concurrency_limit=3,max_size=20) as custom_theme_summaryIndex:
    #interface = gr.Interface(fn=proof_read, inputs=["file"],outputs="markdown",css="footer{display:none !important}",allow_flagging="never")
    interface = gr.Interface(fn=build_Summary_index, inputs=["file"], outputs=["markdown",downloadSummarybutton],allow_flagging="never",analytics_enabled=False)


app = gr.mount_gradio_app(app, custom_theme_summaryIndex, path="/buildsummaryindex")


chatbot = gr.Chatbot(likeable=True,
                            show_share_button=True, 
                            show_copy_button=True, 
                            bubble_full_width = False,
                            )


with gr.Blocks(title=f"Chat with {modelName}",analytics_enabled=False, css="footer{display:none !important}", js=js,theme=gr.themes.Default(spacing_size="sm", radius_size="none", primary_hue="blue")).queue(default_concurrency_limit=3,max_size=20) as custom_theme_ChatBot:
    #interface = gr.Interface(fn=proof_read, inputs=["file"],outputs="markdown",css="footer{display:none !important}",allow_flagging="never")
    with gr.Row():    
        with gr.Column(scale=1):
            with gr.Accordion("Chatbot Configuration", open=True):
                radtio_ptions = gr.Radio(["Azure AI Search","Knowledge Graph", "Recursive Retriever", "Summary Index"], label="Index Type", value="Azure AI Search")
                textbox_index = gr.Textbox("azuresearch_0", label="Search Index Name, can be index folders or Azure AI Search Index Name")
                textbox_systemMessage = gr.Textbox("You are helpful AI.", label="System Message",visible=True, lines=9)

        with gr.Column(scale=3): 
            chatbot.like(print_like_dislike,None, None)                   
            bot = gr.ChatInterface(chat_bot,
                             chatbot=chatbot,
                             additional_inputs=[radtio_ptions, textbox_index, textbox_systemMessage], 
                             examples = [["provide summary for the document"],["give me insights of the document"]])    


app = gr.mount_gradio_app(app, custom_theme_ChatBot, path="/advchatbot")


#custom_theme_rrIndex.launch()