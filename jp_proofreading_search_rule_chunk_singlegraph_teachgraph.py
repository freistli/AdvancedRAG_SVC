
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

"""
# Initiate Azure AI Document Intelligence to load the document. You can either specify file_path or url_path to load the document.
loader = AzureAIDocumentIntelligenceLoader(file_path=".//rules//English.docx", api_key=key, api_endpoint=endpoint,
                                           api_model="prebuilt-layout")
docs = loader.load()
"""


#langchain_openai
client = AzureChatOpenAI(
    openai_api_version="2023-12-01-preview",
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

docClient = DocumentIntelligenceClient(endpoint, AzureKeyCredential(key))

#setup the storage context

graph_store = SimpleGraphStore()

use_storage = True
batch_size = 10


def InitKGRulesIndex(ruleFilePath, persist_dir, graph_store, use_storage, batch_size):
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

        """
        # Split the document into chunks base on markdown headers.
        headers_to_split_on = [
        ("\n", "Paragraph")
        ]
        #splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        """

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
                                        max_triplets_per_chunk=3,
                        storage_context=storage_context,
                        include_embeddings=True,
                        show_progress=True,
                        #index_id="rules_index")
                        index_id="train_index")
        
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
        
        storage_context = StorageContext.from_defaults(graph_store=graph_store,persist_dir=persist_dir)
        rules_index = load_index_from_storage(storage_context)
        return rules_index

    """
    rules_index = KnowledgeGraphIndex.from_documents( documents=docs,
                                        max_triplets_per_chunk=3,
                                        storage_context=storage_context,
                                        include_embeddings=True)
    rules_index.storage_context.persist(persist_dir=persist_dir)
    """
    """
    storage_context = StorageContext.from_defaults()
    rules_index = VectorStoreIndex.from_documents(documents=docs, storage_context=storage_context)
    rules_index.storage_context.persist(persist_dir=persist_dir)
    """

"""
loader = AzureAIDocumentIntelligenceLoader(file_path="C:\\Users\\freistli\\OneDrive - Microsoft\\Copilot\\Draft.docx", api_key=key, api_endpoint=endpoint,
                                           api_model="prebuilt-layout")
docs = loader.load()

docs_string = docs[0].page_content

splitter = MarkdownTextSplitter.from_tiktoken_encoder(chunk_size=512, chunk_overlap=128)
to_be_proofread_content_list = splitter.split_text(docs_string)

print(to_be_proofread_content_list)

prompt="Based on the proof read rule:"+rules_content_list[0]+"\r\n"+"provide detailed correction on these paragraphs:\r\n"+to_be_proofread_content_list[0]
""" 
prompt = ""
systemMessage = SystemMessage(
    #content="You are a Proof Read expert, can point out the accurate content issues with original paragraph, correctd paragraph, and reason bullets"
    content = "Based indexed Japanese proofread rules knowledge graph, use characters and sentance structures from the proofread content as clues to find out related Japanese proofread rules to validate content and point out exact pitfalls in Japanese\
        The response contains three sections: \
         **Proofread**. In this section, point out the accurate proofread issues WITHIN original sentences one by one directly. If the sentences have no issues, keep the original sentances in the section. \
              MUST right behind of the proofread issue, use a superscript citation index number. \
              If it is a word issue, then use a superscript citation index number right behind the word. If it is a sentence issue, then use a superscript citation index number right behind the sentance. For example, 1 is ¹ , 2 is ².\
              **Corrections**. In this section, list all corrections associated with citation index numbers, also include original proofread rules which are exactly used. \
              **Corrected Result**. Implement all corrections, give the corrected content"
)
message = HumanMessage(
    content=prompt
)

"""
thread = Thread(target=client.invoke, kwargs={"input": [systemMessage,message]})
thread.start()
result = ""
while True:
    next_token = q.get(block=True)  # Blocks until an input is available
    if next_token is job_done:
        break
    result += next_token
    print(result,end='\r',flush=True)
thread.join()
"""

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

def compose_query(Graph, QueryRules, to_be_proofread_content_list):

    global rules_index
    global train_index
    global composeGraph

    if Graph == "compose": 
        logging.info("compose")
        if composeGraph is None:

            if rules_index is None:
                if os.path.exists(persist_dir+"/docstore.json") and use_storage:
                    storage_context = StorageContext.from_defaults(graph_store=graph_store,persist_dir=persist_dir)
                    rules_index = load_index_from_storage(storage_context)

            if train_index is None:
                if os.path.exists(train_persist_dir+"/docstore.json") and use_storage:
                    train_storage_context = StorageContext.from_defaults(graph_store=graph_store,persist_dir=train_persist_dir)
                    train_index = load_index_from_storage(train_storage_context)


        service_context = ServiceContext.from_defaults(
                llm=Settings.llm,
                embed_model=Settings.embed_model,
                node_parser=Settings.node_parser
            ) 
        
        storage_context = StorageContext.from_defaults(graph_store=graph_store,persist_dir=persist_dir)

        composeGraph = ComposableGraph.from_indices(
                    KnowledgeGraphIndex,
                    [rules_index,train_index],
                    index_summaries=['rules knowledge graph index','train knowledge graph index'],
                    service_context=service_context,
                    storage_context=storage_context,
                    include_embeddings=True)
        
        response = composeGraph.as_query_engine(verbose=True,max_entities=10,
            graph_traversal_depth=3, similarity_top_k=5).query(QueryRules + "\r\n 以下の文章を校正してください:  \r\n "+to_be_proofread_content_list[0])

    if Graph == "rules":
        logging.info("rules")
        if rules_index is None:
                if os.path.exists(persist_dir+"/docstore.json") and use_storage:
                    #storage_context = StorageContext.from_defaults(graph_store=graph_store,persist_dir=persist_dir)
                    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
                    rules_index = load_index_from_storage(storage_context)
        response = rules_index.as_query_engine(verbose=True, similarity_top_k=5).query(QueryRules + "\r\n 以下の文章を校正してください:  \r\n "+to_be_proofread_content_list[0])
    
    if Graph == "train":
        logging.info("train")
        if train_index is None:
                if os.path.exists(train_persist_dir+"/docstore.json") and use_storage:
                    #train_storage_context = StorageContext.from_defaults(graph_store=graph_store,persist_dir=train_persist_dir)
                    train_storage_context = StorageContext.from_defaults(persist_dir=train_persist_dir)
                    train_index = load_index_from_storage(train_storage_context)
        response = train_index.as_query_engine(verbose=True, similarity_top_k=5).query(QueryRules + "\r\n 以下の文章を校正してください:  \r\n "+to_be_proofread_content_list[0])
    
    return str(response)


def proof_read (Graph, QueryRules, Content,Draft):

    if str(QueryRules).strip() == "":
        QueryRules = systemMessage.content

    if str(Graph).strip() == "":
        Graph = "compose"

    if str(Content).strip() != "":
        logging.info(Content)

        textSplitter = MarkdownTextSplitter.from_tiktoken_encoder(chunk_size=1024)

        to_be_proofread_content_list = textSplitter.split_text(Content)
        
        response = compose_query(Graph, QueryRules, to_be_proofread_content_list)

        logging.info(str(response))

        return str(response)

        """
        for i in range(len(rules_content_list)):

            prompt = "Here are the proofread rules:" + rules_content_list[i] + "\r\n" + "Here is the content to be proofread: "+ "\r\n" + to_be_proofread_content_list[0] + "\r\nHere are the lastest proofread findings: "+ currentProofRead +"\r\n" \
                    + "Please provide detailed corrections in Japanese on the content, need to merge with the latest proofread findings  in the Proofread, Corrections and Corrected Results sections."

            message = HumanMessage(
                content=prompt
            )
            response = client.stream([systemMessage, message])
            partial_message = ""
            for chunk in response:
                partial_message = partial_message + chunk.dict()['content']
                currentProofRead = partial_message
                yield partial_message
        """

    if str(Draft).strip() != "":
        print(Draft)
        with open(Draft, 'rb') as file:
            file_content = file.read()

        to_be_proofread_content = docClient.begin_analyze_document(
            "prebuilt-layout",
            analyze_request=AnalyzeDocumentRequest(bytes_source=file_content),
            output_content_format="markdown"
        )

        docs_string = to_be_proofread_content.result().content

        splitter = MarkdownTextSplitter.from_tiktoken_encoder(chunk_size=1024)
        to_be_proofread_content_list = splitter.split_text(docs_string)

        response = compose_query(QueryRules, to_be_proofread_content_list)

        logging.info(str(response))

        return str(response)


js = """
function createGradioAnimation() {
    var container = document.createElement('div');
    container.id = 'gradio-animation';
    container.style.fontSize = '2em';
    container.style.fontWeight = 'bold';
    container.style.textAlign = 'center';
    container.style.marginBottom = '20px';

    var text = 'Proofreading by Azure OpenAI GPT-4 Turbo';
    for (var i = 0; i < text.length; i++) {
        (function(i){
            setTimeout(function(){
                var letter = document.createElement('span');
                letter.style.opacity = '0';
                letter.style.transition = 'opacity 0.2s';
                letter.innerText = text[i];
                container.appendChild(letter);
                setTimeout(function() {
                    letter.style.opacity = '1';
                }, 50);
            }, i * 50);
        })(i);
    }
var gradioContainer = document.querySelector('.gradio-container');
    gradioContainer.insertBefore(container, gradioContainer.firstChild);

    return 'Animation created';
}
"""

app = FastAPI()
@app.get("/")
def read_main():
    return {"message": "This is your main app"}

with gr.Blocks(title="Proofreading by AI",css="footer{display:none !important}", js=js,theme=gr.themes.Default(spacing_size="sm", radius_size="none", primary_hue="blue"),analytics_enabled=False) as custom_theme:
    #gr.ChatInterface(stream_predict).queue().launch()
    #interface = gr.Interface(fn=proof_read, inputs=["file"],outputs="markdown",css="footer{display:none !important}",allow_flagging="never")
    interface = gr.Interface(fn=proof_read, inputs=["text", "text", "text","file"], outputs=["markdown"],allow_flagging="never")

#app = gr.mount_gradio_app(app, custom_theme, path="/proofread")
custom_theme.launch()