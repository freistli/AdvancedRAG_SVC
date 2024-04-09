
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
from llama_index.core.postprocessor.optimizer import SentenceEmbeddingOptimizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import KnowledgeGraphRAGRetriever
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core.callbacks import LlamaDebugHandler

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
current_fine_Tune = None

job_done = object() # signals the processing is done

q = SimpleQueue()

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

import tiktoken


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
llama_debug = LlamaDebugHandler(print_trace_on_end=True)
token_counter = TokenCountingHandler(
    tokenizer=tiktoken.encoding_for_model('gpt-35-turbo').encode
)
Settings.callback_manager = CallbackManager([token_counter,llama_debug])


docClient = DocumentIntelligenceClient(endpoint, AzureKeyCredential(key))

use_storage = True
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
    content = "Criticize the proofread content, especially for wrong words. Only use 当社の用字・用語の基準,  送り仮名の付け方, 現代仮名遣い,  接続詞の使い方 ，外来語の書き方，公正競争規約により使用を禁止されている語  製品の取扱説明書等において使用することはできない, 常用漢字表に記載されていない読み方, and 誤字 proofread rules, don't use other rules those are not in the retrieved documents.                Pay attention to some known issues:もっとも, または->又は, 「ただし」という接続詞は原則として仮名で表記するため,「又は」という接続詞は原則として漢字で表記するため。また、「又は」は、最後の語句に“など”、「等(とう)」又は「その他」を付けてはならない, 優位性を意味する語.               Firstly show 原文, use bold text to point out every incorrect issue, and then give 校正理由, respond in Japanese. Finally give 修正後の文章, use bold text for modified text. If everything is correct, tell no issues, and don't provide 校正理由 or 修正後の文章."
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

def DebugLlama():
    event_pairs = llama_debug.get_llm_inputs_outputs()
    try:
        print('\n\n=========================================\n\n')
        print(event_pairs)          
    except:
        print("Error")
    llama_debug.flush_event_logs()

def compose_query(Graph, QueryRules, content, fine_tune=None):

    global rules_index
    global train_index
    global composeGraph
    global graph_store
    global query_engine
    global current_fine_Tune

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
            graph_traversal_depth=3, similarity_top_k=5).query(QueryRules + "\r\n 以下の文章を校正してください:  \r\n "+content)

    if Graph == "rules":
        logging.info("rules")
        #if rules_index is None:
        if query_engine is None:
                if os.path.exists(rules_storage_dir+"/docstore.json") and use_storage:
                    #storage_context = StorageContext.from_defaults(graph_store=graph_store,persist_dir=persist_dir)

                    if graph_store is None:
                        graph_store = SimpleGraphStore().from_persist_dir(rules_storage_dir)                    
                    storage_context = StorageContext.from_defaults(graph_store=graph_store,persist_dir=rules_storage_dir)
                    rules_index = load_index_from_storage(storage_context)

        if current_fine_Tune is None or current_fine_Tune != fine_tune:

            print("Fine Tune changed")

            storage_context = StorageContext.from_defaults(graph_store=graph_store,persist_dir=rules_storage_dir)
            '''
            graph_rag_retriever = KnowledgeGraphRAGRetriever(
                storage_context=storage_context,
                llm=Settings.llm,
                #entity_extract_template="",
                #synonym_expand_template="",
                graph_traversal_depth=fine_tune.graph_traversal_depth,
                max_entities=fine_tune.max_entities,
                max_synonyms=fine_tune.max_synonyms,
                max_knowledge_sequence=fine_tune.max_knowledge_sequence,
                verbose=True          
            )
            '''

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
        response_stream = query_engine.query(QueryRules + "\r\n 以下の文章を校正してください:  \r\n "+content) 

        DebugLlama()
        
        
    
    if Graph == "train":
        logging.info("train")
        if train_index is None:
                if os.path.exists(train_storage_dir+"/docstore.json") and use_storage:
                    #train_storage_context = StorageContext.from_defaults(graph_store=graph_store,persist_dir=train_persist_dir)
                    train_storage_context = StorageContext.from_defaults(persist_dir=train_storage_dir)
                    train_index = load_index_from_storage(train_storage_context)
        response_stream = train_index.as_query_engine(streaming=True,graph_traversal_depth=3, verbose=True).query(QueryRules + "\r\n 以下の文章を校正してください:  \r\n "+content)
    
    partial_message = ""
    for text in response_stream.response_gen:
        partial_message = partial_message + text
        # do something with text as they arrive.
        yield partial_message
    DebugLlama()


def proof_read (Graph, QueryRules, Content: str = "" ,Draft: str = "", max_entities: int = 5, max_synonyms: int = 5,graph_traversal_depth: int = 2, max_knowledge_sequence: int = 30):
    
    begin = time.time()

    token_counter.reset_counts()

    if str(QueryRules).strip() == "":
        QueryRules = systemMessage.content

    if str(Graph).strip() == "":
        Graph = "compose"

    
    # Get the current time
    now = datetime.now()

    # Format the current time as a string, including milliseconds
    filename = now.strftime("%Y%m%d%H%M%S%f")
    
    partial_message = ""

    if str(Content).strip() != "":
        logging.info(Content)

        textSplitter = CharacterTextSplitter(chunk_size=350, separator="\n", is_separator_regex=False)

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

        textSplitter = CharacterTextSplitter(chunk_size=350, separator="\n", is_separator_regex=False)

        to_be_proofread_content_list = textSplitter.split_text(docs_string)        

    fine_tune = FineTune(max_entities, max_synonyms,graph_traversal_depth,max_knowledge_sequence)
    
    with open(os.path.join(persist_dir, filename + '_proofread_result.md'), 'a') as f:
        for i in range(len(to_be_proofread_content_list)):             
            status = "\n\nTime elapsed: " +str(time.time() - begin)+ "\n\nPart: "+str(i+1) + " of " + str(len(to_be_proofread_content_list))+" :"
            try:                        
                logging.info(status)
                logging.info("processing content: " + to_be_proofread_content_list[i])
                response = compose_query(Graph, QueryRules, to_be_proofread_content_list[i],fine_tune)
                currentProofRead = ""
                for text in response:
                    status = "\n\nTime elapsed: " +str(time.time() - begin)+ "\n\nPart: "+str(i+1) + " of " + str(len(to_be_proofread_content_list))+" :"
                    currentProofRead = status + "\n\n" + text 
                    yield currentProofRead
                partial_message = currentProofRead
                f.write(partial_message)
                f.flush()
                logging.info(partial_message)
            except Exception as e:
                logging.error("Error: " + str(e))
                f.write(status)
                f.write("Error: " + str(e))
                f.flush()
                yield "Error: " + str(e)
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
    return RedirectResponse(url="/proofread")
 

max_entities = gr.Slider(label="Max Entities", value=5, minimum=1, maximum=10, step=1)
max_synonyms = gr.Slider(label="Max Synonyms", value=5, minimum=1, maximum=10, step=1)
graph_traversal_depth = gr.Slider(label="Graph Traversal Depth", value=2, minimum=1, maximum=10, step=1)
max_knowledge_sequence = gr.Slider(label="Max Knowledge Sequence", value=30, minimum=1, maximum=50, step=1)

texbox_Rules = gr.Textbox(lines=1, label="Knowledge Graph of Proofreading Rules (rules, train, compose)", value="rules")
textbox_QueryRules = gr.Textbox(lines=10, label="Preset Query Prompt", value=systemMessage.content)
textbox_Content = gr.Textbox(lines=10, label="Content to be Proofread", value="今回は半導体製造装置セクターの最近の動きを分析します。このセクターが成長性のあるセクターであるという意見は変えません。また、後工程（テスタ、ダイサなど）は2023年4-6月期、前工程（ウェハプロセス装置）は7-9月期または 10-12月期等 で大底を打ち、その後は回復、再成長に向かうと思われます。但し 、足元ではいくつか問題も出ています。")


with gr.Blocks(title="Proofreading by AI",analytics_enabled=False, css="footer{display:none !important}", js=js,theme=gr.themes.Default(spacing_size="sm", radius_size="none", primary_hue="blue")).queue(default_concurrency_limit=3,max_size=20) as custom_theme:
    #interface = gr.Interface(fn=proof_read, inputs=["file"],outputs="markdown",css="footer{display:none !important}",allow_flagging="never")
    interface = gr.Interface(fn=proof_read, inputs=[texbox_Rules, 
                                                    textbox_QueryRules, 
                                                    textbox_Content,                                                    
                                                    "file",
                                                    max_entities,
                                                    max_synonyms,
                                                    graph_traversal_depth,
                                                    max_knowledge_sequence,
                                                    ], outputs=["markdown"],allow_flagging="never",analytics_enabled=False)

app = gr.mount_gradio_app(app, custom_theme, path="/advproofread")


with gr.Blocks(title="Proofreading by AI",analytics_enabled=False, css="footer{display:none !important}", js=js,theme=gr.themes.Default(spacing_size="sm", radius_size="none", primary_hue="blue")).queue(default_concurrency_limit=3,max_size=20) as custom_theme_base:
    #interface = gr.Interface(fn=proof_read, inputs=["file"],outputs="markdown",css="footer{display:none !important}",allow_flagging="never")
    interface = gr.Interface(fn=proof_read, inputs=[texbox_Rules, 
                                                    textbox_QueryRules,
                                                    textbox_Content,                                                    
                                                    "file"], outputs=["markdown"],allow_flagging="never",analytics_enabled=False)

app = gr.mount_gradio_app(app, custom_theme_base, path="/proofread")

#custom_theme.launch()