'''
Functionality: Use Recursive Retriever method to generate index and search the index
Author: Freist Li
Date: 2024-05
Version: 1.0
'''
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from langchain_text_splitters import MarkdownTextSplitter
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from dotenv import load_dotenv

from llama_index.vector_stores.azureaisearch import AzureAISearchVectorStore, IndexManagement, MetadataIndexFieldType
from llama_index.core import StorageContext, VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import Settings,load_index_from_storage
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents import SearchClient

from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult, AnalyzeDocumentRequest
from llama_index.core import Document
from llama_index.core.storage.docstore import SimpleDocumentStore

from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core.callbacks import LlamaDebugHandler
import tiktoken
from llama_index.core.schema import IndexNode
from llama_index.core.retrievers import RecursiveRetriever
import pickle

load_dotenv('.env_4_SC')
#logging.basicConfig(stream=sys.stdout, level=logging.INFO,format='%(message)s')
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

class RecursiveRetriverIndexGenerator:
    def __init__(self, docPath, indexName, idPrefix):
        self.docPath = docPath
        self.idPrefix = idPrefix
        self.aoai_api_key = os.environ['AZURE_OPENAI_API_KEY']  
        self.aoai_endpoint = os.environ['AZURE_OPENAI_ENDPOINT']
        self.aoai_api_version = os.environ['AZURE_OPENAI_API_VERSION']
        self.aoai_modeldeploy_name = os.environ['AZURE_OPENAI_Deployment']

        self.docai_endpoint = os.environ['DOC_AI_BASE']
        self.docai_api_key = os.environ['DOC_AI_KEY']

        self.llm = AzureOpenAI(
            deployment_name = self.aoai_modeldeploy_name,
            api_key=self.aoai_api_key,
            azure_endpoint=self.aoai_endpoint,
            api_version=self.aoai_api_version,
        )
        
        self.embed_model = AzureOpenAIEmbedding(
            deployment_name="text-embedding-ada-002",
            api_key=self.aoai_api_key,
            azure_endpoint=self.aoai_endpoint,
            api_version=self.aoai_api_version,
        )

        self.search_service_api_key = os.environ['AZURE_SEARCH_API_KEY']
        self.search_service_endpoint = os.environ['AZURE_SEARCH_ENDPOINT']
        self.search_service_api_version = os.environ['AZURE_SEARCH_API_VERSION']

        self.index_name = indexName

        self.index_client = SearchIndexClient(
            endpoint=self.search_service_endpoint,
            credential= AzureKeyCredential(self.search_service_api_key)
        )

        self.searchClient = SearchClient(
            endpoint=self.search_service_endpoint,
            index_name=self.index_name,
            credential=AzureKeyCredential(self.search_service_api_key)
        )

        self.docClient = DocumentIntelligenceClient(self.docai_endpoint, AzureKeyCredential(self.docai_api_key))

        self.metadata_fields = {
            "author": "author",
            "theme": ("topic", MetadataIndexFieldType.STRING),
            "director": "director",
        }

        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        Settings.node_parser = SemanticSplitterNodeParser(buffer_size=1, breakpoint_percentile_threshold=95,embed_model=self.embed_model)
        self.llama_debug = LlamaDebugHandler(print_trace_on_end=True)
        self.token_counter = TokenCountingHandler(
            tokenizer=tiktoken.encoding_for_model('gpt-35-turbo').encode
        )
        Settings.callback_manager = CallbackManager([self.token_counter,self.llama_debug])

        self.tmpdirname = tempfile.gettempdir()
       
        logging.info('Temporary directory ' + self.tmpdirname)

        self.index_persist_dir = self.tmpdirname+"/index_cache"

        os.makedirs(self.index_persist_dir, exist_ok=True)

        self.index_persist_dir = self.index_persist_dir+"/" + Path(self.docPath).stem

        os.makedirs(self.index_persist_dir, exist_ok=True)

        self.use_storage = True

        self.all_nodes_dict = None
    

    def LoadIndex(self):
        partialMessage = ""
        try:
            begin = time.time()
            self.token_counter.reset_counts()        
            if os.path.exists(self.index_persist_dir+"/docstore.json"):
                docstore = SimpleDocumentStore(self.index_persist_dir)
                storage_context = StorageContext.from_defaults(docstore=docstore,persist_dir=self.index_persist_dir)
                self.index = load_index_from_storage(storage_context)
                partialMessage += '\n\nIndex loaded from storage:' + self.index_persist_dir
                logging.info(partialMessage)                           
                yield partialMessage
            else:
                partialMessage += '\n\nIndex not found in storage:' + self.index_persist_dir
                logging.info(partialMessage)
                yield partialMessage
        except Exception as e:
            partialMessage += '\n\nError loading index'             
            logging.error(partialMessage)
            yield partialMessage
        finally:
            end = time.time()
            partialMessage += '\n\nIndex loading took ' + str(end - begin) + ' seconds'
            logging.info(partialMessage)
            self.TokenCount()
            yield partialMessage
    
    def TokenCount(self):
        print(
            "Embedding Tokens: ",
            self.token_counter.total_embedding_token_count,
            "\n",
            "LLM Prompt Tokens: ",
            self.token_counter.prompt_llm_token_count,
            "\n",
            "LLM Completion Tokens: ",
            self.token_counter.completion_llm_token_count,
            "\n",
            "Total LLM Token Count: ",
            self.token_counter.total_llm_token_count,
            "\n",
            )

    def GenerateOrLoadIndex(self, use_storage=True):
        zipFile = None
        partialMessage = ""
        try:            
            begin = time.time()
            self.token_counter.reset_counts()            
            self.use_storage = use_storage
            if os.path.exists(self.index_persist_dir+"/docstore.json") and self.use_storage:
                docstore = SimpleDocumentStore(self.index_persist_dir)
                storage_context = StorageContext.from_defaults(docstore=docstore,persist_dir=self.index_persist_dir)
                self.index = load_index_from_storage(storage_context)
                partialMessage += '\n\nIndex loaded from storage:' + self.index_persist_dir
                logging.info(partialMessage)

                zipFile = shutil.make_archive(tempfile.gettempdir()+"/"+Path(self.index_persist_dir).stem, 'zip', self.index_persist_dir)

                partialMessage += '\n\nIndex saved to ' + zipFile
                logging.info(partialMessage)
           
                yield [partialMessage, zipFile]
            else:
                layoutJson = self.index_persist_dir + "/"+Path(self.docPath).stem+".json"
                # Load the document and analyze the layout from online service
                if not os.path.exists(layoutJson):
                    with open(self.docPath, 'rb') as file:
                        file_content = file.read()
                    partialMessage += '\n\nAnalyzing document layout from online service'
                    logging.info(partialMessage)
                    yield [partialMessage,"Pending"]

                    layoutDocs = self.docClient.begin_analyze_document(
                        "prebuilt-layout",
                        analyze_request=AnalyzeDocumentRequest(bytes_source=file_content),
                        output_content_format="markdown"
                    )        
                    docs_string = layoutDocs.result().content
                    with open(layoutJson, 'w') as json_file:
                        json.dump(layoutDocs.result().content, json_file)
                    partialMessage += '\n\nDocument layout saved to ' + layoutJson
                    logging.info(partialMessage)
                    yield [partialMessage,"Pending"]
                # Load the document and analyze the layout from local file
                else:
                    with open(layoutJson) as json_file:
                        docs_string = json.load(json_file)  
                    partialMessage += '\n\nDocument layout loaded from ' + layoutJson
                    logging.info(partialMessage)
                    yield [partialMessage,"Pending"]
                        
                docs = [Document(text=docs_string)]
                base_nodes = Settings.node_parser.get_nodes_from_documents(docs)

                partialMessage += '\n\nGenerating base nodes: '+str(len(base_nodes))
                logging.info(partialMessage)
                yield [partialMessage,"Pending"]

                # set node ids to be a constant
                for idx, node in enumerate(base_nodes):
                    node.id_ = f"node-{idx}"

                sub_chunk_sizes = [128, 256, 512]
                sub_node_parsers = [
                    SentenceSplitter(chunk_size=c, chunk_overlap=20) for c in sub_chunk_sizes
                ]

                partialMessage += '\n\nGenerating subnodes'
                logging.info(partialMessage)
                yield [partialMessage,"Pending"]

                all_nodes = []
                for base_node in base_nodes:
                    for n in sub_node_parsers:
                        sub_nodes = n.get_nodes_from_documents([base_node])
                        sub_inodes = [
                            IndexNode.from_text_node(sn, base_node.node_id) for sn in sub_nodes
                        ]
                        all_nodes.extend(sub_inodes)

                    # also add original node to node
                    original_node = IndexNode.from_text_node(base_node, base_node.node_id)
                    all_nodes.append(original_node)

                all_nodes_dict = {n.node_id: n for n in all_nodes}
                
                # Persist all_nodes_dict using Pickle
                pickle_file_path = self.index_persist_dir + "/all_nodes_dict.pickle"
                with open(pickle_file_path, 'wb') as pickle_file:
                    pickle.dump(all_nodes_dict, pickle_file)

                partialMessage += '\n\nall_nodes_dict saved to ' + pickle_file_path
                logging.info(partialMessage)
                

                partialMessage += '\n\nGenerating index'
                index = VectorStoreIndex(all_nodes, embed_model=self.embed_model)
                index.storage_context.persist(persist_dir=self.index_persist_dir) 
                partialMessage += '\n\nIndex saved to the folder ' + self.index_persist_dir
                logging.info(partialMessage)

                
                zipFile = shutil.make_archive(tempfile.gettempdir()+"/"+Path(self.index_persist_dir).stem, 'zip', self.index_persist_dir)
                partialMessage += '\n\nIndex compressed as ' + zipFile
                logging.info(partialMessage)
    
                yield [partialMessage, zipFile]

        except Exception as e:
            partialMessage += '\n\nError generating index'
            logging.error(partialMessage)
            print(e)
            yield [partialMessage,"None"]
        finally:
            end = time.time()
            partialMessage += '\n\nIndexing took ' + str(end - begin) + ' seconds'
            logging.info(partialMessage)
            self.TokenCount() 

            yield [partialMessage,zipFile]


    def TestSummary(self):
        self.query_engine = self.index.as_query_engine(self.llm,streaming=True)
        response = self.query_engine.query("Give a summary of the document")
        partialMessage = ""
        for text in response.response_gen:
            partialMessage += text
            print(partialMessage+"\n")
    
    def RecursiveRetriverSearch(self,query):
        partialMessage = ""
        try:
            begin = time.time()
            self.token_counter.reset_counts()

            vector_retriever_chunk = self.index.as_retriever(similarity_top_k=2)

            if self.all_nodes_dict is None:
                pickle_file_path = self.index_persist_dir + "/all_nodes_dict.pickle"
                with open(pickle_file_path, 'rb') as pickle_file:
                    self.all_nodes_dict = pickle.load(pickle_file)

            retriever_chunk = RecursiveRetriever(
            "vector",
            retriever_dict={"vector": vector_retriever_chunk},
            node_dict=self.all_nodes_dict,
            verbose=True            
            )

            nodes = retriever_chunk.retrieve(
                "Help to summarize the document content"
            )
            for node in nodes:
                logging.info(node.node_id+':\n\n')
                logging.info(node.text)

            '''
            self.query_engine = RetrieverQueryEngine.from_args(retriever_chunk, llm=self.llm, streaming=True)

            response_stream = self.query_engine.query(query)
            
            for text in response_stream.response_gen:
                partialMessage += text
                logging.info(partialMessage+"\n")
                yield partialMessage
            '''

        except Exception as e:
            logging.error('Error searching index')
            partialMessage  += '\n\n'+str(e)
            logging.info(e)            
        finally:
            end = time.time()
            status = '\n\nResponse took ' + str(end - begin) + ' seconds'
            logging.info(status)
            self.TokenCount()
            yield status
        

def TestGenerateIndex():
        testPath = ".\\rules\\rules_original.pdf"
        recursiveRetrieverIndexGenerator = RecursiveRetriverIndexGenerator(testPath, "","")
        logging.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        result= recursiveRetrieverIndexGenerator.GenerateOrLoadIndex(use_storage=False)
        for text,zip in result:
            pass
        logging.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        result = recursiveRetrieverIndexGenerator.LoadIndex()
        for text in result:
            pass

def TestIndex():
        testPath = ".\\rules\\rules_original.pdf"
        recursiveRetrieverIndexGenerator = RecursiveRetriverIndexGenerator(testPath, "","")
        logging.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        result = recursiveRetrieverIndexGenerator.LoadIndex()
        for text in result:
            pass
        logging.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        result = recursiveRetrieverIndexGenerator.RecursiveRetriverSearch("Give a summary of the document")
        for text in result:
            pass
    

if __name__ == "__main__":
     #TestGenerateIndex()
     TestIndex()
      
        

