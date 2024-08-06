'''
Functionality: 
Generate an index for a document using Azure AI services and Azure Search. The document is first analyzed for layout using Azure Document AI service. The layout is then used to split the document into chunks.
The chunks are then indexed using Azure Search. The index is then used to search the document for a query.
Author: Freist Li
Date: 2024-05
Version: 1.0
'''
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import time
import Common
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

from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core.callbacks import LlamaDebugHandler
from llama_index.core.tools import QueryEngineTool, ToolMetadata
import tiktoken
from llama_index.core.query_engine import SubQuestionQueryEngine
from Environment import *


load_dotenv('.env_4_SC')
logging.basicConfig(stream=sys.stdout, level=os.environ['LOG_LEVEL'])
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


class AzureAISearchIndexGenerator:
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
            deployment_name= Embedding_Mode,
            api_key=self.aoai_api_key,
            azure_endpoint=self.aoai_endpoint,
            api_version=self.aoai_api_version,

        )

        self.search_service_api_key = os.environ['AZURE_SEARCH_API_KEY']
        self.search_service_endpoint = os.environ['AZURE_SEARCH_ENDPOINT']
        self.search_service_api_version = os.environ['AZURE_SEARCH_API_VERSION']
        
        self.useSubQueryEngine = os.environ['USE_SUB_QUERY_ENGINE']

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
        Settings.node_parser = SentenceSplitter(chunk_size=512,chunk_overlap=128)
        self.llama_debug = LlamaDebugHandler(print_trace_on_end=True)
        self.token_counter = TokenCountingHandler(
            tokenizer=tiktoken.encoding_for_model('gpt-35-turbo').encode
        )
        Settings.callback_manager = CallbackManager([self.token_counter,self.llama_debug])

        self.tmpdirname = tempfile.gettempdir()
       
        logging.info('Temporary directory ' + self.tmpdirname)

        self.persist_dir= self.tmpdirname+"/file_layout_cache"

        os.makedirs(self.persist_dir, exist_ok=True)
    

    def LoadIndex(self):
        try:
            begin = time.time()
            self.token_counter.reset_counts()
        
            self.vector_store = AzureAISearchVectorStore(
                search_or_index_client=self.searchClient,
                filterable_metadata_field_keys=self.metadata_fields,
                index_management=IndexManagement.VALIDATE_INDEX,
                id_field_key="id",
                chunk_field_key="chunk",
                embedding_field_key="embedding",
                embedding_dimensionality=1536,
                metadata_string_field_key="metadata",
                doc_id_field_key="doc_id",
            )

            self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

            self.index = VectorStoreIndex.from_documents(
                [],
                storage_context=self.storage_context,
            )
        except Exception as e:             
            logging.error('Error loading index')
            print(e)
        finally:
            end = time.time()
            logging.info('Index loading took ' + str(end - begin) + ' seconds')
            self.TokenCount()

    def ChunkTest(self,doc_id):
        try:
            items = self.searchClient.search(search_text="*",filter=f"doc_id eq '{doc_id}'", top=1,include_total_count=True,select="doc_id")
            if items and items.get_count() > 0:
                logging.info("Chunk exists: "+doc_id)
                for item in items:
                    logging.info(item)            
                logging.info("Items Count: "+str(items.get_count()))
                return True
            else:
                return False
        except Exception as e:
            logging.error('Error checking chunk')
            print(e)
            return False
    
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

    def GenerateIndex(self):
        try:
            begin = time.time()
            self.token_counter.reset_counts()
            partialMessage = ""
            first_doc_id = self.idPrefix + "_0"

            if self.ChunkTest(first_doc_id) == False:
            
                self.vector_store = AzureAISearchVectorStore(  
                search_or_index_client=self.index_client,  
                filterable_metadata_field_keys=self.metadata_fields,
                index_name=self.index_name,  
                index_management = IndexManagement.CREATE_IF_NOT_EXISTS,  
                id_field_key="id",  
                chunk_field_key="chunk",  
                embedding_field_key="embedding",  
                metadata_string_field_key="metadata",
                doc_id_field_key="doc_id",
                embedding_dimensionality=1536,
                language_analyzer="en.lucene",
                vector_algorithm_type="exhaustiveKnn"
                )

                self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

                layoutJson = self.persist_dir + "/"+Path(self.docPath).stem+".json"

                isText = False
                common = Common.Common(self.docPath)
                isText = common.isTextFile()

                if isText is not True:
                    # Load the document and analyze the layout from online service
                    if not os.path.exists(layoutJson):
                        with open(self.docPath, 'rb') as file:
                            file_content = file.read()

                            layoutDocs = self.docClient.begin_analyze_document(
                                "prebuilt-layout",
                                analyze_request=AnalyzeDocumentRequest(bytes_source=file_content),
                                output_content_format="markdown"
                            )        
                            docs_string = layoutDocs.result().content
                            with open(layoutJson, 'w') as json_file:
                                json.dump(layoutDocs.result().content, json_file)
                            partialMessage += 'Layout analysis result saved to ' + layoutJson + "\n"
                            logging.info(partialMessage)
                            yield partialMessage

                        # Load the document and analyze the layout from local file
                    else:
                        with open(layoutJson) as json_file:
                            docs_string = json.load(json_file) 
                        partialMessage += 'Layout analysis result loaded from ' + layoutJson + "\n"
                        logging.info(partialMessage)
                        yield partialMessage
                else:
                    with open(self.docPath, 'r') as file:
                        docs_string = file.read()
                    partialMessage += 'Text file loaded from ' + self.docPath + "\n"
                    logging.info(partialMessage)
                    yield partialMessage

                splitter = MarkdownTextSplitter.from_tiktoken_encoder(chunk_size=512)
                content_list = splitter.split_text(docs_string)  
                
                partialMessage += '\n\nNumber of chunks: ' + str(len(content_list)) + "\n"
                logging.info(partialMessage)
                yield partialMessage

                self.docs = []

                for i in range(len(content_list)):
                    doc = Document(text=content_list[i],id_=self.idPrefix+"_"+str(i))
                    self.docs.append(doc)

                self.index = VectorStoreIndex.from_documents(
                documents=self.docs,
                storage_context=self.storage_context)

                self.index.storage_context.persist()
            else:
                partialMessage += f'The Document {first_doc_id} already exists in the index, skipping indexing'
                logging.info(partialMessage)
                yield partialMessage                       

        except Exception as e:
            partialMessage += 'Error generating index'
            logging.error(partialMessage)
            print(e)
            yield partialMessage
        finally:
            end = time.time()
            partialMessage += '\n\nIndexing took ' + str(end - begin) + ' seconds'
            logging.info(partialMessage)
            self.TokenCount() 
            yield partialMessage


    def TestSummary(self):
        self.query_engine = self.index.as_query_engine(self.llm,streaming=True)
        response = self.query_engine.query("Give a summary of the document")
        partialMessage = ""
        for text in response.response_gen:
            partialMessage += text
            print(partialMessage+"\n")
    
    def HybridSearch(self,query):
        partialMessage = ""
        try:
            begin = time.time()
            self.token_counter.reset_counts()

            '''
            self.hybrid_retriever = self.index.as_retriever(
                vector_store_query_mode=VectorStoreQueryMode.HYBRID,
                similarity_top_k=5                
            )
            self.query_engine = RetrieverQueryEngine.from_args(
                    self.hybrid_retriever,
                    streaming=True,
                    verbose=True
                )
            '''

            if self.useSubQueryEngine == "False":
                self.query_engine = self.index.as_query_engine(self.llm,
                                                           streaming=True,
                                                           verbose=True, 
                                                           vector_store_query_mode= VectorStoreQueryMode.SEMANTIC_HYBRID, 
                                                           similarity_top_k=5)
                response_stream = self.query_engine.query(query)
            
                for text in response_stream.response_gen:
                    partialMessage += text
                    print(partialMessage+"\n")          
                    yield partialMessage
                
            elif self.useSubQueryEngine == "True":
                # setup base query engine as tool
                query_engine_tools = [
                    QueryEngineTool(
                        query_engine=self.index.as_query_engine(llm=self.llm, vector_store_query_mode= VectorStoreQueryMode.SEMANTIC_HYBRID,streaming=True),
                        metadata=ToolMetadata(
                            name=os.getenv('QueryEngineTool_Name'),
                            description=os.getenv('QueryEngineTool_Description') ,
                        ),
                    ),
                ]

                self.sub_query_engine = SubQuestionQueryEngine.from_defaults(
                                query_engine_tools=query_engine_tools,
                                llm=self.llm,
                                use_async=False)
                
                
                response_stream = self.sub_query_engine.query(query)
                
                partialMessage += response_stream.response

                yield partialMessage 
                
                
                         

        except Exception as e:
            logging.error('Error searching index')
            partialMessage  += '\n\n'+str(e)
            print(e)            
        finally:
            end = time.time()
            status = '\n\nResponse took ' + str(end - begin) + ' seconds'
            logging.info(status)
            self.TokenCount()
            yield partialMessage
        

def TestSearch():
        testPath1 = 'rules_original.pdf'        
        azureaisearchIndexGenerator = AzureAISearchIndexGenerator(testPath1, "llamaindex_0", Path(testPath1).stem)
        for result in azureaisearchIndexGenerator.GenerateIndex():
            print(result)
        azureaisearchIndexGenerator.LoadIndex()
        azureaisearchIndexGenerator.HybridSearch("Summarize the document")

if __name__ == "__main__":
     TestSearch()
        

