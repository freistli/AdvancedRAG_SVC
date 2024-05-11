import json
import logging
import os
from pathlib import Path
import sys
import tempfile
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


load_dotenv('.env_4_SC')
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


class AzureAISearchIndexGenerator:
    def __init__(self, docPath, indexName, idPrefix):
        self.docPath = docPath
        self.idPrefix = idPrefix
        self.aoai_api_key = os.environ['AZURE_OPENAI_API_KEY']  
        self.aoai_endpoint = os.environ['AZURE_OPENAI_ENDPOINT']
        self.aoai_api_version = "2023-05-15"
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
        self.search_service_api_version = "2023-11-01"

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

        self.tmpdirname = tempfile.gettempdir()
       
        logging.info('Temporary directory ' + self.tmpdirname)

        self.persist_dir= self.tmpdirname+"/file_layout_cache"

        os.makedirs(self.persist_dir, exist_ok=True)
    

    def LoadIndex(self):
       
        self.vector_store = AzureAISearchVectorStore(
            search_or_index_client=self.searchClient,
            filterable_metadata_field_keys=self.metadata_fields,
            index_management=IndexManagement.VALIDATE_INDEX,
            id_field_key="id",
            chunk_field_key="Text",
            embedding_field_key="Embedding",
            embedding_dimensionality=1536,
            metadata_string_field_key="metadata",
            doc_id_field_key="doc_id",
        )

        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        self.index = VectorStoreIndex.from_documents(
            [],
            storage_context=self.storage_context,
        )
        

    def GenerateIndex(self):
        try:
            self.vector_store = AzureAISearchVectorStore(  
            search_or_index_client=self.index_client,  
            filterable_metadata_field_keys=self.metadata_fields,
            index_name=self.index_name,  
            index_management = IndexManagement.CREATE_IF_NOT_EXISTS,  
            id_field_key="id",  
            chunk_field_key="Text",  
            embedding_field_key="Embedding",  
            metadata_string_field_key="metadata",
            doc_id_field_key="doc_id",
            embedding_dimensionality=1536,
            language_analyzer="en.lucene",
            vector_algorithm_type="exhaustiveKnn"
            )

            self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

            layoutJson = self.persist_dir + "/"+Path(self.docPath).stem+".json"
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
                    logging.info('Layout analysis result saved to ' + layoutJson)

                # Load the document and analyze the layout from local file
            else:
                with open(layoutJson) as json_file:
                    docs_string = json.load(json_file) 
                logging.info('Layout analysis result loaded from ' + layoutJson)

            splitter = MarkdownTextSplitter.from_tiktoken_encoder(chunk_size=300)
            content_list = splitter.split_text(docs_string)  
            
            logging.info('Number of chunks: ' + str(len(content_list)))

            self.docs = []

            for i in range(len(content_list)):
                doc = Document(text=content_list[i],id_=self.idPrefix+"_"+str(i))
                self.docs.append(doc)

            self.index = VectorStoreIndex.from_documents(
            documents=self.docs,
            storage_context=self.storage_context)

            self.index.storage_context.persist()

        except Exception as e:
            print(e)

    def TestSummary(self):
        self.query_engine = self.index.as_query_engine(self.llm,streaming=True)
        response = self.query_engine.query("Give a summary of the document")
        partialMessage = ""
        for text in response.response_gen:
            partialMessage += text
            print(partialMessage+"\n")
    
    def HybridSearch(self,query):
        self.hybrid_retriever = self.index.as_retriever(
            vector_store_query_mode=VectorStoreQueryMode.SEMANTIC_HYBRID
        )
        self.query_engine = RetrieverQueryEngine.from_args(
                #graph_rag_retriever,
                self.hybrid_retriever,
                streaming=True,
                verbose=True
            )

        response_stream = self.query_engine.query(query)
        partialMessage = ""
        for text in response_stream.response_gen:
            partialMessage += text
            print(partialMessage+"\n")
        

    def TestSearch(self):
        testPath1 = 'C:\\Users\\freistli\\Downloads\\Northwind_Standard_Benefits_Details.pdf'
        azureaisearchIndexGenerator = AzureAISearchIndexGenerator(testPath1, "llamaindex_test2", Path(testPath1).stem)
        azureaisearchIndexGenerator.GenerateIndex()
        azureaisearchIndexGenerator.LoadIndex()
        azureaisearchIndexGenerator.HybridSearch("summarize the doc")
