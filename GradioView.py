
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
from GradioFunc import *

def proofread_addin_view(app=None):
     with gr.Blocks(title=f"Proofreading by {modelName}",analytics_enabled=False, css="footer{display:none !important}", js=js,theme=gr.themes.Default(spacing_size="sm", radius_size="none", primary_hue="blue")).queue(default_concurrency_limit=Predict_Concurrency,max_size=Max_Queue_Size) as custom_theme_addin:
                #interface = gr.Interface(fn=proof_read, inputs=["file"],outputs="markdown",css="footer{display:none !important}",allow_flagging="never")
                button = gr.Button("Choose Selected Content",elem_id="ChooseSelectedContent")
                StreamingCheckBox = gr.Checkbox(label="Streaming", value=True)
                textbox_Content = gr.Textbox(lines=10, elem_id="ProofreadContent", label="Content to be Proofread", value=os.environ['Sample_Content'])
                interface = gr.Interface(fn=proof_read_addin, inputs=[textbox_Content,StreamingCheckBox], outputs=["markdown"],allow_flagging="never",analytics_enabled=False)
                interface.queue(default_concurrency_limit=Build_Concurrency,max_size=Max_Queue_Size) 
     if app is not None:
        app = gr.mount_gradio_app(app, custom_theme_addin, path="/proofreadaddin")   

def advchat_bot_view(app=None):     
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
                            with gr.Accordion("Chat Settings", open=False):         
                                textbox_systemMessage = gr.Textbox(default_system_message, label="System Message",visible=True, lines=5)            
                            interface = gr.ChatInterface(fn=chat_bot,
                                            chatbot=chatbot,
                                            additional_inputs=[radtio_ptions, textbox_index, textbox_systemMessage, checkbox_Stream], 
                                            submit_btn="Chat",                             
                                            examples = [["provide summary for the document"],["give me insights of the document"]])  
                            interface.queue(default_concurrency_limit=Build_Concurrency,max_size=Max_Queue_Size)  

            else:
                with gr.Blocks(title=f"Chat with {modelName}",analytics_enabled=False, css="footer{display:none !important}", js=js,theme=gr.themes.Default(spacing_size="sm", radius_size="none", primary_hue="blue")).queue(default_concurrency_limit=int(os.environ['Predict_Concurrency']),max_size=Max_Queue_Size) as custom_theme_ChatBot:
                    gr.Label("Chat Mode is now configured into Each Index Tab") 
            if app is not None:
                app = gr.mount_gradio_app(app, custom_theme_ChatBot, path="/advchatbot")


def adv_proofread_view(app=None):
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

            if app is not None:
                app = gr.mount_gradio_app(app, custom_theme, path="/advproofread")


def proofread_view(app=None):
            with gr.Blocks(title=f"Proofreading by {modelName}",analytics_enabled=False, css="footer{display:none !important}", js=js,theme=gr.themes.Default(spacing_size="sm", radius_size="none", primary_hue="blue")).queue(default_concurrency_limit=Predict_Concurrency,max_size=Max_Queue_Size) as custom_theme_base:
                #interface = gr.Interface(fn=proof_read, inputs=["file"],outputs="markdown",css="footer{display:none !important}",allow_flagging="never")
                texbox_Rules = gr.Textbox(lines=1, label="Knowledge Graph of Proofreading Rules (rules, train, compose)", value="rules")
                textbox_QueryRules = gr.Textbox(lines=10, label="Preset Query Prompt", value=systemMessage.content)
                textbox_Content = gr.Textbox(lines=10, elem_id="ProofreadContent", label="Content to be Proofread", value=os.environ['Sample_Content'])
                downloadproofreadbutton = gr.DownloadButton(label="Download Proofread Result")
                interface = gr.Interface(fn=proof_read, inputs=[texbox_Rules, 
                                                                textbox_QueryRules,
                                                                textbox_Content], outputs=["markdown",downloadproofreadbutton],allow_flagging="never",analytics_enabled=False)
                interface.queue(default_concurrency_limit=Build_Concurrency,max_size=Max_Queue_Size) 
            if app is not None:
                app = gr.mount_gradio_app(app, custom_theme_base, path="/proofread")    

def azuresearch_view(app=None):
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
    if app is not None:
        app = gr.mount_gradio_app(app, custom_theme_AzureSearchV2, path="/buildrunazureindex")


def summaryindex_view(app=None):
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
    if app is not None:
        app = gr.mount_gradio_app(app, custom_theme_summaryIndexV2, path="/buildrunsummaryindex")


def recursive_retriever_view(app=None):
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
      if app is not None:                       
        app = gr.mount_gradio_app(app, custom_theme_rrIndexV2, path="/buildrunrrindex")

def knowledgegraph_view(app=None):
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
    if app is not None:
        app = gr.mount_gradio_app(app, custom_theme_indexV2, path="/buildrunragindex")

def graphrag_view(app=None):
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
    if app is not None:
        app = gr.mount_gradio_app(app, custom_theme_GraphRAGIndexV2, path="/buildrunGraphRAGindex")

file_upload_csv = gr.File(label="Upload CSV File",file_types=["csv"])
textbox_csv_file =  gr.Textbox(lines=1, label="CSV File Path") 
  
def csvqueryengine_view(app=None):
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
    if app is not None:
        app = gr.mount_gradio_app(app, custom_theme_CSVAnalysis, path="/csvqueryengine")

def toggle_darkmode():
    return """
        () => {
            document.body.classList.toggle('dark');
        }
        """
    
def darkmode_button():
    btn = gr.Button("Switch Theme")
    btn.click(None, [], [], js=toggle_darkmode())
