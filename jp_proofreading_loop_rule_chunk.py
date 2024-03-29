
import base64
import os
import sys

from dotenv import load_dotenv
from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter,MarkdownTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.callbacks import BaseCallbackHandler
from langchain_experimental.text_splitter import SemanticChunker
from openai import AzureOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from queue import SimpleQueue
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import AzureChatOpenAI
from typing import Any, Union, Dict, List
from langchain.schema import LLMResult
from threading import Thread
from azure.ai.documentintelligence.models import AnalyzeResult, AnalyzeDocumentRequest
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from fastapi import FastAPI

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

file_path=".//rules//rules.pdf"

"""
# Initiate Azure AI Document Intelligence to load the document. You can either specify file_path or url_path to load the document.
loader = AzureAIDocumentIntelligenceLoader(file_path=".//rules//English.docx", api_key=key, api_endpoint=endpoint,
                                           api_model="prebuilt-layout")
docs = loader.load()
"""



docClient = DocumentIntelligenceClient(endpoint, AzureKeyCredential(key))

with open(file_path, 'rb') as file:
        file_content = file.read() 

docs = docClient.begin_analyze_document(
        "prebuilt-layout",
        analyze_request=AnalyzeDocumentRequest(bytes_source=file_content),
        output_content_format="markdown"
    )

"""
# Split the document into chunks base on markdown headers.
headers_to_split_on = [
  ("\n", "Paragraph")
]
#splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
"""
docs_string = docs.result().content

splitter = MarkdownTextSplitter.from_tiktoken_encoder(chunk_size=20000)
rules_content_list = splitter.split_text(docs_string)  # chunk the original content
#splits = text_splitter.split_text(docs_string)
print(rules_content_list)


client = AzureChatOpenAI(
    openai_api_version="2023-12-01-preview",
    azure_deployment=os.environ['AZURE_OPENAI_Deployment'],
    streaming=True,
    callbacks=[StreamingGradioCallbackHandler(q)],
    azure_openai_api_key=os.environ['AZURE_OPENAI_API_KEY']
)


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
    content = "You are an Japanese Proof Read expert, I will give you a set of proofread rules and the content to be proofread. You check the content word by word to give professional insights in three sections using Japanese with highlighted Markdown format: \
              **Proofread**. In this section, point out the accurate content issues WITHIN original sentences one by one directly. If the sentences have no issues, keep the original sentances in the section. \
              MUST right behind of the proofread issue, use a superscript citation index number. \
              If it is a word issue, then use a superscript citation index number right behind the word. If it is a sentence issue, then use a superscript citation index number right behind the sentance. For example, 1 is ¹ , 2 is ².\
              **Corrections**. In this section, list all corrections associated with citation index numbers, also explain which proofread rules are exactly used. \
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

def proof_read (Content,Draft):

    currentProofRead = "**Proofread** Empty \r\n **Corrections** Empty \r\n **Corrected Result** Empty \r\n"
    if str(Content).strip() != "":
        print(Content)

        textSplitter = MarkdownTextSplitter.from_tiktoken_encoder(chunk_size=1024)

        to_be_proofread_content_list = textSplitter.split_text(Content)

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

        print(to_be_proofread_content_list[0])

        for i in range(len(rules_content_list)):

            prompt1 = "Here are the proofread rules:" + rules_content_list[i] + "\r\n" 
            
            prompt2 = "Here is the content to be proofread: "+ "\r\n" + to_be_proofread_content_list[0] 
            
            prompt3 = "\r\nHere is the lastest proofread result: "+ currentProofRead +"\r\n" \
                    
            prompt4 = "Please provide detailed corrections in Japanese on the content, need to merge with the latest proofread result in the Proofread and Corrections sections."

            message1 = HumanMessage(
                content=prompt1
            )
            message2 = HumanMessage(
                content=prompt2
            )
            message3 = AIMessage(
                content=prompt3
            )
            message4 = HumanMessage(
                content=prompt4
            )
            response = client.stream([systemMessage, message1,message2,message3,message4])
            partial_message = ""
            for chunk in response:
                partial_message = partial_message + chunk.dict()['content']
                currentProofRead = partial_message
                yield partial_message


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

with gr.Blocks(title="Proofreading by AI", js=js,theme=gr.themes.Default(spacing_size="sm", radius_size="none", primary_hue="blue")) as custom_theme:
    #gr.ChatInterface(stream_predict).queue().launch()
    #interface = gr.Interface(fn=proof_read, inputs=["file"],outputs="markdown",css="footer{display:none !important}",allow_flagging="never")
    interface = gr.Interface(fn=proof_read, inputs=["text","file"], outputs=["markdown"],allow_flagging="never")

app = gr.mount_gradio_app(app, custom_theme, path="/proofread")
#custom_theme.launch()