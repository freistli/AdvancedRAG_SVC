import getpass
import json
import logging
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
import os
import magic
import requests

load_dotenv('.env_4_SC')
logging.basicConfig(stream=sys.stdout, level=os.environ['LOG_LEVEL'])
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


ocp_apim_subscription_key = os.environ['GRAPHRAG_API_KEY']

"""
"Ocp-Apim-Subscription-Key": 
    This is a custom HTTP header used by Azure API Management service (APIM) to 
    authenticate API requests. The value for this key should be set to the subscription 
    key provided by the Azure APIM instance in your GraphRAG resource group.
"""
headers = {"Ocp-Apim-Subscription-Key": ocp_apim_subscription_key}

file_directory = "C:\\Users\\freistli\\OneDrive - Microsoft\\POC\\GraphRAGProofread\\Input"
storage_name = "whatiworkedon"
index_name = "whatiworkedon"
endpoint = os.environ['GRAPHRAG_ENDPOINT']

assert (
    file_directory != "" and storage_name != "" and index_name != "" and endpoint != ""
)

def upload_files(
    file_directory: str,
    storage_name: str,
    batch_size: int = 100,
    overwrite: bool = True,
    max_retries: int = 5,
) -> requests.Response | list[Path]:
    """
    Upload files to a blob storage container.

    Args:
    file_directory - a local directory of .txt files to upload. All files must have utf-8 encoding.
    storage_name - a unique name for the Azure storage blob container.
    batch_size - the number of files to upload in a single batch.
    overwrite - whether or not to overwrite files if they already exist in the storage blob container.
    max_retries - the maximum number of times to retry uploading a batch of files if the API is busy.

    NOTE: Uploading files may sometimes fail if the blob container was recently deleted
    (i.e. a few seconds before. The solution "in practice" is to sleep a few seconds and try again.
    """
    url = endpoint + "/data"

    def upload_batch(
        files: list, storage_name: str, overwrite: bool, max_retries: int
    ) -> requests.Response:
        for _ in range(max_retries):
            response = requests.post(
                url=url,
                files=files,
                params={"storage_name": storage_name, "overwrite": overwrite},
                headers=headers,
            )
            # API may be busy, retry
            if response.status_code == 500:
                print("API busy. Sleeping and will try again.")
                time.sleep(10)
                continue
            return response
        return response

    batch_files = []
    accepted_file_types = ["text/plain"]
    filepaths = list(Path(file_directory).iterdir())
    for file in tqdm(filepaths):
        # validate that file is a file, has acceptable file type, has a .txt extension, and has utf-8 encoding
        if (
            not file.is_file()
            or file.suffix != ".txt"
            or magic.from_file(str(file), mime=True) not in accepted_file_types
        ):
            print(f"Skipping invalid file: {file}")
            continue
        # open and decode file as utf-8, ignore bad characters
        batch_files.append(
            ("files", open(file=file, mode="r", encoding="utf-8", errors="ignore"))
        )
        # upload batch of files
        if len(batch_files) == batch_size:
            response = upload_batch(batch_files, storage_name, overwrite, max_retries)
            # if response is not ok, return early
            if not response.ok:
                return response
            batch_files.clear()
    # upload remaining files
    if len(batch_files) > 0:
        response = upload_batch(batch_files, storage_name, overwrite, max_retries)
    return response

def test_upload_files():
    response = upload_files(
        file_directory=file_directory,
        storage_name=storage_name,
        batch_size=100,
        overwrite=True,
    )
    if not response.ok:
        print(response.text)
    else:
        print(response)

def build_index(
    storage_name: str,
    index_name: str,
) -> requests.Response:
    """Create a search index.
    This function kicks off a job that builds a knowledge graph index from files located in a blob storage container.
    """
    url = endpoint + "/index"
    request = {"storage_name": storage_name, "index_name": index_name}
    return requests.post(url, params=request, headers=headers)

def test_build_index():
    response = build_index(storage_name=storage_name, index_name=index_name)
    if response.ok:
        print(response.text)
    else:
        print(f"Failed to submit job.\nStatus: {response.text}")


def index_status(index_name: str) -> requests.Response:
    url = endpoint + f"/index/status/{index_name}"
    return requests.get(url, headers=headers)

def test_index_status():
    response = index_status(index_name)
    if response.ok:
        print(response.json())
    else:
        print(f"Failed to get index status.\nStatus: {response.text}")


# a helper function to parse out the result from a query response
def parse_query_response(
    response: requests.Response, return_context_data: bool = False
) -> requests.Response | dict[list[dict]]:
    """
    Prints response['result'] value and optionally
    returns associated context data.
    """
    if response.ok:
        print(json.loads(response.text)["result"])
        if return_context_data:
            return json.loads(response.text)["context_data"]
        return response
    else:
        print(response.reason)
        print(response.content)
        return response
    
    
def global_search(index_name: str | list[str], query: str) -> requests.Response:
    """Run a global query over the knowledge graph(s) associated with one or more indexes"""
    url = endpoint + "/query/global"
    request = {"index_name": index_name, "query": query}
    return requests.post(url, json=request, headers=headers)


def test_global_search(query):
    global_response = global_search(
    index_name=index_name, query=query)
    global_response_data = parse_query_response(global_response, return_context_data=True)
    global_response_data

if __name__ == "__main__":
    #test_upload_files()
    #test_build_index()
    #test_index_status()
    sysMessage = os.environ['System_Message']
    contentPrefix = os.environ['Content_Prefix']
    #query = "Please proofread this content: \n 今回は半導体製造装置セクターの最近の動きを分析します。このセクターが成長性のあるセクターであるという意見は変えません。また、後工程（テスタ、ダイサなど）は2023年4-6月期、前工程（ウェハプロセス装置）は7-9月期または 10-12月期等 で大底を打ち、その後は回復、再成長に向かうと思われます。但し 、足元ではいくつか問題も出ています。\n"
    query = sysMessage + "\n" + contentPrefix +"\n 温古知新, 質議応答, 接渉"
    test_global_search(query=query)
