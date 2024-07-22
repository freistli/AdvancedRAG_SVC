from datetime import datetime
import getpass
import json
import logging
import sys
import time
from pathlib import Path
from zipfile import ZipFile
from dotenv import load_dotenv
from tqdm import tqdm
import os
import magic
import requests

load_dotenv('.env_4_SC')
logging.basicConfig(stream=sys.stdout, level=os.environ['LOG_LEVEL'])
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

class GraphRagIndexGenerator:
    def __init__(self, file_directory: str, storage_name: str, index_name: str):
        self.file_directory = file_directory
        self.storage_name = storage_name
        self.index_name = index_name
        self.endpoint = os.environ['GRAPHRAG_ENDPOINT']
        self.ocp_apim_subscription_key = os.environ['GRAPHRAG_API_KEY']
        self.headers = {"Ocp-Apim-Subscription-Key": self.ocp_apim_subscription_key}


    def upload_files(
            self,
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
        url = self.endpoint + "/data"

        def upload_batch( files: list, storage_name: str, overwrite: bool, max_retries: int
        ) -> requests.Response:
            for _ in range(max_retries):
                response = requests.post(
                    url=url,
                    files=files,
                    params={"storage_name": storage_name, "overwrite": overwrite},
                    headers=self.headers,
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

    def test_upload_files(self):
        response = self.upload_files(
            file_directory=self.file_directory,
            storage_name=self.storage_name,
            batch_size=100,
            overwrite=True,
        )
        if not response.ok:
            print(response.text)
        else:
            print(response)

    def build_index_default(
            self,
        storage_name: str,
        index_name: str,
    ) -> requests.Response:
        """Create a search index.
        This function kicks off a job that builds a knowledge graph index from files located in a blob storage container.
        """
        url = self.endpoint + "/index"
        request = {"storage_name": storage_name, "index_name": index_name}
        return requests.post(url, params=request, headers=self.headers)

    def test_build_index_default(self):
        response = self.build_index_default(storage_name=self.storage_name, index_name=self.index_name)
        if response.ok:
            print(response.text)
        else:
            print(f"Failed to submit job.\nStatus: {response.text}")
    
    def test_build_index_custom(self):
        self.get_prompts_tempaltes()
        response = self.build_index_custom(storage_name=self.storage_name, 
                                           index_name=self.index_name,
                                           entity_extraction_prompt_filepath=self.entity_prompt,
                                           community_prompt_filepath=self.community_prompt,
                                           summarize_description_prompt_filepath=self.summarize_prompt)
        if response.ok:
            print(response.text)
        else:
            print(f"Failed to submit job.\nStatus: {response.text}")
    
    def build_index_custom(
            self,
    storage_name: str,
    index_name: str,
    entity_extraction_prompt_filepath: str = None,
    community_prompt_filepath: str = None,
    summarize_description_prompt_filepath: str = None,
) -> requests.Response:
        """Create a search index.
        This function kicks off a job that builds a knowledge graph (KG) index from files located in a blob storage container.
        """
        url = self.endpoint + "/index"
        prompt_files = dict()
        if entity_extraction_prompt_filepath:
            prompt_files["entity_extraction_prompt"] = open(
                entity_extraction_prompt_filepath, "r"
            )
        if community_prompt_filepath:
            prompt_files["community_report_prompt"] = open(community_prompt_filepath, "r")
        if summarize_description_prompt_filepath:
            prompt_files["summarize_descriptions_prompt"] = open(
                summarize_description_prompt_filepath, "r"
            )
        return requests.post(
            url,
            files=prompt_files if len(prompt_files) > 0 else None,
            params={"index_name": index_name, "storage_name": storage_name},
            headers=self.headers,
        )


    def index_status(self,index_name: str) -> requests.Response:
        url = self.endpoint + f"/index/status/{index_name}"
        return requests.get(url, headers=self.headers)

    def test_index_status(self) -> bool:
        response = self.index_status(self.index_name)
        if response.ok:
            status_data = response.json()
            #logging.info(status_data)
            percent_complete = status_data.get('percent_complete', 0)
            progress = status_data.get('progress', 0)
            if percent_complete >= 100:
                print("Indexing is 100% complete.")
                return True
            else:
                print(f"{datetime.now()}: {percent_complete}% --> {progress}")
                return False
        else:
            print(f"Failed to get index status.\nStatus: {response.text}")
            return True


    # a helper function to parse out the result from a query response
    def parse_query_response(self,
        response: requests.Response, return_context_data: bool = False
    ) -> requests.Response | dict[list[dict]]:
        """
        Prints response['result'] value and optionally
        returns associated context data.
        """
        if response.ok:
            print(json.loads(response.text)["result"])
            if return_context_data:
                return json.loads(response.text)["result"], json.loads(response.text)["context_data"]
            return response
        else:
            print(response.reason)
            print(response.content)
            return response
        
        
    def global_search(self,index_name: str | list[str], query: str) -> requests.Response:
        """Run a global query over the knowledge graph(s) associated with one or more indexes"""
        url = self.endpoint + "/query/global"
        request = {"index_name": index_name, "query": query}
        return requests.post(url, json=request, headers=self.headers)


    def test_global_search(self,query):
        global_response = self.global_search(index_name=self.index_name, query=query)
        result, context = self.parse_query_response(global_response, True)
        print(result)
        return result, context

    def local_search(self, index_name: str | list[str], query: str) -> requests.Response:
        """Run a local query over the knowledge graph(s) associated with one or more indexes"""
        url = self.endpoint + "/query/local"
        request = {"index_name": index_name, "query": query}
        return requests.post(url, json=request, headers=self.headers)
    
    def test_local_search(self,query):
        local_response = self.local_search(index_name=self.index_name, query=query)
        result, context = self.parse_query_response(local_response, True)
        print(result)
        return result, context

    def list_files(self) -> requests.Response:
        """List all data storage containers."""
        url = self.endpoint + "/data"
        response = requests.get(url=url, headers=self.headers)
        print(response.text)
        try:
            indexes = json.loads(response.text)
            return indexes["storage_name"]
        except json.JSONDecodeError:
            print(response.text)
            return response
        
    
    def list_indexes(self) -> list:
        """List all search indexes."""
        url = self.endpoint + "/index"
        response = requests.get(url, headers=self.headers)
        print(response.text)
        try:
            indexes = json.loads(response.text)
            return indexes["index_name"]
        except json.JSONDecodeError:
            print(response.text)
            return response
        
    def test_files(self):
        response = self.list_files()
        print(response)

    def test_indexes(self):
        response = self.list_indexes()
        print(response)

    def generate_prompts(self, storage_name: str, zip_file_name: str, limit: int = 1) -> None:
        """Generate graphrag prompts using data provided in a specific storage container."""
        url = self.endpoint + "/index/config/prompts"
        params = {"storage_name": storage_name, "limit": limit}
        with requests.get(url, params=params, headers=self.headers, stream=True) as r:
            r.raise_for_status()
            with open(zip_file_name, "wb") as f:
                for chunk in r.iter_content():
                    f.write(chunk)

    def get_prompts_tempaltes(self):
        # check if prompt files exist
        entity_extraction_prompt_filepath = "prompts/entity_extraction.txt"
        community_prompt_filepath = "prompts/community_report.txt"
        summarize_description_prompt_filepath = "prompts/summarize_descriptions.txt"
        self.entity_prompt = (
            entity_extraction_prompt_filepath
            if os.path.isfile(entity_extraction_prompt_filepath)
            else None
        )
        self.community_prompt = (
            community_prompt_filepath if os.path.isfile(community_prompt_filepath) else None
        )
        self.summarize_prompt = (
            summarize_description_prompt_filepath
            if os.path.isfile(summarize_description_prompt_filepath)
            else None
        )
        print(f"Entity Prompt: {self.entity_prompt}")
        print(f"Community Prompt: {self.community_prompt}")
        print(f"Summarize Prompt: {self.summarize_prompt}")

    def test_generate_prompts(self):
        zip_file_name = "prompts.zip"
        self.generate_prompts(storage_name=self.storage_name, zip_file_name=zip_file_name)
        print(f"Prompts have been generated and saved to {zip_file_name}")
        with ZipFile("prompts.zip", "r") as zip_ref:
            zip_ref.extractall()

    def delete_files(self,storage_name: str) -> requests.Response:
        """Delete a blob storage container."""
        url = self.endpoint + f"/data/{self.storage_name}"
        return requests.delete(url=url, headers=self.headers)
    
    def delete_index(self,index_name: str) -> requests.Response:
        """Delete a search index."""
        url = self.endpoint + f"/index/{index_name}"
        return requests.delete(url, headers=self.headers)
    
    def test_delete_index(self):
        response = self.delete_index(index_name=self.index_name)
        print(response.text)

if __name__ == "__main__":
    file_directory = "C:\\Users\\freistli\\OneDrive - Microsoft\\POC\\GraphRAGProofread\\Input"
    storage_name = "workedon02"
    index_name = "workedon02"
    #sysMessage = os.environ['System_Message']
    #contentPrefix = os.environ['Content_Prefix']
    #query = "Please proofread this content: \n 今回は半導体製造装置セクターの最近の動きを分析します。このセクターが成長性のあるセクターであるという意見は変えません。また、後工程（テスタ、ダイサなど）は2023年4-6月期、前工程（ウェハプロセス装置）は7-9月期または 10-12月期等 で大底を打ち、その後は回復、再成長に向かうと思われます。但し 、足元ではいくつか問題も出ています。\n"
    #query = sysMessage + "\n" + contentPrefix +"\n 温古知新, 質議応答, 接渉"
    query = "这篇文章给人的最大启发是什么"
    graghRAGEngine = GraphRagIndexGenerator(file_directory, storage_name, index_name)
    #graghRAGEngine.delete_files(storage_name="whatiworkedon")
    #graghRAGEngine.test_upload_files()
    #graghRAGEngine.test_generate_prompts()
    #graghRAGEngine.test_build_index_custom()
    #graghRAGEngine.test_delete_index()
    #while (graghRAGEngine.test_index_status() == False):
    #    time.sleep(5)
    #graghRAGEngine.test_global_search(query=query)
    graghRAGEngine.test_index_status()
    graghRAGEngine.test_local_search(query=query)
    graghRAGEngine.test_files()
    graghRAGEngine.test_indexes()
    
