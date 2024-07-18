from gradio_client import Client

AdvRAGSvcURI = "http://127.0.0.1:8000"

def ProofReadTest():
	systemMessage = "Criticize the proofread content, especially for wrong words. Only use 当社の用字・用語の基準,  送り仮名の付け方, 現代仮名遣い,  接続詞の使い方 ，外来語の書き方，公正競争規約により使用を禁止されている語  製品の取扱説明書等において使用することはできない, 常用漢字表に記載されていない読み方, and 誤字 proofread rules, don't use other rules those are not in the retrieved documents.                Pay attention to some known issues:もっとも, または->又は, 「ただし」という接続詞は原則として仮名で表記するため,「又は」という接続詞は原則として漢字で表記するため。また、「又は」は、最後の語句に“など”、「等(とう)」又は「その他」を付けてはならない, 優位性を意味する語.               Firstly show 原文, use bold text to point out every incorrect issue, and then give 校正理由, respond in Japanese. Finally give 修正後の文章, use bold text for modified text. If everything is correct, tell no issues, and don't provide 校正理由 or 修正後の文章."
	content = "今回は半導体製造装置セクターの最近の動きを分析します。このセクターが成長性のあるセクターであるという意見は変えません。また、後工程（テスタ、ダイサなど）は2023年4-6月期、前工程（ウェハプロセス装置）は7-9月期または 10-12月期等 で大底を打ち、その後は回復、再成長に向かうと思われます。但し 、足元ではいくつか問題も出ています。"
	client = Client(f"{AdvRAGSvcURI}/proofread/")
	result = client.predict(
			"rules",	# str in 'Knowledge Graph of Proofreading Rules (rules, train, compose)' Textbox component
			systemMessage,	# str in 'Preset Query Prompt' Textbox component
			content,	# str in 'Content to be Proofread' Textbox component
			api_name="/predict"
	)
	print("result")

def ChatBotTest():
	client = Client(f"{AdvRAGSvcURI}/advchatbot/")
	result = client.predict(
			"Help to summarize the content",	# str in 'Message' Textbox component
			"Azure AI Search",	# Literal['Azure AI Search', 'Knowledge Graph', 'Recursive Retriever', 'Summary Index'] in 'Index Type' Radio component
			"azuresearch_0",	# str in 'Search Index Name, can be index folders or Azure AI Search Index Name' Textbox component
			"You are a friendly AI Assistant",	# str in 'System Message' Textbox component
			api_name="/chat"
	)
	print(result)

def ProofReadAddinTest():
	content = "今回は半導体製造装置セクターの最近の動きを分析します。このセクターが成長性のあるセクターであるという意見は変えません。また、後工程（テスタ、ダイサなど）は2023年4-6月期、前工程（ウェハプロセス装置）は7-9月期または 10-12月期等 で大底を打ち、その後は回復、再成長に向かうと思われます。但し 、足元ではいくつか問題も出ています。"
	client = Client(f"{AdvRAGSvcURI}/proofreadaddin/")
	result = client.predict(			
			content,	# str in 'Content to be Proofread' Textbox component
			False,
			api_name="/predict"
	)
	print(result)

def PrintAPIInfo():
	client = Client(f"{AdvRAGSvcURI}/advchatbot/")
	api_info= client._get_api_info()
	print("start")
	print(api_info)
	

if __name__ == '__main__':
	PrintAPIInfo()
	#ChatBotTest()
	ProofReadAddinTest()