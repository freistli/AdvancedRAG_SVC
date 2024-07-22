## Advanced RAG Service

### Purpose

Quick MVP/POCm playground to verify which Index type can exactly address the specific (usually related to accuracy) LLM RAG use case.

Http Clients and MS Office Word/Outlook Add-In can easily use proper vector index types to search doc info and generate LLM response from the service.

### Functionalities

Build and perform queries on multiple importnat Index types. Below is the info about index types and how the project implements them:
      
- Azure AI Search : Azure AI Search Python SDK + [Hybrid Semantic Search](https://techcommunity.microsoft.com/t5/ai-azure-ai-services-blog/azure-ai-search-outperforming-vector-search-with-hybrid/ba-p/3929167) + [Sub Question Query](https://docs.llamaindex.ai/en/v0.10.17/api_reference/query/query_engines/sub_question_query_engine.html)

- MS GraghRAG Local : REST APIs Provided by [GraghRAG accelerator](https://github.com/azure-samples/graphrag-accelerator)
  
- MS GraghRAG Global : REST APIs Provided by [GraghRAG accelerator](https://github.com/azure-samples/graphrag-accelerator)

- Knowledge Graph : [LlamaIndex](https://docs.llamaindex.ai/en/stable/examples/index_structs/knowledge_graph/KnowledgeGraphDemo/)

- Recursive Retriever : [LlamaIndex](https://docs.llamaindex.ai/en/stable/examples/retrievers/recursive_retriever_nodes/?h=recursive+retriever)

- Summary Index : [LlamaIndex](https://docs.llamaindex.ai/en/latest/api_reference/indices/summary/)

### Quick Deployment & User Manul

https://github.com/freistli/AdvancedRAG
