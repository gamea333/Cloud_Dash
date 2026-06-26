from retrieval.retriever import KnowledgeRetriever

retriever = KnowledgeRetriever()
results = retriever.retrieve("alerts not firing", top_k=3)
print(results)