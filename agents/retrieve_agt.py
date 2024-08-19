import weaviate
import openai
import os
from rag_set.set_models import get_embedding
# from langchain.embeddings.openai import OpenAIEmbeddings

Openai_API_KEY = ""

def retrieve(state,client,weaviate_class):
    print("---RETRIEVE from Vector Store DB---")
    question = state["question"]
    property_list = list(map(lambda x: x["name"], client.schema.get(weaviate_class)['properties']))
    query_vector = get_embedding(question)
    documents = client.query.get(weaviate_class, property_list).with_hybrid(question, vector=query_vector).with_limit(6).do()
    web_search = "No"
    return {"documents": documents, "question": question}
