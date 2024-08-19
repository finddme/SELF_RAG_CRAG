import os
import weaviate
import re
import cohere

coher_API_KEY = ""

def reranker_cohere(state,co,weaviate_class):
    print("--- RERANK ---")
    question = state["question"]
    documents_or = state["documents"]

    documents=[r["text"] for r in documents_or["data"]["Get"][weaviate_class]]
    source=[{"source_title":r["source_title"],"source":r["source"]} for r in documents_or["data"]["Get"][weaviate_class]]
    
    rerank_res = co.rerank(
        model="rerank-multilingual-v3.0",
        query=question,
        documents=documents, 
        top_n=4, 
    )
    
    doc_txt = ""
    
    for idx,result in enumerate(rerank_res.results):
        doc_txt += f"doc {idx}. {documents[result.index]} \n"
        
    return {"documents": doc_txt, "question": question, "source": source}

def reranker_fr(state,reranker_model,weaviate_class):
    print("---Reranking---")
    question = state["question"]
    documents = state["documents"]
    
    # Rerank
    sentence_pairs = [[question, rr["text"]] for rr in documents["data"]["Get"][weaviate_class]]
    similarity_scores = reranker_model.compute_score(sentence_pairs)
    paired_list = list(zip(similarity_scores, documents["data"]["Get"][weaviate_class]))
    paired_list.sort(key=lambda x: x[0],reverse=True)
    sorted_b = [item[1] for item in paired_list]
    documents["data"]["Get"][weaviate_class]=sorted_b
    return {"documents": documents, "question": question, "source": source}

