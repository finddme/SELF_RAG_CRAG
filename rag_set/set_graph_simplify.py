from langgraph.graph import END, StateGraph

from graders.doc_grader import grade_documents
from agents.generate_agt import generate,generate_conv
from graders.hallucination_grader import grade_hallucination
from tools.query_router import route_question
from tools.reranker import reranker_cohere
from graders.response_grader import grade_response_adequacy
from agents.retrieve_agt import retrieve
from agents.websearch_agt import web_search_tavily
from db.db_management import *
from db.data_processing import crawling_and_processing
from rag_set.set_models import load_models
from rag_set.set_vectordb import set_db_client
from rag_set.set_data_parsing_models import GraphState
from db.config import weaviate_class

(llm, question_router, 
retrieval_grader_relevance, 
rag_chain, hallucination_grader, 
answer_grader, tavily, co)=load_models()

client=set_db_client()
del_weaviate_class(client, weaviate_class)
chunks=crawling_and_processing()
create_weaviate_class(client, weaviate_class)
save_weaviate(client, weaviate_class, chunks)

def grade_documents_node(state):
    return grade_documents(state,retrieval_grader_relevance)

def generate_node(state):
    return generate(state,rag_chain)

def generate_conv_node(state):
    return generate_conv(state,llm)

def grade_hallucination_node(state):
    return grade_hallucination(state,hallucination_grader)

def route_question_node(state):
    return route_question(state,question_router)

def reranker_cohere_node(state):
    return reranker_cohere(state,co,weaviate_class)

def grade_response_adequacy_node(state):
    return grade_response_adequacy(state,answer_grader)

def retrieve_node(state):
    return retrieve(state,client,weaviate_class)

def web_search_tavily_node(state):
    return web_search_tavily(state,tavily)

def decide_to_generate(state):
    print("---ASSESS GRADED DOCUMENTS---")
    question = state["question"]
    web_search = state["web_search"]
    filtered_documents = state["documents"]
    source = state["source"]

    if web_search == "Yes":
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---")
        return "websearch"
    else:
        print("---DECISION: GENERATE---")
        return "generate"

def rag_graph():
    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("web_search_tavily", web_search_tavily_node) # web search # key: action to do
    workflow.add_node("retrieve", retrieve_node) # retrieve
    workflow.add_node("generate_conv", generate_conv_node)
    # workflow.add_node("grade_documents", grade_documents) # grade documents
    workflow.add_node("reranker_cohere", reranker_cohere_node)
    workflow.add_node("generate", generate_node) # generatae

    workflow.add_edge("web_search_tavily", "generate") #start -> end of node
    # workflow.add_edge("retrieve", "generate")
    workflow.add_edge("retrieve", "reranker_cohere")
    workflow.add_edge("reranker_cohere", "generate")
    workflow.add_edge("generate_conv", END)

    # Build graph
    workflow.set_conditional_entry_point(
        route_question_node,
        {
            "websearch": "web_search_tavily",
            "vectorstore": "retrieve",
            "casualconv":"generate_conv",
        },
    )
    
    workflow.add_conditional_edges(
        "generate", # start: node
        grade_hallucination_node, # defined function
        {
            "not supported": "generate", #returns of the function
            "useful": END,               #returns of the function
            "not useful": "web_search_tavily",   #returns of the function
        },
    )

    # Compile
    app = workflow.compile()

    return app
