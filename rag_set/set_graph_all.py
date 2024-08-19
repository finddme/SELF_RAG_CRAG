from langgraph.graph import END, StateGraph

from agents.doc_grader import grade_documents
from agents.generate_agt import generate,generate_conv
from agents.hallucination_grader import grade_hallucination
from agents.query_router import route_question
from agents.reranker import reranker_fr
from agents.response_grader import grade_response_adequacy_hallucination
from agents.retrieve_agt import retrieve
from agents.websearch_agt import web_search_tavily, web_search_ddg

from rag_set.set_models import load_models_for_all
from rag_set.set_vectordb import set_db_client
from rag_set.set_data_parsing_models import GraphState

weaviate_class="B_with_title"

(question_router, 
retrieval_grader_relevance, 
rag_chain, answer_grader, 
ddg_search, 
reranker_model)=load_models_for_all()

client=set_db_client()

def grade_documents_node(state):
    return grade_documents(state,retrieval_grader_relevance)

def generate_node(state):
    return generate(state,rag_chain)

def grade_hallucination_node(state):
    return grade_hallucination(state,hallucination_grader)

def route_question_node(state):
    return route_question(state,question_router)

def reranker_fr_node(state):
    return reranker_fr(state,reranker_model,weaviate_class)

def grade_response_adequacy_hallucination_node(state):
    return grade_response_adequacy_hallucination(state,answer_grader)

def retrieve_node(state):
    return retrieve(state,client,weaviate_class)

def web_search_ddg_node(state):
    return web_search_ddg(state,ddg_search)

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
    workflow.add_node("web_search_ddg", web_search_ddg_node) # web search # key: action to do
    workflow.add_node("retrieve", retrieve_node) # retrieve
    workflow.add_node("grade_documents", grade_documents_node) # grade documents
    workflow.add_node("generate", generate_node) # generatae

    workflow.add_node("reranker_fr",reranker_fr_node)
    workflow.add_edge("web_search_ddg", "generate") #start -> end of node
    workflow.add_edge("retrieve", "reranker_fr")
    workflow.add_edge("reranker_fr", "grade_documents")

    # Build graph
    workflow.set_conditional_entry_point(
        route_question_node,
        {
            "websearch": "web_search_ddg",
            "vectorstore": "retrieve",
        },
    )
    
    workflow.add_conditional_edges(
        "grade_documents", # start: node
        decide_to_generate, # defined function
        {
            "websearch": "web_search_ddg", #returns of the function
            "generate": "generate",   #returns of the function
        },
    )

    workflow.add_conditional_edges(
        "generate", # start: node
        grade_response_adequacy_hallucination_node, # defined function
        {
            "not supported": "generate", #returns of the function
            "useful": END,               #returns of the function
            "not useful": "web_search_ddg",   #returns of the function
        },
    )

    # Compile
    app = workflow.compile()

    return app