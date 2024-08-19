"""
command:
python rag_fastapi.py

http://192.168.2.186:8088/docs#/default/api_chat_post
http://eleven.acryl.ai:37808/docs#/default/api_chat_post
"""
import os
from dotenv import load_dotenv
import weaviate
import json
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from fastapi.responses import StreamingResponse
# from starlette.responses import StreamingResponse
"""
pydantic_v1
- https://python.langchain.com/v0.1/docs/modules/model_io/output_parsers/types/pydantic/
- 특정 schema에 맞게 output을 구성하도록 LLM에 quary할 수 있게하는 parser.
"""
from langchain_core.output_parsers import StrOutputParser # 출력물을 기본 str 형태로 받는 라이브러리
from langchain.schema import Document
from langgraph.graph import END, StateGraph
import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_google_vertexai import VertexAIEmbeddings 
import re
from langchain_anthropic import ChatAnthropic
import anthropic
import cohere
from tavily import TavilyClient
from typing_extensions import TypedDict
from typing import List

TAVILY_API_KEY = ""
Openai_API_KEY = ""
GROQ_API_KEY = ""
coher_API_KEY = ""
claude_api_key=""

client = weaviate.Client(
    url="http://192.168.2.186:8080"
)

# llm = ChatGroq(temperature=0, groq_api_key=GROQ_API_KEY)
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620",api_key=claude_api_key,streaming=True)

tavily = TavilyClient(api_key=TAVILY_API_KEY)

"""
=====================================
prepare embedding model
=====================================
"""
os.environ["OPENAI_API_KEY"] =  Openai_API_KEY
openai.api_key =os.getenv("OPENAI_API_KEY")
embed_model = "text-embedding-ada-002"
embeddings = OpenAIEmbeddings(model=embed_model)

def get_embedding(text, engine="text-embedding-ada-002") : 
    res = openai.Embedding.create(input=text,engine=engine)['data'][0]['embedding']
    # from openai import OpenAI
    # embedding_client = OpenAI(api_key=Openai_API_KEY)
    # res= embedding_client.embeddings.create(input = text, model=engine).data[0].embedding
    return res


"""
=====================================
TypedDict (to apply to workflow functions)
=====================================
"""
class GraphState(TypedDict):
    """
    Represents the stat.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents 
    """
    question : str
    generation : str
    web_search : str
    documents : List[str]
    source: List[dict]

"""
=====================================
Query Router
=====================================
"""

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "websearch", "casualconv"] = Field(
        ...,
        description="Given a user question choose to route it to casual conversation or web search or a vectorstore.",
    )

structured_llm_router = llm.with_structured_output(RouteQuery)

system = """You are an expert at routing a user question to a vectorstore or web search or casual conversation.
The vectorstore contains documents related to language processing, and artificial intelligence.
Use the vectorstore for questions on these topics. For all else, use web-search.
If the question is a casual conversation use original knowledge"""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router

def route_question(state):
    """
    Route question to web search or RAG 

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    question = state["question"]
    source = question_router.invoke({"question": question})   
    if source.datasource == 'websearch':
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "websearch"
    elif source.datasource == 'vectorstore':
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"
    elif source.datasource == 'casualconv':
        print("---ROUTE QUESTION TO CASUAL CONV.---")
        return "casualconv"

"""
=====================================
Retrieval
=====================================
"""
def retrieve(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE from Vector Store DB---")
    question = state["question"]

    # Retrieval
    query_vector = get_embedding(question)
    documents = client.query.get("B_with_title", ["text","source_title","source"]).with_hybrid(question, vector=query_vector).with_limit(6).do()
    web_search = "No"
    return {"documents": documents, "question": question}

"""
=====================================
Reranker
=====================================
"""
def reranker_cohere(state):
    co = cohere.Client(coher_API_KEY)
    
    """
    cohere models
    https://docs.cohere.com/docs/models
    """
    print("--- RERANK ---")
    question = state["question"]
    documents_or = state["documents"]

    documents=[r["text"] for r in documents_or["data"]["Get"]["B_with_title"]]
    source=[{"source_title":r["source_title"],"source":r["source"]} for r in documents_or["data"]["Get"]["B_with_title"]]
    
    rerank_res = co.rerank(
        model="rerank-multilingual-v3.0",
        query=question, # search query used for docs retrieval
        documents=documents, #list of retrieved docs
        top_n=4, # selecting top 4 docs
    )
    
    doc_txt = ""
    
    for idx,result in enumerate(rerank_res.results):
        doc_txt += f"doc {idx}. {documents[result.index]} \n"
        
    return {"documents": doc_txt, "question": question, "source": source}



"""
=====================================
Relevant Grader
=====================================
"""

# Data model
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

# LLM with function call 
structured_llm_grader_docs = llm.with_structured_output(GradeDocuments)

# Prompt 
system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader_relevance = grade_prompt | structured_llm_grader_docs

def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    
    # Score each doc
    filtered_docs = []
    web_search = "No"
    for d in documents["data"]["Get"]["Test"]:
        score = retrieval_grader_relevance.invoke({"question": question, "document": d["text"]})
        grade = score.binary_score
        # Document relevant
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        # Document not relevant
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            # We do not include the document in filtered_docs
            # We set a flag to indicate that we want to run web search
            web_search = "Yes"
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search}

"""
=====================================
Generate
=====================================
"""

prompt = ChatPromptTemplate.from_template(
    """You are a Korean-speaking assistant specializing in question-answering tasks. 
    Use the provided context informations and relevant documents to answer the following question as accurately as possible. 
    If the answer is not clear from the context or if you do not know the answer, explicitly state "모르겠습니다." (I don't know). 
    Use three sentences maximum and keep the answer concise.
    All responses must be given in Korean.
    Based on the given information, return a very detailed response.
Question: {question}
Context: {context}
Answer:"""
)

# Chain
rag_chain = prompt | llm | StrOutputParser()

def generate(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE Answer---")
    question = state["question"]
    documents = state["documents"]
    source = state["source"]
    
    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question,"source": source, "generation": generation}

def generate_conv(state):
    
    print("---GENERATE Answer (conversation)---")
    question = state["question"]
    documents = state["documents"]
    source = state["source"]
    
    prompt=f"""
    You are a friendly and helpful assistant designed to engage in everyday chats with users. Your goal is to make the conversation pleasant, informative, and engaging. 
    Use a casual and approachable tone. Be polite, patient, and show empathy when needed. 
        
    Guidelines for casual conversation:
    - Use informal language and a friendly tone
    - Feel free to use common contractions (e.g., "don't", "can't", "I'm")
    - Incorporate occasional humor or light-hearted comments when appropriate
    - Be empathetic and show interest in the user's thoughts and experiences
    - Avoid overly formal or technical language unless the conversation calls for it
    
    When you receive a message from the user, follow these steps:
    
    1. Read and analyze the user's message.
    
    2. Identify the main topic or intent of the message.
    
    3. Consider an appropriate casual response that addresses the user's input and maintains the flow of conversation.
    
    4. If the user asks a question or seeks information, provide a helpful answer in a casual manner. If you're unsure about something, it's okay to admit that you don't know.
    
    5. If appropriate, ask a follow-up question to keep the conversation going or to learn more about the user's interests or thoughts.
    
    Formulate your response based on the above guidelines and analysis. Your response should be friendly, engaging, and natural-sounding.
    
    Write your response inside <response> tags. Make sure your response is appropriate for a casual conversation and maintains a friendly tone throughout.
    """
    messages = [
        (
            "system",
                prompt,
        ),
        ("human", question),
    ]
    ai_msg = llm.invoke(messages)
    response = ai_msg.content.replace("<response>","").replace("</response>","").strip("\n")
    source = [{"source_title":" ","source":"casual conversation"}]
    return {"documents": documents, "question": question,"source": source, "generation": response}

"""
=====================================
Hallucination Grader
=====================================
"""

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(description="Don't consider calling external APIs for additional information. Answer is supported by the facts, 'yes' or 'no'.")
 
# LLM with function call 
structured_llm_grader_hallucination = llm.with_structured_output(GradeHallucinations)
 
# Prompt 
system = """You are a grader assessing whether an LLM generation is supported by a set of retrieved facts. \n 
     Restrict yourself to give a binary score, either 'yes' or 'no'. If the answer is supported or partially supported by the set of facts, consider it a yes. \n
    Don't consider calling external APIs for additional information as consistent with the facts."""

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
]
)
  
hallucination_grader = hallucination_prompt | structured_llm_grader_hallucination

def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    source = state["source"]

    score = hallucination_grader.invoke({"documents": documents, "generation": generation})
    grade = score.binary_score

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        return "useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"

"""
=====================================
Answer Grader
=====================================
"""

class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(description="Answer addresses the question, 'yes' or 'no'")

# LLM with function call 
structured_llm_grader_answer = llm.with_structured_output(GradeAnswer)

# Prompt 
system = """You are a grader assessing whether an answer addresses / resolves a question \n 
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

answer_grader = answer_prompt | structured_llm_grader_answer



"""
=====================================
Web Search
=====================================
"""
def web_search_tavily(state):
    """
    Web search based based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """

    print("---WEB SEARCH. Append to vector store db---")
    question = state["question"]
    documents = state["documents"]

    # Web search
    tavily_response = tavily.search(query=question,search_depth="advanced")
    tavily_response2 = tavily.qna_search(query=question, search_depth="advanced",max_results =3)
    web_results = "\n".join([d["content"] for d in tavily_response["results"]])
    web_results+=f"\n{tavily_response2}"
    web_results = Document(page_content=web_results)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    source = [{"source_title":" ","source":"web search result"}]
    return {"documents": documents, "question": question, "source": source}

def decide_to_generate(state):
    """
    Determines whether to generate an answer, or add web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    question = state["question"]
    web_search = state["web_search"]
    filtered_documents = state["documents"]
    source = state["source"]

    if web_search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---")
        return "websearch"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"

"""
=====================================
Graph
=====================================
"""

from langgraph.graph import END, StateGraph

def rag_graph():
    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("web_search_tavily", web_search_tavily) # web search # key: action to do
    workflow.add_node("retrieve", retrieve) # retrieve
    workflow.add_node("generate_conv", generate_conv)
    # workflow.add_node("grade_documents", grade_documents) # grade documents
    workflow.add_node("reranker_cohere", reranker_cohere)
    workflow.add_node("generate", generate) # generatae

    workflow.add_edge("web_search_tavily", "generate") #start -> end of node
    # workflow.add_edge("retrieve", "generate")
    workflow.add_edge("retrieve", "reranker_cohere")
    workflow.add_edge("reranker_cohere", "generate")
    workflow.add_edge("generate_conv", END)

    # Build graph
    workflow.set_conditional_entry_point(
        route_question,
        {
            "websearch": "web_search_tavily",
            "vectorstore": "retrieve",
            "casualconv":"generate_conv",
        },
    )
    
    workflow.add_conditional_edges(
        "generate", # start: node
        grade_generation_v_documents_and_question, # defined function
        {
            "not supported": "generate", #returns of the function
            "useful": END,               #returns of the function
            "not useful": "web_search_tavily",   #returns of the function
        },
    )

    # Compile
    app = workflow.compile()
    return app

