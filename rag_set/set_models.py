from typing import Literal
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser # 출력물을 기본 str 형태로 받는 라이브러리
from langchain.schema import Document
import openai
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
import anthropic
import cohere
from tavily import TavilyClient
from rag_set.set_data_parsing_models import *
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults
from FlagEmbedding import FlagReranker
import os 

TAVILY_API_KEY = ""
Openai_API_KEY = ""
GROQ_API_KEY = ""
coher_API_KEY = ""
claude_api_key=""

# llm = ChatGroq(temperature=0, groq_api_key=GROQ_API_KEY)
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620",api_key=claude_api_key)

def load_models():
    global llm
    question_router=query_route_chain()
    retrieval_grader_relevance=docs_grader_chain()
    rag_chain=generate_chain()
    hallucination_grader=hallucination_grader_chain()
    answer_grader=answer_grader_chain()
    tavily=tavily_engine()
    co=cohere_engine()
    return llm, question_router, retrieval_grader_relevance, rag_chain, hallucination_grader, answer_grader, tavily, co

def load_models_for_all():
    question_router=query_route_chain()
    retrieval_grader_relevance=docs_grader_chain()
    rag_chain=generate_chain()
    answer_grader=answer_grader_chain()
    ddg_search=ddg_engine()
    reranker_model=rerank_local_engine()
    return question_router, retrieval_grader_relevance, rag_chain, answer_grader, ddg_search, reranker_model

def get_embedding(text, engine="text-embedding-ada-002"):
    global Openai_API_KEY
    os.environ["OPENAI_API_KEY"] =  Openai_API_KEY
    openai.api_key =os.getenv("OPENAI_API_KEY") 
    res = openai.Embedding.create(input=text,engine=engine)['data'][0]['embedding']
    # from openai import OpenAI
    # embedding_client = OpenAI(api_key=Openai_API_KEY)
    # res= embedding_client.embeddings.create(input = text, model=engine).data[0].embedding
    return res

def query_route_chain():
    global llm
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
    return question_router

def docs_grader_chain():
    global llm
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

    return retrieval_grader_relevance


def generate_chain():
    global llm
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

    return rag_chain

def hallucination_grader_chain():
    global llm
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
    
    return hallucination_grader


def answer_grader_chain():
    global llm
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

    return answer_grader

def tavily_engine(TAVILY_API_KEY=TAVILY_API_KEY):
    tavily = TavilyClient(api_key=TAVILY_API_KEY)
    return tavily

def ddg_engine():
    ddg_wrapper = DuckDuckGoSearchAPIWrapper(max_results=1)
    ddg_search = DuckDuckGoSearchResults(api_wrapper=ddg_wrapper)
    return ddg_search

def cohere_engine(coher_API_KEY=coher_API_KEY):
    co = cohere.Client(coher_API_KEY)
    return co

def rerank_local_engine():
    reranker_model = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True, device="cpu") 
    return reranker_model