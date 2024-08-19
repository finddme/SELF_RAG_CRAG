from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser # 출력물을 기본 str 형태로 받는 라이브러리
from langchain_anthropic import ChatAnthropic
import anthropic
import cohere

def generate(state,rag_chain):
    print("---GENERATE Answer---")
    question = state["question"]
    documents = state["documents"]
    source = state["source"]
    
    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question,"source": source, "generation": generation}

def generate_conv(state, llm):
    
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
