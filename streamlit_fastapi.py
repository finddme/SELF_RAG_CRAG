"""
command:
streamlit run streamlit_fastapi.py

http://192.168.2.186:8501/
"""
import asyncio
from asyncio import run_coroutine_threadsafe
from threading import Thread
import os
import sys
import streamlit as st
import requests
import random
import time
from streamlit_chat import message
from langserve import RemoteRunnable
from pprint import pprint
import json
# """
# st.session_state 사용: 
# https://handmadesoftware.medium.com/streamlit-asyncio-and-mongodb-f85f77aea825
# """

# #@st.cache_resource
async def get_response(user_input):
    print("user_input",user_input)
    client="http://192.168.2.186:7808/chat"
    headers = { 'Content-Type': 'application/json' }
    # response = requests.request("POST", client , headers=headers, data=json.dumps(user_input))
    user_input={"user_input": user_input}
    # response = requests.post(client, json.dumps(user_input))
    response = requests.request("POST", client , headers=headers, data=json.dumps(user_input))
    # print(response)
    answer = response.json()['output']
    source = response.json()['source']
    return answer,source

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Hi! How can I help you?"}]

# Inject custom CSS to set the width of the sidebar
st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            width: 50px !important; # Set the width to your desired value
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.button('Clear History', on_click=clear_chat_history)

# ===============================================================================================================
# https://medium.com/gitconnected/langgraph-fastapi-and-streamlit-gradio-the-perfect-trio-for-ai-development-f1a82775496a
async def run_convo():
    # st.title(""":blue[Search / Chat with finddme]""")
    st.title(""":blue[RAG 기반 검색창]""")
    st.markdown("""
    <style>
    .small-font {
        font-size:12px !important;
        color:gray;
        margin-top: 0%;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown(""":orange[*Query Router | Hallucination Grader | Reranker | Web search*<br><p class='small-font'>인공지능/LLM/NLP와 관련된 질문은 <strong>vector DB</strong>를 기반으로 답변하고, 
                일반적인 질문은 <strong>web search</strong>를 통해 답변합니다.
                정보를 요구하는 질문이 아닐 경우 <strong>일상 대화 chatbot 버전</strong>으로 사용할 수 있습니다.</p>]""", 
                unsafe_allow_html=True 
                )
    # st.markdown("<p class='small-font'>정보를 요구하는 질문이 아닐 경우 일상 대화 chatbot 버전으로 사용할 수 있습니다.</p>",unsafe_allow_html=True)
    # st.markdown('---')
    # colored_header(label='', description='', color_name='gray-30')
    user_input = st.text_input('검색어를 입력하세요.')
    # submitted = st.form_submit_button('Send')

    if user_input:
        with st.spinner("Processing..."):
            try:
                # app = RemoteRunnable("http://192.168.2.186:8088/chat")
                # for output in app.stream({"input": input_text}):
                # input_text={"question": input_text}
                answer, source = await get_response(user_input)
                source= [s["source_title"].split("\n")[0] if s["source_title"] != " " else " " for s in source]
                if source[0] !=" ":
                    answer+= "\n\nsources->\n"+"\n".join(source)
                st.write(str(answer))
            
            except Exception as e:
                st.error(f"Error: {e}")
# ===============================================================================================================

if __name__ == '__main__':
    asyncio.run(run_convo())

