from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
import asyncio
# from motor.motor_asyncio import AsyncIOMotorClient
import streamlit as st
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from rag import rag_graph

"""
ref.
- https://generativeai.pub/chatbot-cheat-code-qwen110b-on-streamlit-without-spending-a-penny-part-2-3731a827f27f
- https://handmadesoftware.medium.com/streamlit-asyncio-and-mongodb-f85f77aea825
- 
"""

app = rag_graph()

# FastAPI
fast_api_app = FastAPI(
    title="CHAT"
)

origins = ["*"]

fast_api_app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@fast_api_app.get("/")
def home():
    return {"message": "CHAT"}

class ConnectionManager:
    """Web socket connection manager."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)


conn_mgr = ConnectionManager()

from pydantic import BaseModel as BM
class UserInput(BM):
    user_input: str

@fast_api_app.post("/chat")
async def api(user_input: UserInput):
    global app
    # print("ttttttttttttt",type(user_input.user_input))
    user_input=user_input.user_input
    inputs = {"question": user_input}
    res= await app.ainvoke(inputs)
    result = {
        "output": res["generation"],
        "source": res["source"]
    }
    return result


@fast_api_app.websocket("/chat_ws/{client_id}")
async def api_ws(websocket: WebSocket, client_id: int):
    await conn_mgr.connect(websocket)
    try:
        while True:
            user_input = await websocket.receive_text()
            inputs = {"question": user_input}
            res= await app.ainvoke(inputs)
            result = {
                "output": res["generation"],
                "source": res["source"]
            }
            await conn_mgr.send_message(json.dumps(result), websocket)
    except WebSocketDisconnect:
        conn_mgr.disconnect(websocket)
        await conn_mgr.broadcast(f"Client #{client_id} left the chat")
    return res

if __name__ == '__main__':
    uvicorn.run(fast_api_app, host="0.0.0.0", port=7808)

