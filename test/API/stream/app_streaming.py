from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from rag_streaming import rag_graph

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
    
from typing import AsyncGenerator
from typing import AsyncIterable
import httpx

async def graph_stream(inputs: dict) -> AsyncIterable[str]:
        async for response in app.astream_events(inputs, version="v2"):
            with open("./stream_test.txt","a",encoding="utf-8")as e:
                e.write(f"\n\n\n====================\n\n{response}\n\n====================")
            if response["event"] == "on_chain_stream":
                if "generate_conv" in list(response["data"]["chunk"].keys()):
                    try:
                        yield response["data"]["chunk"]["generate_conv"]["generation"]
                    except Exception as e: 
                        try:
                            yield response["data"]["chunk"]["generation"]
                        except Exception as e: pass


@fast_api_app.post("/chat_stream")
async def api(user_input: UserInput):
    user_input=user_input.user_input
    inputs = {"question": user_input}
    streaming=graph_stream(inputs)
    return StreamingResponse(streaming, media_type="text/event-stream",headers={"X-Accel-Buffering": "no"})
    # return StreamingResponse(graph_stream(inputs), media_type="application/x-ndjson")


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
    uvicorn.run(fast_api_app, host="0.0.0.0", port=7888)
