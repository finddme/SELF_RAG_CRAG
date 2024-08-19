"""
command:
chainlit run rag_chainlit.py -w --port 8000 --host 0.0.0.0 -c
"""
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
import chainlit as cl
from rag import rag_graph

"""
ref.
- https://langchain-ai.github.io/langgraph/
- https://github.com/brucechou1983/chainlit_langgraph/blob/main/app.py
- https://github.com/Chainlit/cookbook
- https://medium.com/@tahreemrasul/building-a-chatbot-application-with-chainlit-and-langchain-3e86da0099a6
- https://docs.chainlit.io/integrations/langchain
"""
app = rag_graph()

@cl.on_message
async def run_convo(message: cl.Message):
    global app
    #"what is the weather in sf"
    inputs = {"question": message.content}
    cb = cl.AsyncLangchainCallbackHandler(stream_final_answer=True)
    config = RunnableConfig(callbacks=[cb])
    
    # res = app.invoke(inputs, config=RunnableConfig(callbacks=[
    #     cl.LangchainCallbackHandler(
    #         # to_ignore=["ChannelRead", "RunnableLambda", "ChannelWrite", "__start__", "_execute"]
    #         # can add more into the to_ignore: "agent:edges", "call_model"
    #         # to_keep=

    #     )]))
    res = await app.ainvoke(inputs,config=config)
    # await cl.Message(content=res["question"][-1].content).send()
    await cl.Message(content=res["generation"]).send()

