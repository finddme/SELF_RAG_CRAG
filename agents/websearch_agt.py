from tavily import TavilyClient
from langchain.schema import Document

def web_search_tavily(state,tavily):
    print("---WEB SEARCH ---")
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

def web_search_ddg(state,ddg_search):
    print("---WEB SEARCH ---")
    question = state["question"]
    documents = state["documents"]

    # Web search
    search_res=ddg_search.run(question)
    search_res1=[]
    for s in search_res.split(", title:"):
        search_res1.append(s.replace("[snippet: ","").replace("]",""))
    web_results="\n".join(search_res1)
    web_results = Document(page_content=web_results)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    source = [{"source_title":" ","source":"web search result"}]
    return {"documents": documents, "question": question, "source": source}