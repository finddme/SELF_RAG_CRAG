# Docker image

docker pull baseten/vllm

# 1. [free operational RAG] : Version including router, 3 grader, reranker, and web search
## Query Router | Relevant Grader | Hallucination Grader | Answer Grader | Reranker | Web Search
### Langchain | LangGraph | GROQ | Wevieate | Sentence-Transformers | Tavily | Chainlit

<center><img width="300" src="./readme_img/3grader.png"></center>

### summary

RAG Framework : Langchain\
Workflow control : LangGraph\
LLM : Mixtral-8x7b\
Inference accelerate : GROQ\
text embedding : sentence-transformers/all-MiniLM-L6-v2\
vector DB : Wevieate\
reranker : BAAI/bge-reranker-v2-m3\
chunk method : RecursiveCharacterTextSplitter\
web search : DuckDuckGoSearch\
Application Interface : Chainlit

### related files
- ./rag_set/set_graph_all.py

# 2. [Paid operational RAG] : Practical version
## Query Router | Hallucination Grader | Reranker | Web Search
### Langchain | LangGraph | Claude3.5 | Wevieate | Cohere | Tavily(advanced) | Fastapi | Streamlit

<center><img width="300" src="./readme_img/blog_rag.png"></center>

### summary
RAG Framework : Langchain\
Workflow control : LangGraph\
LLM : Claude sonnet 3.5\
text embedding : Openai\
vector DB : Wevieate\
chunk method : RecursiveCharacterTextSplitter\
reranker : Cohere\
web search : Tavily\
Application Interface : Fastapi + Streamlit

### related files
- ./rag_set/set_graph_simplify.py
