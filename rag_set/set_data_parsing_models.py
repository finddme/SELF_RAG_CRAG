from langchain_core.pydantic_v1 import BaseModel, Field
from typing_extensions import TypedDict
from typing import Literal
from typing import List

# Data model
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

class GradeHallucinations(BaseModel):
    binary_score: str = Field(description="Don't consider calling external APIs for additional information. Answer is supported by the facts, 'yes' or 'no'.")

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["vectorstore", "websearch", "casualconv"] = Field(
        ...,
        description="Given a user question choose to route it to casual conversation or web search or a vectorstore.",)

class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""
    binary_score: str = Field(description="Answer addresses the question, 'yes' or 'no'")

class GraphState(TypedDict):
    question : str
    generation : str
    web_search : str
    documents : List[str]
    source: List[dict]