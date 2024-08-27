import weaviate
from db.config import weavidat_url

def set_db_client():
    global weavidat_url
    client = weaviate.Client(
        url=weavidat_url
    )
    return client
