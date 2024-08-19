import weaviate

def set_db_client():
    client = weaviate.Client(
        url="http://192.168.2.186:8080"
    )
    return client

