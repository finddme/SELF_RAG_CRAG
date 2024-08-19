import weaviate
from rag_set.set_models import get_embedding

def save_weaviate(client, weaviate_class, chunks):
    client.batch.configure(batch_size=100)
    with client.batch as batch:
        for i, chunk in enumerate(chunks):
            vector = get_embedding(chunk["text"])
            batch.add_data_object(data_object=chunk, class_name=weaviate_class, vector=vector)

def create_weaviate_class(client, weaviate_class):
    print("--- Create weaviate class ---")
    class_obj = {
        "class": weaviate_class,
        "vectorizer": "none",
    }
    client.schema.create_class(class_obj)

def del_weaviate_class(client, weaviate_class):
    client.schema.delete_class(weaviate_class)