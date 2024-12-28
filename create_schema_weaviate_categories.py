import weaviate

# Connect to Weaviate (assumes a running Weaviate instance)
client = weaviate.Client(
    url="http://10.32.50.3:34511"
)

class_name = "BusinessCategories"


# Define a schema for your data
schema = {
    "class": class_name,
    "vectorIndexType": "hnsw",
    "vectorizer": "none",
    "properties": [
        {"name": "business_id", "dataType": ["string"]},
        {"name": "business_name", "dataType": ["string"]},
        {"name": "categories", "dataType": ["string"]},
    ],
}

client.schema.delete_class("BusinessCategories")

client.schema.create_class(schema)

