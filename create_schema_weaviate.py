import weaviate

# Connect to Weaviate (assumes a running Weaviate instance)
client = weaviate.Client(
    url="http://10.32.50.3:34511"
)

class_name = "BusinessReview"


# Define a schema for your data
schema = {
    "class": class_name,
    "vectorIndexType": "hnsw",
    "vectorizer":"none",
    "properties": [
        {"name": "business_id", "dataType": ["string"]},
        {"name": "business_name", "dataType"
        : ["string"]},
        {"name": "useful_list", "dataType": ["number[]"]},
        {"name": "stars_list", "dataType": ["number[]"]},
        {"name": "review_list", "dataType": ["string[]"]},
        {"name": "average_sentiment_score", "dataType": ["number"]}
    ],
}

client.schema.delete_class("BusinessReview")

client.schema.create_class(schema)