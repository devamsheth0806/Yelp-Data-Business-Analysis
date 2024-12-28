import pandas as pd
from pyspark.sql import SparkSession
import weaviate
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# initialize spark
spark = SparkSession.builder.appName("Parquet to Weaviate").config("spark.sql.parquet.enableVectorizedReader", "false") \
    .config("spark.driver.memory", "16g") \
    .config("spark.executor.memory", "16g") \
    .config("spark.executor.memoryOverhead", "4g") \
    .config("spark.driver.maxResultSize", "4g") \
    .getOrCreate()

# Load Parquet file
data_path = "processed/sentiment_summary"
df = spark.read.parquet(data_path)

df = df.repartition(100)

sc = spark.sparkContext

tokenizer_broadcast = sc.broadcast(AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2"))
model_broadcast = sc.broadcast(AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2"))
class_name = "BusinessReview"

# Define a function to generate embeddings
def generate_embeddings_and_store(partition):
    # Connect to Weaviate (assumes a running Weaviate instance)
    client = weaviate.Client(
        url="http://10.32.50.3:34511"
    )
    tokenizer = tokenizer_broadcast.value
    model = model_broadcast.value
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Initialize Weaviate batch for this partition
    weaviate_batch = client.batch
    with weaviate_batch as batch:
        for row in partition:
            try:
                # Generate embeddings for concatenated reviews
                reviews = " ".join(row["review_list"])
                business_reviews = row["name"] + ":" + reviews
                tokenized_review = tokenizer(business_reviews, return_tensors="pt", truncation=True, padding=True).to(device)
                embedding = model(**tokenized_review).last_hidden_state.mean(dim=1).squeeze().detach().cpu().numpy()
            

                # Create data object for Weaviate
                data_object = {
                    "business_id": row["business_id"],
                    "business_name": row["name"],
                    "useful_list": row["useful_list"],
                    "stars_list": row["stars_list"],
                    "review_list": row["review_list"],
                    "average_sentiment_score": row["average_sentiment_score"],
                }

                # Add to Weaviate batch
                batch.add_data_object(
                    data_object=data_object,
                    class_name=class_name,
                    vector=embedding.tolist()
                )
            except Exception as e:
                print(f"Error processing row {row['name']}: {e}")
    return []

# Apply mapPartitions to process data and store it in Weaviate
df.rdd.mapPartitions(generate_embeddings_and_store).count()  # Trigger computation

print("Data successfully stored in Weaviate!")