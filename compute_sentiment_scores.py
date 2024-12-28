from pyspark.sql import SparkSession
from pyspark.sql.functions import avg

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("Compute Sentiment Scores and Average Stars") \
    .getOrCreate()

# File paths
predictions_path = "./processed/sentiment_predictions"
business_data_path = "./processed/sentiment_summary"

# Load sentiment predictions and business data
predictions_df = spark.read.parquet(predictions_path)
business_data_df = spark.read.parquet(business_data_path)

# Debug: Check schema of business_data_df
print("Schema of business_data_df:")
business_data_df.printSchema()

# Compute average sentiment score and average stars for each business
result_df = predictions_df.groupBy("business_id") \
    .agg(
        avg("average_prediction").alias("average_sentiment_score"),
        avg("average_label").alias("average_sentiment_label")
    )

# Add average stars data (update column name based on actual schema)
result_df = result_df.join(
    business_data_df.select("business_id", "average_stars"),  # Replace "average_stars" with the correct column name
    on="business_id",
    how="left"
).withColumnRenamed("average_stars", "average_star_rating")

# Write the result to Parquet
output_path = "./processed/business_sentiment_analysis"
result_df.write.parquet(output_path, mode="overwrite", compression="snappy")

# Show sample data
result_df.show(20, truncate=False)

# Stop the SparkSession
spark.stop()
