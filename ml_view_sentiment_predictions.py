from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder \
    .appName("View Sentiment Predictions") \
    .getOrCreate()

# Path to the output file
output_path = "./processed/sentiment_predictions"

# Load the Parquet file
predictions_df = spark.read.parquet(output_path)

# Show the first few rows
print("Sample Data from Sentiment Predictions:")
predictions_df.show(truncate=False)

# Print the schema for column names
print("Schema of Sentiment Predictions:")
predictions_df.printSchema()

# Stop Spark session
spark.stop()
