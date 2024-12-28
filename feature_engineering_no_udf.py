from pyspark.sql import SparkSession
from pyspark.sql.functions import col, size, array_contains, flatten, lit, when

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("Feature Engineering with Fine-Tuned Sentiment Analysis") \
    .config("spark.sql.parquet.enableVectorizedReader", "false") \
    .config("spark.driver.memory", "16g") \
    .config("spark.executor.memory", "16g") \
    .config("spark.executor.memoryOverhead", "4g") \
    .getOrCreate()

# File paths (keep the same as the original path)
tokenized_file_path = "./tokenized/business"
output_path = "./processed/sentiment_analysis"  # Use the same output path

# Weighted positive and negative word lists
weighted_positive_words = {
    "awesome": 1.5, "great": 1.2, "good": 1.0, "nice": 0.8, "amazing": 1.5,
    "wonderful": 1.4, "best": 1.8, "comfortable": 1.3, "fantastic": 1.7,
    "satisfied": 1.2, "pleasant": 1.0, "delightful": 1.5, "perfect": 1.6,
    "beautiful": 1.4, "enjoyable": 1.2, "happy": 1.3, "excellent": 1.5,
    "love": 1.8, "outstanding": 1.6, "superb": 1.7, "exceptional": 1.9,
    "marvelous": 1.5, "impressive": 1.4, "brilliant": 1.6, "cool": 1.0,
    "friendly": 1.2, "helpful": 1.3, "polite": 1.2, "cheerful": 1.4,
    "peaceful": 1.3, "modern": 1.1, "affordable": 1.2, "generous": 1.4
}

weighted_negative_words = {
    "terrible": -1.5, "bad": -1.2, "horrible": -1.5, "worst": -1.8,
    "boring": -0.8, "ugly": -1.3, "miserable": -1.4, "uncomfortable": -1.3,
    "disappointing": -1.2, "annoying": -1.1, "frustrating": -1.2,
    "poor": -1.0, "awful": -1.5, "mediocre": -0.9, "dirty": -1.3,
    "rude": -1.2, "cheap": -0.8, "noisy": -1.0, "hateful": -1.5,
    "broken": -1.4, "untrustworthy": -1.6, "fake": -1.5, "depressing": -1.4,
    "stressful": -1.3, "underwhelming": -0.8, "overpriced": -1.2,
    "inconsistent": -1.0, "dangerous": -1.8, "bland": -0.7, "subpar": -0.9
}

# Read the tokenized Parquet file
business_df = spark.read.parquet(tokenized_file_path)

# Flatten the nested arrays into a single array
df_with_flattened = business_df.withColumn("flattened_reviews", flatten(col("tokenized_reviews")))

# Calculate positive and negative weighted counts
positive_count_expr = sum(
    when(array_contains(col("flattened_reviews"), lit(word)), count).otherwise(0)
    for word, count in weighted_positive_words.items()
)
negative_count_expr = sum(
    when(array_contains(col("flattened_reviews"), lit(word)), count).otherwise(0)
    for word, count in weighted_negative_words.items()
)

# Add positive and negative counts as columns
df_with_sentiment = df_with_flattened \
    .withColumn("positive_count", positive_count_expr) \
    .withColumn("negative_count", negative_count_expr)

# Combine counts for overall sentiment
df_with_sentiment = df_with_sentiment \
    .withColumn("sentiment_count", col("positive_count") + col("negative_count"))

# Review Length
df_with_sentiment = df_with_sentiment.withColumn("review_length", size(col("flattened_reviews")))

# Write the processed data back to the same output path
df_with_sentiment.repartition(100).write.parquet(output_path, mode="overwrite", compression="snappy")

# Stop the SparkSession
spark.stop()
