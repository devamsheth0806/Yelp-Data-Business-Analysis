from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, col, expr

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("Sentiment Summary") \
    .config("spark.sql.parquet.enableVectorizedReader", "false") \
    .config("spark.driver.memory", "16g") \
    .config("spark.executor.memory", "16g") \
    .config("spark.executor.memoryOverhead", "4g") \
    .getOrCreate()

# File paths
sentiment_file_path = "./processed/sentiment_analysis"

# Read the processed sentiment Parquet file
df = spark.read.parquet(sentiment_file_path)

# Verify the columns in the DataFrame
print("Columns in the DataFrame:")
for col_name in df.columns:
    print(col_name)

# Perform aggregation to calculate average sentiment score and average stars
result = df.withColumn(
    "average_sentiment_score", 
    (col("positive_score") - col("negative_score")) / col("review_length")  # Calculate sentiment score
)

# selecting required columns
result = result.select('business_id','categories','name','useful_list','stars_list','review_list','average_sentiment_score')

# Optionally, write the result to a new Parquet file
output_path = "./processed/sentiment_summary"
result.write.parquet(output_path, mode="overwrite", compression="snappy")

# Stop the SparkSession
spark.stop()
