from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace, split, lower, expr, transform, array_except, filter, trim, lit
from pyspark.sql.types import ArrayType, StringType
import re 
import nltk 
from nltk.corpus import stopwords

spark = SparkSession.builder \
    .appName("Yelp dataset analysis") \
    .config("spark.shuffle.spill", True) \
    .config("spark.driver.memory", "16g") \
    .config("spark.executor.memory", "16g") \
    .config("spark.executor.memoryOverhead", "4096") \
    .config("parquet.block.size", 536870912) \
    .getOrCreate()
stopwords_list = stopwords.words('english')
stopwords_broadcast = spark.sparkContext.broadcast(set(stopwords_list))

# File path
processed_business_file_path = "./processed/business"

# Read the processed Parquet file
business_df = spark.read.parquet(processed_business_file_path)

df_tokenized = business_df.withColumn(
    "tokenized_reviews",
    transform(
        col("review_list"),
        lambda review: split(regexp_replace(lower(review), "[^a-zA-Z0-9\\s]", ""), " ")
    )
)
df_no_stopwords = df_tokenized.withColumn(
    "tokenized_reviews",
    transform(
        col("tokenized_reviews"),
        lambda tokens: array_except(tokens, lit(stopwords_list))
    )
)

df_no_blanks = df_no_stopwords.withColumn(
    "tokenized_reviews",
    transform(
        col("tokenized_reviews"),
        lambda tokens: filter(tokens, lambda word: word != "")
    )
)

df_filtered = df_no_blanks.withColumn(
    "tokenized_reviews",
    transform(
        col("tokenized_reviews"),
        lambda tokens: transform(tokens, lambda word: trim(word))
    )
)

partitioned_df = df_filtered.repartition(50)

business_write_path = 'tokenized/business'

partitioned_df.write.parquet(business_write_path, mode="overwrite", compression="snappy")

# Stop the Spark session
spark.stop()