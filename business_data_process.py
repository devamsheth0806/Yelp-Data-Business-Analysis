from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import collect_list, col, array_sort, struct

spark = SparkSession.builder \
    .appName("Yelp dataset analysis") \
    .config("spark.shuffle.spill", True) \
    .config("spark.driver.memory", "16g") \
    .config("spark.executor.memory", "16g") \
    .config("spark.executor.memoryOverhead", "4096") \
    .config("parquet.block.size", 536870912) \
    .getOrCreate()

# File path
business_file_path = "./yelp_academic_dataset_business.json"
review_file_path = "./yelp_academic_dataset_review.json"

# Read the CSV file
business_df = spark.read.json(business_file_path)
review_df = spark.read.json(review_file_path)

sorted_reviews = review_df.orderBy("business_id", col("useful").desc())
aggregated_reviews = sorted_reviews.groupBy("business_id").agg(
    collect_list("useful").alias("useful_list"),
    collect_list("stars").alias("stars_list"),
    collect_list("text").alias("review_list")
)

# Join the aggregated review data with the business dataset
joined_df = business_df.join(
    aggregated_reviews,
    on="business_id",
    how="inner"
)

joined_df = joined_df.repartition(20)

business_write_path = 'processed/business'

joined_df.write.parquet(business_write_path, mode="overwrite", compression="snappy")

# Stop the Spark session
spark.stop()