from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql.functions import col

# Initialize Spark session
spark = SparkSession.builder \
    .appName("YelpBusinessCompetitiveness") \
    .getOrCreate()

# Load Yelp dataset
business_data = spark.read.json("yelp_academic_dataset_business.json")  # Adjust file path
review_data = spark.read.json("yelp_academic_dataset_review.json")      # Adjust file path

assembler = VectorAssembler(
    inputCols=["average_stars", "review_count", "total_usefulness"],
    outputCol="features"
)

# Select relevant columns for competitiveness analysis
# Join business and review data to compute average ratings and review counts
business_reviews = business_data.alias("business").join(
    review_data.alias("review"),
    col("business.business_id") == col("review.business_id"),
    "inner"
).select(
    col("business.business_id").alias("business_id"),
    col("business.name").alias("name"),
    col("business.categories").alias("categories"),
    col("review.stars").alias("stars"),
    col("review.review_id").alias("review_id"),
    col("review.useful").alias("useful")
)

# Aggregate metrics: average stars, total reviews, and usefulness
competitiveness_data = business_reviews.groupBy("name") \
    .agg(
        {"stars": "avg", "review_id": "count", "useful": "sum", "categories":"first"}
    ) \
    .withColumnRenamed("avg(stars)", "average_stars") \
    .withColumnRenamed("count(review_id)", "review_count") \
    .withColumnRenamed("sum(useful)", "total_usefulness") \
    .withColumnRenamed("first(categories)", "categories")
dataset = assembler.transform(competitiveness_data).select("name", "categories", "features")

# Apply K-Means clustering to group businesses
kmeans = KMeans().setK(5).setSeed(1)  # Set K to 5 (number of clusters)
model = kmeans.fit(dataset)

# Make predictions
predictions = model.transform(dataset)

# Evaluate clustering using Silhouette score
evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(predictions)
print(f"Silhouette score: {silhouette}")

# Show clustered businesses
predictions.select("name", "categories", "prediction").show(10, truncate=False)

# Save the model if needed
model.save("kmeans_model")

output_path = 'processed/business_reviews'
competitiveness_data.write.parquet(output_path, mode="overwrite", compression="snappy")