import matplotlib.pyplot as plt
from pyspark.ml.clustering import KMeansModel
from pyspark.ml.feature import VectorAssembler

from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Kmeans") \
    .getOrCreate()

# Load the trained KMeans model
model = KMeansModel.load("kmeans_model")
assembler = VectorAssembler(
    inputCols=["average_stars", "review_count", "total_usefulness"],
    outputCol="features"
)
# Get the cluster centers
cluster_centers = model.clusterCenters()

# Convert cluster centers to a 2D array for visualization
# (Use only the second two features for simplicity)
cluster_centers_2d = [[center[1],center[2]] for center in cluster_centers]

competitiveness_data = spark.read.parquet('processed/business_reviews')
competitiveness_data = assembler.transform(competitiveness_data)
# Extract features and predictions for visualization
predictions = model.transform(competitiveness_data)
features_and_clusters = predictions.select("features", "prediction").collect()

# Separate points into clusters
clusters = {}
for row in features_and_clusters:
    cluster_id = row["prediction"]
    point = [row["features"][1], row["features"][2]] # Use first two dimensions
    if cluster_id not in clusters:
        clusters[cluster_id] = []
    clusters[cluster_id].append(point)

# Plot the clusters
colors = ['r', 'g', 'b', 'y', 'c']  # Color palette for clusters

plt.figure(figsize=(10, 6))
for cluster_id, points in clusters.items():
    points_x = [point[0] for point in points]
    points_y = [point[1] for point in points]
    plt.scatter(points_x, points_y, c=colors[cluster_id % len(colors)], label=f"Cluster {cluster_id}")

# Plot the cluster centers
centers_x = [center[0] for center in cluster_centers_2d]
centers_y = [center[1] for center in cluster_centers_2d]
plt.scatter(centers_x, centers_y, c='k', marker='x', s=100, label="Centroids")

plt.title("KMeans Clustering")
plt.xlabel("Review Count")
plt.ylabel("Usefulness")
plt.legend()
plt.savefig('cluster.png', bbox_inches="tight")
