from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import GBTClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import col, lit

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("Sentiment Analysis with Metrics Evaluation") \
    .config("spark.sql.files.maxPartitionBytes", "64m") \
    .config("spark.sql.shuffle.partitions", "100") \
    .config("spark.executor.memory", "16g") \
    .config("spark.driver.memory", "16g") \
    .config("spark.executor.memoryOverhead", "4g") \
    .getOrCreate()

# Step 1: Load the Sentiment Feature File
input_path = "./processed/sentiment_analysis"
df = spark.read.parquet(input_path)

# Step 2: Select Necessary Columns
required_columns = ["business_id", "positive_count", "negative_count", "review_length", "stars"]
df_filtered = df.select(*required_columns)

# Step 3: Calculate Dynamic Threshold
threshold = df_filtered.approxQuantile("stars", [0.5], 0.01)[0]  # Median
print(f"Dynamic threshold for labeling: {threshold}")

# Step 4: Define Labels Dynamically
df_filtered = df_filtered.withColumn("label", (col("stars") > lit(threshold)).cast("int"))

# Step 5: Combine Features
assembler = VectorAssembler(
    inputCols=["positive_count", "negative_count", "review_length"],
    outputCol="features"
)
df_with_features = assembler.transform(df_filtered).select("features", "label", "business_id")

# Step 6: Split Data into Train and Test Sets
train_data, test_data = df_with_features.randomSplit([0.8, 0.2], seed=42)

# Step 7: Train GBTClassifier
print("Training GBTClassifier...")
gbt = GBTClassifier(featuresCol="features", labelCol="label", maxIter=50)
gbt_model = gbt.fit(train_data)

# Step 8: Train Logistic Regression
print("Training Logistic Regression...")
lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=10)
lr_model = lr.fit(train_data)

# Step 9: Apply Models to Test Data
print("Evaluating Models...")
gbt_predictions = gbt_model.transform(test_data)
lr_predictions = lr_model.transform(test_data)

# Step 10: Define Evaluation Metrics
def evaluate_model(predictions, model_name):
    evaluator_roc_auc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    roc_auc = evaluator_roc_auc.evaluate(predictions)

    tp = predictions.filter((col("label") == 1) & (col("prediction") == 1)).count()
    tn = predictions.filter((col("label") == 0) & (col("prediction") == 0)).count()
    fp = predictions.filter((col("label") == 0) & (col("prediction") == 1)).count()
    fn = predictions.filter((col("label") == 1) & (col("prediction") == 0)).count()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"\n{model_name} Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"F1-Score: {f1_score:.2%}")
    print(f"ROC-AUC: {roc_auc:.2%}")

# Step 11: Evaluate Models
evaluate_model(gbt_predictions, "GBTClassifier")
evaluate_model(lr_predictions, "Logistic Regression")

# Step 12: Save Predictions
output_path_gbt = "./processed/sentiment_predictions_gbt"
output_path_lr = "./processed/sentiment_predictions_lr"
gbt_predictions.select("business_id", "label", "prediction", "probability").write.parquet(output_path_gbt, mode="overwrite", compression="snappy")
lr_predictions.select("business_id", "label", "prediction", "probability").write.parquet(output_path_lr, mode="overwrite", compression="snappy")

# Stop SparkSession
spark.stop()
