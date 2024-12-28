from pyspark.sql import SparkSession
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from prettytable import PrettyTable

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("Evaluate Sentiment Models") \
    .getOrCreate()

# File paths for predictions of both models
gbt_predictions_path =  "./processed/sentiment_predictions_gbt"
lr_predictions_path = "./processed/sentiment_predictions_lr"

# Load predictions for both models
gbt_predictions_df = spark.read.parquet(gbt_predictions_path).withColumnRenamed("label", "label_gbt").withColumnRenamed("prediction", "prediction_gbt")
lr_predictions_df = spark.read.parquet(lr_predictions_path).withColumnRenamed("label", "label_lr").withColumnRenamed("prediction", "prediction_lr")

# Initialize evaluators
metrics = {
    "Accuracy": "accuracy",
    "Precision": "weightedPrecision",
    "Recall": "weightedRecall",
    "F1-Score": "f1"
}

def evaluate_model(predictions_df, label_col, prediction_col):
    evaluator = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol=prediction_col)
    return {metric: evaluator.setMetricName(metrics[metric]).evaluate(predictions_df) for metric in metrics}

# Evaluate both models
gbt_metrics = evaluate_model(gbt_predictions_df, "label_gbt", "prediction_gbt")
lr_metrics = evaluate_model(lr_predictions_df, "label_lr", "prediction_lr")

# Display metrics in a table
table = PrettyTable()
table.field_names = ["Metric", "GBT Model", "Logistic Regression Model"]
for metric in metrics:
    table.add_row([metric, f"{gbt_metrics[metric]:.4f}", f"{lr_metrics[metric]:.4f}"])
print(table)

# Stop the SparkSession
spark.stop()
