from math import pi
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Paths to dynamically load data
gbt_predictions_path = "./processed/sentiment_predictions_gbt"
lr_predictions_path = "./processed/sentiment_predictions_lr"

# Load GBT predictions
gbt_predictions = pd.read_parquet(gbt_predictions_path)
# Load Logistic Regression predictions
lr_predictions = pd.read_parquet(lr_predictions_path)

# Function to calculate evaluation metrics
def calculate_metrics(predictions_df, model_name):
    tp = predictions_df[(predictions_df["label"] == 1) & (predictions_df["prediction"] == 1)].shape[0]
    tn = predictions_df[(predictions_df["label"] == 0) & (predictions_df["prediction"] == 0)].shape[0]
    fp = predictions_df[(predictions_df["label"] == 0) & (predictions_df["prediction"] == 1)].shape[0]
    fn = predictions_df[(predictions_df["label"] == 1) & (predictions_df["prediction"] == 0)].shape[0]
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1_score
    }

# Calculate metrics for both models
gbt_metrics = calculate_metrics(gbt_predictions, "GBT Model")
lr_metrics = calculate_metrics(lr_predictions, "Logistic Regression")

# Combine metrics for radar chart
metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
gbt_values = [gbt_metrics[m] for m in metrics]
lr_values = [lr_metrics[m] for m in metrics]

# Radar Chart
def plot_radar_chart(gbt_values, lr_values, metrics):
    # Close the radar chart by appending the first metric
    gbt_values += gbt_values[:1]
    lr_values += lr_values[:1]
    metrics_with_closure = metrics + [metrics[0]]  # Add closure label for metrics

    # Angles for the radar chart
    angles = [n / float(len(metrics_with_closure)) * 2 * pi for n in range(len(metrics_with_closure))]

    # Create the radar chart
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Plot GBT Model
    ax.plot(angles, gbt_values, linewidth=2, linestyle="solid", label="GBT Model")
    ax.fill(angles, gbt_values, alpha=0.25)

    # Plot Logistic Regression Model
    ax.plot(angles, lr_values, linewidth=2, linestyle="solid", label="Logistic Regression")
    ax.fill(angles, lr_values, alpha=0.25)

    # Add labels and legend
    ax.set_xticks(angles)
    ax.set_xticklabels(metrics_with_closure)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], color="grey", size=8)
    ax.set_title("Radar Chart: Model Performance Metrics", y=1.1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    # Save and show
    plt.savefig("./visualizations/metric_radar_chart.png")
    plt.show()

plot_radar_chart(gbt_values, lr_values, metrics)

# Confusion Matrices
def plot_confusion_matrices(gbt_predictions, lr_predictions):
    # Compute confusion matrices
    gbt_cm = confusion_matrix(gbt_predictions["label"], gbt_predictions["prediction"])
    lr_cm = confusion_matrix(lr_predictions["label"], lr_predictions["prediction"])
    
    # Plot confusion matrices side by side
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    # GBT Confusion Matrix
    disp_gbt = ConfusionMatrixDisplay(confusion_matrix=gbt_cm, display_labels=["Negative", "Positive"])
    disp_gbt.plot(ax=ax[0], cmap="Blues", colorbar=False)
    ax[0].set_title("GBT Model Confusion Matrix")
    
    # Logistic Regression Confusion Matrix
    disp_lr = ConfusionMatrixDisplay(confusion_matrix=lr_cm, display_labels=["Negative", "Positive"])
    disp_lr.plot(ax=ax[1], cmap="Greens", colorbar=False)
    ax[1].set_title("Logistic Regression Confusion Matrix")
    
    # Save and show
    plt.savefig("./visualizations/confusion_matrix_comparison.png")
    plt.show()

plot_confusion_matrices(gbt_predictions, lr_predictions)
