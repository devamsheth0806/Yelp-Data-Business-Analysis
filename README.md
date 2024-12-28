# Yelp-Data-Business-Analysis
Predicting and Understanding Business Success with Yelp Data

## Order of execution
The following provides the order of execution of codes

Order of commands to execute files:

### Required libraries
1. pandas
2. pyarrow
3. sentence-transformers
4. weaviate-client
5. pyspark
6. google-generativeai

### Pre-processing:
1. Read business and review json files.
2. Aggregate reviews based on business
3. Join business data with reviews command:  
	`spark-submit --conf spark.executor.memory=8g --conf spark.driver.memory=8g --conf spark.executor.memoryOverhead=2g business_data_process.py`	
4. Tokenize reviews
	`spark-submit --conf spark.executor.memory=8g --conf spark.driver.memory=8g --conf spark.executor.memoryOverhead=2g tokenize_data_process.py`
	
### Feature Engineering and sentiment analysis
1. Feature engineering
	- compute positive and negative word counts in reviews
	Command: `spark-submit --conf spark.executor.memory=8g --conf spark.driver.memory=8g --conf spark.executor.memoryOverhead=2g feature_engineering_no_udf.py`

2. ML for sentiment analysis
	Command: `spark-submit --conf spark.executor.memory=8g --conf spark.driver.memory=8g --conf spark.executor.memoryOverhead=2g train_and_evaluate_sentiment.py`
	
3. Evaluate models
	Command: `spark-submit --conf spark.executor.memory=8g --conf spark.driver.memory=8g --conf spark.executor.memoryOverhead=2g ml_evaluate_sentiment_model.py`
	
4. Read predictions file (for debugging purposes)
	Command: `spark-submit --conf spark.executor.memory=8g --conf spark.driver.memory=8g --conf spark.executor.memoryOverhead=2g ml_view_sentiment_predictions.py`
	
5. Compute average sentiment scores and average star ratings
	Command: `spark-submit --conf spark.executor.memory=8g --conf spark.driver.memory=8g --conf spark.executor.memoryOverhead=2g compute_sentiment_scores.py`
	
6. Visualize Sentiment predictions from models, generates visualization like correlation heatmaps, histograms, and scatterplots
	Command: `spark-submit --conf spark.executor.memory=8g --conf spark.driver.memory=8g --conf spark.executor.memoryOverhead=2g visualize_sentiment_predictions.py`
	
7. Create Sentiment Summary for LLM
	Command: `spark-submit --conf spark.executor.memory=8g --conf spark.driver.memory=8g --conf spark.executor.memoryOverhead=2g sentiment_summary.py`

### Competitive analysis using Kmeans
1. Cluster training
	Command: `spark-submit --conf spark.executor.memory=8g --conf spark.driver.memory=8g --conf spark.executor.memoryOverhead=2g clustering_training.py`
		
2. Cluster Visualization:
	Command: `spark-submit --conf spark.executor.memory=8g --conf spark.driver.memory=8g --conf spark.executor.memoryOverhead=2g clustering_visualization.py`

### RAG with LLM
1. Install weaviate 
	Commands: `wget https://github.com/weaviate/weaviate/releases/download/v1.25.27/weaviate-v1.25.27-linux-amd64.tar.gz`
	`tar -xvzf weaviate-v1.25.27-linux-amd64.tar.gz`

2. Start weaviate: `./weaviate --tls-disabled`
	Get Port number from the verbose of command line
	
3. Create Schema and collections in Weaviate for business with reviews data and business category data
	Commands: `python create_schema_weaviate.py`
	`python create_schema_weaviate_categories.py`
	
4. Feed data into collections in weaviate
	Commands: `spark-submit --conf spark.executor.memory=8g --conf spark.driver.memory=8g --conf spark.executor.memoryOverhead=2g feed_business_categories_to_weaviate.py`
	`spark-submit --conf spark.executor.memory=8g --conf spark.driver.memory=8g --conf spark.executor.memoryOverhead=2g feed_business_categories_to_weaviate_categories.py`
	
5. Query weaviate (for debugging)
	Commands: `python query_weaviate.py`
	
6. Use RAG with LLM:
	- For a business review: `python RAG_LLM.py`
	- For reviews of top n businesses within a category: `python RAG_LLM_category.py`
	
