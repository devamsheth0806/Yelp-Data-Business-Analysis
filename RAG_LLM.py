import os
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModel
import weaviate
import pandas as pd
import torch

client = weaviate.Client(
    url="http://10.32.50.3:34511"
)
genai.configure(api_key=os.environ["API_KEY"])
llm_model = genai.GenerativeModel("gemini-1.5-flash")

def fetch_reviews_from_vector_db(business_name, business_name_embedding):
    """
    Fetch reviews from the vector database based on business name.
    """
    try:
        # Query Weaviate for the business name
        results = client.query.get("BusinessReview", ["business_name", "average_sentiment_score", "stars_list", "useful_list", "review_list"]) \
            .with_near_vector({
                'vector': business_name_embedding,
                'certainty': 0.7
            }).with_limit(1).do()
        
        # Extract results
        data = results.get("data", {}).get("Get", {}).get("BusinessReview", [])
        if not data:
            raise ValueError(f"No data found for business: {business_name}")
        
        return data[0]
    except Exception as e:
        print(f"Error fetching data: {e}")
        return []

def generate_summary_and_sentiment(data):
    """
    Use an LLM to generate a summary and sentiment analysis of the reviews.
    """
    # Prepare the prompt for the LLM
    df = pd.DataFrame(data)
    df = df[['review_list','useful_list','stars_list']]
    
    llm_input = "\n".join(
    f"Review: {row['review_list']} | Usefulness: {row['useful_list']} | Reviewer_Rating: {row['stars_list']}" for _, row in df.iterrows())

    # Create the prompt for the LLM
    prompt = f"""
    Below are some reviews with corresponding usefulness and their corresponding rating for a business, named {data['business_name']}.
    Summarize the overall sentiment of the reviews, the predictable star rating, and provide a short summary of key points mentioned in the reviews:

    {llm_input}

    Average Sentiment Score: {data['average_sentiment_score']}
    """

    try:
        # Call the LLM (ChatGPT or Gemini) via API
        response = llm_model.generate_content(prompt, stream=True)
        for chunk in response:
            print(chunk.text, end='')
    except Exception as e:
        print(f"Error generating summary: {e}")

def main():
    """
    Main function to execute the script.
    """
    # User input
    embed_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    
    business_name = input("Enter the business name: ").strip()
    tokenized_review = tokenizer(business_name, return_tensors="pt", truncation=True, padding=True).to(device)
    embedding = embed_model(**tokenized_review).last_hidden_state.mean(dim=1).squeeze().detach().cpu().numpy()
    
    # Fetch data from vector database
    reviews_data = fetch_reviews_from_vector_db(business_name, embedding)
    if not reviews_data:
        print("No reviews found for the provided business name.")
        return

    # Generate summary and sentiment analysis
    generate_summary_and_sentiment(reviews_data)


if __name__ == "__main__":
    main()
