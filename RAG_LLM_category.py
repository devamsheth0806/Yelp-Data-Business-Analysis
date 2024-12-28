import os
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModel
import weaviate
import pandas as pd
import torch

client = weaviate.Client(
    url="http://10.32.50.3:35603"
)

genai.configure(api_key=os.environ["API_KEY"])
llm_model = genai.GenerativeModel("gemini-1.5-flash")

def fetch_business_from_vector_db(category_name, category_name_embedding, n):
    """
    Fetch reviews from the vector database based on business name.
    """
    try:
        # Query Weaviate for the business name
        results = client.query.get("BusinessCategories", ["business_name"]) \
            .with_near_vector({
                'vector': category_name_embedding,
                'certainty': 0.7
            }).with_limit(n).do()
        
        # Extract results
        data = results.get("data", {}).get("Get", {}).get("BusinessCategories", [])
        print(data)
        if not data:
            raise ValueError(f"No data found for business: {category_name}")
        
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return []


def fetch_reviews_from_vector_db(business_name, business_name_embedding):
    """
    Fetch reviews from the vector database based on business name.
    """
    try:
        # Query Weaviate for the business name
        results = client.query.get("BusinessReview", ["business_name", "average_sentiment_score", "stars_list", "useful_list", "review_list"]) \
            .with_near_vector({
                'vector': business_name_embedding,
                'certainty': 0.5
            }).with_limit(1).do()
        
        # Extract results
        data = results.get("data", {}).get("Get", {}).get("BusinessReview", [])
        if not data:
            raise ValueError(f"No data found for business: {business_name}")
        
        return data[0]
    except Exception as e:
        print(f"Error fetching data: {e}")
        return []

def generate_summary_and_sentiment(data, category_name):
    """
    Use an LLM to generate a summary and sentiment analysis of the reviews.
    """
    # Prepare the prompt for the LLM
    llm_input = ""
    for business_name in list(data.keys()):
        df = pd.DataFrame(data[business_name])
        df = df[['review_list','useful_list','stars_list']]
        llm_input +=  business_name + " review data: "
        llm_input += "\n".join(
        f"Review: {row['review_list']} | Usefulness: {row['useful_list']} | Reviewer_Rating: {row['stars_list']}" for _, row in df.iterrows())
        
        llm_input += "Average Sentiment Score: {data['average_sentiment_score']}"
    # Create the prompt for the LLM
    prompt = f"""
    Below are some reviews with corresponding usefulness and their corresponding rating for a few businesses in category {category_name}.
    Summarize the overall sentiment of the reviews, the predictable star rating, and provide a short summary of key points mentioned in the reviews:

    {llm_input}

    
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
    
    category_name = input("Enter the category name: ").strip()
    n = int(input("Number of top businesses: "))
    tokenized_review = tokenizer(category_name, return_tensors="pt", truncation=True, padding=True).to(device)
    embedding = embed_model(**tokenized_review).last_hidden_state.mean(dim=1).squeeze().detach().cpu().numpy()
    
    businesses = fetch_business_from_vector_db(category_name, embedding, n)
    full_data = {}
    for business in businesses:
        business_name = business["business_name"]
        tokenized_review = tokenizer(business_name, return_tensors="pt", truncation=True, padding=True).to(device)
        embedding = embed_model(**tokenized_review).last_hidden_state.mean(dim=1).squeeze().detach().cpu().numpy()
            
        # Fetch data from vector database
        reviews_data = fetch_reviews_from_vector_db(business_name, embedding)
        
        if not reviews_data:
            print("No reviews found for the provided business name.")
            return
        
        full_data[business_name]=reviews_data
        
    # Generate summary and sentiment analysis
    generate_summary_and_sentiment(full_data, category_name)


if __name__ == "__main__":
    main()
