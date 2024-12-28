import weaviate
from transformers import AutoTokenizer, AutoModel
import torch
# Initialize the client
client = weaviate.Client(url="http://10.32.50.3:34511")
class_name = "BusinessCategories"
schema = client.schema.get()

# Extract properties of the class
properties = []
for cls in schema["classes"]:
    if cls["class"] == class_name:
        properties = [prop["name"] for prop in cls["properties"]]
        break  
embed_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embed_model.to(device)
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

vector = 'cafes'
tokenized_review = tokenizer(vector, return_tensors="pt", truncation=True, padding=True).to(device)
embedding = embed_model(**tokenized_review).last_hidden_state.mean(dim=1).squeeze().detach().cpu().numpy()
if not properties:
    print(f"No properties found for class '{class_name}'")
else:
    # Build and execute a dynamic GraphQL query
    query = f"""
    {{
      Get {{
        {class_name} (
          limit: 5
          nearVector:{{
            vector: {embedding}
            certainty: 0.8
          }}
        ) {{
          {" ".join(properties)}
        }}
      }}
    }}
    """
    result = client.query.raw(query)
    print(f"Data for class '{class_name}':", result)
