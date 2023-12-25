from src.components.deepface_module_fastapi import extract_embedding
from src.components.pinecone_module_fastapi import init_pinecone, query_index
import os
import json
import pinecone

def load_config(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
    
config = load_config('config.json')

# Pinecone configuration
API_KEY_PINECONE = config['API_KEY_PINECONE']
ENVIRONMENT = config['ENVIRONMENT']
INDEX_NAME = config['INDEX_NAME']
DIMENSIONS = 128

# Initialize Pinecone index
pinecone.init(      
    api_key=API_KEY_PINECONE,      
    environment=ENVIRONMENT      
    )

index = pinecone.Index(INDEX_NAME)

# Path to the test image
test_image_path = r"C:\Users\Nitish Kundu\Desktop\test_images\ronaldo_test_1.jpg"

# Extract embedding from the image
embedding, error = extract_embedding(test_image_path)
if error:
    print(error)
else:
    # Perform the query
    top_k = 1  # Number of closest matches to return
    query_response = query_index(index, embedding, top_k)
    if query_response and query_response['matches']:
        for i, match in enumerate(query_response['matches'][:5]):
            match_id = match['id']
            score = match['score']
            print(f"Match {i+1} - ID: {match_id}, Score: {score}")
    else:
        print("No matches found")