import asyncio
import pinecone
from PIL import Image
import json

# Import your asynchronous functions
from src.components.deepface_module_fastapi  import extract_embedding
from src.components.pinecone_module_fastapi import update_index


def load_config(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
    
config = load_config('config.json')

# Pinecone Configuration
API_KEY_PINECONE = config['API_KEY_PINECONE']
ENVIRONMENT = config['ENVIRONMENT']
INDEX_NAME = config['INDEX_NAME']


# Initialize Pinecone index
pinecone.init(      
	api_key=API_KEY_PINECONE,      
	environment=ENVIRONMENT      
)      
index = pinecone.Index(INDEX_NAME)


# Test image path
IMAGE_PATH = r"C:\Users\Nitish Kundu\Documents\image_data\images\Leo_messi_3.jpg"  # Replace with the path to your test image

# Asynchronous test function
async def test_update_index():
    # Load the image
    image = Image.open(IMAGE_PATH).convert('RGB')

    # Extract embedding from the image
    embedding, error = await extract_embedding(image)
    if error:
        print(f"Error extracting embedding: {error}")
        return

    vector_id = "Leo_messi_3"  # Replace with an actual vector ID in your index

    # Call the update_index function
    update_response, error = await update_index(index, vector_id, embedding)

    # Check the result
    if error:
        print(f"Error updating vector: {error}")
    else:
        print(f"Vector updated successfully: {update_response}")

# Run the test
asyncio.run(test_update_index())