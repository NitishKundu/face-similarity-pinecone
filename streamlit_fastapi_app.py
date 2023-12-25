import streamlit as st
from PIL import Image
import os
import asyncio
import pinecone
import tempfile
import cv2
from deepface import DeepFace
from src.components.deepface_module_fastapi import _extract_embedding_sync
from src.components.pinecone_module_fastapi import _query_index_sync
from src.exception import CustomException
import json


def load_config(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

config = load_config('config.json')



# Resize image function
def resize_image(image, size=(500, 700)):
    return image.resize(size, Image.Resampling.LANCZOS)


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


# Async functions
async def extract_embedding(image_input):
    try:
        return await asyncio.to_thread(_extract_embedding_sync, image_input)
    except Exception as e:
        raise CustomException(str(e), sys)

async def query_index(index, embedding, top_k):
    try:
        return await asyncio.to_thread(_query_index_sync, index, embedding, top_k)
    except Exception as e:
        raise CustomException(str(e), sys)

# Utility function to run async functions in Streamlit
def run_async(func):
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(func(*args, **kwargs))
    return wrapper



# Streamlit App

st.title("Image Matching Application")


# Upload Image
st.header("Upload Image")
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

col1, col2 = st.columns(2)

with col1:
    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert('RGB')
        resized_image = resize_image(image)
        st.image(resized_image, caption='Uploaded Image')

matched_image_placeholder = col2.empty()

# Search button
if st.button("Search"):
    if uploaded_image is not None:
        embedding, error = run_async(extract_embedding)(image)
        if error:
            st.error(error)
        else:
            top_k = 1
            query_response = run_async(query_index)(index, embedding, top_k)
            if query_response and query_response['matches']:
                match = query_response['matches'][0]
                match_id = match['id']
                score = match['score']

                if score < 100:
                    image_path = os.path.join(r"C:\Users\Nitish Kundu\Documents\image_data\images", match_id + ".jpg")
                    if os.path.exists(image_path):
                        matched_image = Image.open(image_path)
                        resized_matched_image = resize_image(matched_image)
                        col2.image(resized_matched_image, caption=f"Matched Image: {match_id}")
                        col2.write(f"Score: {score}")
                        if score < 15:
                            col2.success("Exact image found")
                        else:
                            col2.warning("Similar image found")
                else:
                    col2.error("No similar image found")
    else:
        st.error("Please upload an image before searching.")