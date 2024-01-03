import asyncio
from src.components.deepface_module_fastapi import extract_embedding
from src.components.pinecone_module_fastapi import insert_to_index
import os
import json
import pinecone
from src.logger import logging



def load_config(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


async def main():
    config = load_config('config.json')
    API_KEY_PINECONE = config['API_KEY_PINECONE']
    ENVIRONMENT = config['ENVIRONMENT']
    INDEX_NAME = config['INDEX_NAME']
    DIMENSIONS = config['DIMENSIONS']
    IMAGE_FOLDER_PATH = "C:/Users/Nitish Kundu/Documents/image_data/images"

    # Initialize Pinecone
    pinecone.init(api_key=API_KEY_PINECONE, environment=ENVIRONMENT)

    # Check if the index already exists
    active_indexes = pinecone.list_indexes()
    if INDEX_NAME not in active_indexes:
        pinecone.create_index(name=INDEX_NAME, dimension=DIMENSIONS)
        logging.info(f'Created new Pinecone index: {INDEX_NAME}')
    else:
        logging.info(f'Index "{INDEX_NAME}" already exists.')

    # Connect to the Pinecone index
    index = pinecone.Index(INDEX_NAME)

    for image_file in os.listdir(IMAGE_FOLDER_PATH):
        file_id = os.path.splitext(image_file)[0]
        image_path = os.path.join(IMAGE_FOLDER_PATH, image_file)
        embedding, error = await extract_embedding(image_path)

        if error:
            logging.info(f"{error} in image {image_file}")
            continue

        await insert_to_index(index, file_id, embedding)

    logging.info('Embeddings uploaded successfully')   
    print('Embeddings uploaded successfully')

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
