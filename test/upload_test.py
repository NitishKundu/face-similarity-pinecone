from src.components.deepface_module_fastapi import extract_embedding
from src.components.pinecone_module_fastapi import init_pinecone, insert_to_index, query_index, remove_from_index
from src.logger import logging
from src.exception import CustomException
import os
import json
import pinecone



def load_config(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


def main():
    
    config = load_config('config.json')
    API_KEY_PINECONE = config['API_KEY_PINECONE']
    ENVIRONMENT = config['ENVIRONMENT']
    INDEX_NAME = config['INDEX_NAME']
    DIMENSIONS = 128
    IMAGE_FOLDER_PATH = "C:/Users/Nitish Kundu/Documents/image_data/images"

    pinecone.init(      
            api_key=API_KEY_PINECONE,      
            environment=ENVIRONMENT,
            dimensions=DIMENSIONS   
            )
    index = pinecone.Index(INDEX_NAME)

    for image_file in os.listdir(IMAGE_FOLDER_PATH):
        file_id = os.path.splitext(image_file)[0]
        image_path = os.path.join(IMAGE_FOLDER_PATH, image_file)
        embedding, error = extract_embedding(image_path)

        if error:
            logging.info(f"{error} in image {image_file}")
            continue

        insert_to_index(index, file_id, embedding)
    
    logging.info('Index created and uploded successfully')   
    print('Index created and uploded successfully')


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")