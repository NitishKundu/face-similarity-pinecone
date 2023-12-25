from src.components.pinecone_module_fastapi import init_pinecone, insert_to_index, query_index, remove_from_index
from src.logger import logging
from src.exception import CustomException
import pinecone
import sys
import os
import json

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

    try:
        pinecone.init(      
            api_key=API_KEY_PINECONE,      
            environment=ENVIRONMENT      
            )  
        index = pinecone.Index(INDEX_NAME)
        
        for image_file in os.listdir(IMAGE_FOLDER_PATH):
            file_id = os.path.splitext(image_file)[0]

        # Delete all vectors
            remove_from_index(index, file_id)

    except CustomException as e:
        logging.error(e)
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    main()