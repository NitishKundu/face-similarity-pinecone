from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Security
from fastapi.security.api_key import APIKeyHeader, APIKey
from typing import List
import os
import shutil
import tempfile
from src.components.deepface_module_fastapi import extract_embedding
from src.components.pinecone_module_fastapi import insert_to_index, query_index, remove_from_index, update_index
from src.exception import CustomException
import pinecone
import tempfile
import warnings
import json
warnings.filterwarnings("ignore")



def load_config(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

config = load_config('config.json')


API_KEY = config['API_KEY_Fastapi']
API_KEY_NAME = "access_token"
API_KEY_HEADER = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


app = FastAPI()

# Pinecone Configuration
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



async def get_api_key(api_key_header: str = Security(API_KEY_HEADER)):
    if api_key_header == API_KEY:
        return api_key_header
    else:
        raise HTTPException(status_code=403, detail="Invalid API Key")




@app.post("/AddImageToIndex")
async def add_image(file: UploadFile = File(...), api_key: APIKey = Depends(get_api_key)):
    # Create a temporary file to save the uploaded image
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_file_name = temp_file.name

    try:
        embedding, error = await extract_embedding(temp_file_name)
        if error:
            raise HTTPException(status_code=500, detail=error)

        file_id = os.path.splitext(file.filename)[0]
        await insert_to_index(index, file_id, embedding)
        return {"message": "Image added successfully", "id": file_id}
    finally:
        os.unlink(temp_file_name)



@app.delete("/DeleteImageFromIndex")
async def delete_vector(user_id: str, api_key: APIKey = Depends(get_api_key)):
    if not user_id:
        raise HTTPException(status_code=400, detail="ID is required")

    try:
        await remove_from_index(index, user_id)
        return {"message": "Vector deleted successfully", "id": user_id}
    except CustomException as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/ValidateImage")
async def query_index_endpoint(file: UploadFile = File(...), api_key: APIKey = Depends(get_api_key)):
    # Create a temporary file to save the uploaded image
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_file_name = temp_file.name

    try:
        embedding, error = await extract_embedding(temp_file_name)
        if error:
            raise HTTPException(status_code=500, detail=error)

        query_response = await query_index(index, embedding, top_k=1)
    
        results = []
        for match in query_response['matches']:
            score = match['score']
            id_value = match['id']
            message = "Exact image found" if score < 15 else "Similar image found" if score < 100 else "No similar image found"
            results.append({"id": id_value, "score": score, "message": message})
        return results
    finally:
        os.unlink(temp_file_name)
        
        
        
        
        
@app.post("/ReplaceImage")
async def update_vector_endpoint(user_id: str, file: UploadFile = File(...), api_key: APIKey = Depends(get_api_key)):
    """
    Endpoint to update a vector in the Pinecone index.
    Accepts a user ID and an image file, extracts the embedding from the image,
    and updates the vector associated with the user ID.
    """
    # Create a temporary file to save the uploaded image
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_file_name = temp_file.name

    try:
        # Extract embedding from the image file
        embedding, error = await extract_embedding(temp_file_name)

        # Cleanup: remove the temporary file
        os.unlink(temp_file_name)

        # Check for errors in embedding extraction
        if error:
            raise HTTPException(status_code=400, detail=f"Error in embedding extraction: {error}")

        # Update the vector in Pinecone index
        update_response, update_error = await update_index(index, user_id, embedding)

        # Handle possible errors during the update process
        if update_error:
            raise HTTPException(status_code=500, detail=f"Error updating vector: {update_error}")

        return {"message": "Vector updated successfully", "update_response": update_response}
    except Exception as e:
        # Cleanup: remove the temporary file in case of an error
        os.unlink(temp_file_name)
        raise HTTPException(status_code=500, detail=str(e))



# Run the FastAPI app with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5000)