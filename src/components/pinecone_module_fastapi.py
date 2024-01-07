import asyncio
import pinecone
from src.exception import CustomException
from src.logger import logging
import sys




async def insert_to_index(index, user_id, embedding):
    """
    Asynchronously inserts a vector with the given user ID and embedding into the Pinecone index.

    Args:
        index (PineconeIndex): The Pinecone index to insert the vector into.
        user_id (str): The ID of the vector.
        embedding (list): The embedding vector to be inserted.

    Raises:
        CustomException: If there is an error inserting data into the Pinecone index.

    Returns:
        None
    """
    try:
        await asyncio.to_thread(_insert_to_index_sync, index, user_id, embedding)
    except Exception as e:
        raise CustomException(str(e), sys)

def _insert_to_index_sync(index, user_id, embedding):
    # Original synchronous code of insert_to_index goes here
    try:
        # Check if the vector with the given ID already exists
        existing_vector = index.fetch(ids=[user_id])
        if existing_vector['vectors'].get(user_id) is not None:
            logging.info(f"Vector with ID {user_id} already exists in the index. Skipping insertion.")
            return

        if not embedding or not isinstance(embedding, list):
            logging.warning(f"Invalid embedding for {user_id}. Skipping insertion.")
            return

        # Structure the data correctly for Pinecone's upsert method
        vector_data = {'id': user_id, 'values': embedding}
        logging.info(f"Inserting {user_id} with embedding: {embedding[:10]}...")
        index.upsert(vectors=[vector_data])
        logging.info(f"Inserted embedding for {user_id} into Pinecone index.")
    except Exception as e:
        logging.error(f"Error inserting data into Pinecone index for {user_id}: {str(e)}")
        
        
        
        
        
        
        
async def query_index(index, embedding, top_k):
    """
    Asynchronously queries the Pinecone index with the given embedding vector and returns the query response.

    Args:
        index (PineconeIndex): The Pinecone index to query.
        embedding (numpy.ndarray): The embedding vector to query with.
        top_k (int): The number of nearest neighbors to retrieve.

    Returns:
        dict: The query response containing the nearest neighbors and their distances.

    Raises:
        CustomException: If there is an error querying the Pinecone index.
    """
    try:
        return await asyncio.to_thread(_query_index_sync, index, embedding, top_k)
    except Exception as e:
        raise CustomException(str(e), sys)

def _query_index_sync(index, embedding, top_k):
    # Original synchronous code of query_index goes here
    try:
        query_response = index.query(top_k=top_k, include_values=True, vector=embedding)
        logging.info(f"Query executed in Pinecone index. Response: {query_response}")
        return query_response
    except Exception as e:
        logging.error(f"Error querying Pinecone index: {str(e)}")
        
        
        
        
        
        
async def remove_from_index(index, ids):
    """
    Asynchronously removes vectors from the Pinecone index.

    Args:
        index (Pinecone.Index): The Pinecone index object.
        ids (str or list): The ID(s) of the vectors to be removed from the index.

    Raises:
        CustomException: If there is an error removing data from the Pinecone index.

    Returns:
        None
    """
    try:
        await asyncio.to_thread(_remove_from_index_sync, index, ids)
    except Exception as e:
        raise CustomException(str(e), sys)

def _remove_from_index_sync(index, ids):
    # Original synchronous code of remove_from_index goes here
    try:
        # Check if ids is a list or a single ID and format it for deletion
        ids_to_delete = ids if isinstance(ids, list) else [ids]

        # Perform the deletion
        delete_response = index.delete(ids=ids_to_delete)
        logging.info(f"Deleted vectors with IDs: {ids_to_delete}, response: {delete_response}")
    except Exception as e:
        logging.error(f"Error removing data from Pinecone index: {str(e)}")
        
        
        
        
        
# Asynchronous function for updating a vector
async def update_index(index, vector_id, new_embedding):
    try:
        # Update the vector with the new embedding
        update_response = await asyncio.to_thread(
            _update_index_sync, index, vector_id, new_embedding
        )
        return update_response, None
    except Exception as e:
        return None, str(e)

# Synchronous function for updating a vector
def _update_index_sync(index, vector_id, new_values):
    try:
        update_response = index.update(
            id=vector_id,
            values=new_values
        )
        return update_response
    except Exception as e:
        raise e
    
    
    
    
    
async def insert_to_index_full(index, user_ids, embeddings):
    """
    Asynchronously inserts vectors with given user IDs and embeddings into the Pinecone index.
    This function supports both single and multiple vector insertions.

    Args:
        index (PineconeIndex): The Pinecone index to insert vectors into.
        user_ids (list of str): The IDs of the vectors.
        embeddings (list of list): The embedding vectors to be inserted.

    Raises:
        CustomException: If there is an error inserting data into the Pinecone index.

    Returns:
        None
    """
    try:
        await asyncio.to_thread(_insert_to_index_full_sync, index, user_ids, embeddings)
    except Exception as e:
        raise CustomException(str(e), sys)

def _insert_to_index_full_sync(index, user_ids, embeddings):
    try:
        vector_data = []
        for user_id, embedding in zip(user_ids, embeddings):
            # Check if the vector with the given ID already exists
            existing_vector = index.fetch(ids=[user_id])
            if existing_vector['vectors'].get(user_id) is not None:
                logging.info(f"Vector with ID {user_id} already exists in the index. Skipping insertion.")
                continue

            if not embedding or not isinstance(embedding, list):
                logging.warning(f"Invalid embedding for {user_id}. Skipping insertion.")
                continue

            # Structure the data correctly for Pinecone's upsert method
            vector_data.append({'id': user_id, 'values': embedding})

        if not vector_data:
            logging.warning("No valid data to insert. Skipping.")
            return

        logging.info(f"Inserting embeddings for user IDs: {user_ids}...")
        index.upsert(vectors=vector_data)
        logging.info("Inserted embeddings into Pinecone index.")
    except Exception as e:
        logging.error(f"Error inserting data into Pinecone index: {str(e)}")
