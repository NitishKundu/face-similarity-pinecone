import asyncio
from deepface import DeepFace
from deepface.commons import functions
import cv2
import tempfile
import sys
from PIL import Image
import numpy as np
from src.exception import CustomException
from src.logger import logging
from src.components.face_detection import FaceDetector
import os



async def extract_embedding(image_input):
    """
    Extracts the facial embedding from an image asynchronously.

    Args:
        image_input (str or PIL.Image.Image): The path to the image file or the image as a PIL image object.

    Returns:
        tuple: A tuple containing the facial embedding (numpy array) and an error message (str).
               If the embedding is successfully extracted, the error message will be None.
               If no face is detected or the embedding could not be created, the embedding will be None
               and the error message will provide more details.

    Raises:
        CustomException: If any error occurs during the extraction process.
    """
    try:
        return await asyncio.to_thread(_extract_embedding_sync, image_input)
    except Exception as e:
        raise CustomException(str(e), sys)

def _extract_embedding_sync(image_input):
    
    try:
        # Check if the input is a file path (string) or an image (PIL image)
        if isinstance(image_input, str):
            img = cv2.imread(image_input)
            if img is None:
                raise ValueError(f"Image at {image_input} cannot be read.")
        elif isinstance(image_input, Image.Image):
            img = np.array(image_input)
            img = img[:, :, ::-1]  # Convert RGB to BGR, which OpenCV expects
        else:
            raise ValueError("Invalid image input. Must be a file path or a PIL image.")

        # Initialize the FaceDetector and detect face
        face_detector = FaceDetector()
        face = face_detector.detect_face(img)
        if face is None:
            return None, "No face detected in the image."
        
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmpfile:
            face_image = Image.fromarray(face)  # Convert NumPy array to PIL Image
            face_image.save(tmpfile.name)
        

        # Proceed with embedding extraction
        embedding = DeepFace.represent(tmpfile.name, model_name='Facenet', enforce_detection=False)
        if embedding is None:
            return None, "Embedding could not be created."
        os.unlink(tmpfile.name)
        return embedding, None
    except Exception as e:
        # Here, we're adding more details to understand the error better
        error_info = {
            "error_message": str(e),
            "type": str(type(e)),
            "image_input_type": str(type(image_input)),
            "image_shape": str(img.shape if 'img' in locals() else 'Image not processed'),
            "face_shape": str(face.shape if 'face' in locals() else 'Face not detected')
        }
        raise CustomException(f"Error during embedding extraction: {error_info}", sys)