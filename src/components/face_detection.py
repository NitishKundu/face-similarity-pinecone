# face_detection.py
from mtcnn import MTCNN
import cv2
import sys
from src.exception import CustomException
from src.logger import logging

class FaceDetector:
    """
    Class for detecting faces in an image using MTCNN.
    """

    def __init__(self):
        try:
            self.detector = MTCNN()
            logging.info("MTCNN Face Detector initialized successfully.")
        except Exception as e:
            raise CustomException(e, sys) from e

    def detect_face(self, image):
        """
        Detects a face in the given image.

        Args:
            image (numpy.ndarray): The input image.

        Returns:
            numpy.ndarray: The detected face image, or None if no face is detected.
        """
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.detector.detect_faces(image)
            if results:
                bounding_box = results[0]['box']
                x, y, width, height = bounding_box
                face = image[y:y+height, x:x+width]
                logging.info("Face detected successfully.")
                return face
            else:
                logging.info("No face detected in the image.")
                return None
        except Exception as e:
            raise CustomException(e, sys) from e
