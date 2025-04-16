import cv2
import numpy as np
import os
from django.conf import settings
from PIL import Image
from insightface.app import FaceAnalysis
import logging

# Set up logging
logger = logging.getLogger(__name__)


class FaceRecognition:
    def __init__(self):
        self.known_face_embeddings = []
        self.known_face_ids = []
        self.known_admission_numbers = []
        self.min_confidence = 0.6  # Minimum confidence threshold for recognition
        self.app = None

        self._initialize_face_analysis()

    def _initialize_face_analysis(self):
        """Initialize InsightFace with multiple fallback options"""
        model_priority = [
            {'name': 'buffalo_l', 'det_size': (640, 640)},
            {'name': 'buffalo_s', 'det_size': (640, 640)},
            {'name': 'buffalo_sc', 'det_size': (320, 320)}
        ]

        for model in model_priority:
            try:
                logger.info(f"Attempting to initialize FaceAnalysis with model: {model['name']}")
                self.app = FaceAnalysis(name=model['name'])
                self.app.prepare(ctx_id=0, det_size=model['det_size'])

                # Test with dummy image to verify working
                test_img = np.zeros((100, 100, 3), dtype=np.uint8)
                test_faces = self.app.get(test_img)
                logger.info(f"Successfully initialized with model: {model['name']}")
                return
            except Exception as e:
                logger.error(f"Failed to initialize with {model['name']}: {str(e)}")
                self.app = None
                continue

        logger.error("All FaceAnalysis initialization attempts failed")
        self.app = None

    def ensure_supported_format(self, image):
        """Convert image to RGB uint8 format for InsightFace"""
        try:
            if image is None:
                return None

            img = image.copy()

            if img.dtype != np.uint8:
                if img.max() <= 1.0:  # Float image in 0-1 range
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)

            # Handle grayscale (2D) images
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            # Handle RGBA (4 channel) images
            elif len(img.shape) == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            # Handle BGR (3 channel) images
            elif len(img.shape) == 3 and img.shape[2] == 3:
                rgb_from_bgr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if not np.array_equal(img, rgb_from_bgr):
                    img = rgb_from_bgr

            if len(img.shape) != 3 or img.shape[2] != 3 or img.dtype != np.uint8:
                logger.warning(f"Final validation failed: shape={img.shape}, dtype={img.dtype}")
                return None

            return img
        except Exception as e:
            logger.error(f"Error in ensure_supported_format: {str(e)}")
            return None

    def clean_image(self, image):
        """Apply preprocessing to clean and standardize the image"""
        try:
            img = self.ensure_supported_format(image)
            if img is None:
                return None

            # Normalize lighting using CLAHE
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

            return final_img
        except Exception as e:
            logger.error(f"Error in clean_image: {str(e)}")
            return None

    def load_and_preprocess_image(self, image_path):
        """Load and preprocess an image from file"""
        try:
            # Try with OpenCV first
            image = cv2.imread(image_path)
            if image is None:
                # Fallback to PIL if OpenCV fails
                with Image.open(image_path) as pil_img:
                    if pil_img.mode == 'L':  # Grayscale
                        image = np.array(pil_img)
                    elif pil_img.mode == 'RGBA':
                        image = np.array(pil_img.convert('RGBA'))
                    else:
                        image = np.array(pil_img.convert('RGB'))

            # Ensure supported format
            supported_image = self.ensure_supported_format(image)
            if supported_image is None:
                logger.warning("Failed to convert to supported format")
                return None

            # Clean the image
            cleaned_image = self.clean_image(supported_image)
            if cleaned_image is None:
                logger.warning("Image cleaning failed")
                return None

            return cleaned_image
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            return None

    def load_student_images(self):
        """Load and preprocess all student images using InsightFace"""
        from .models import Student

        if self.app is None:
            logger.error("Cannot load student images - FaceAnalysis not initialized")
            return False

        students = Student.objects.all()
        logger.info(f"Loading {len(students)} student images...")

        # Clear existing data
        self.known_face_embeddings = []
        self.known_face_ids = []
        self.known_admission_numbers = []

        success_count = 0
        for student in students:
            if not student.photo:
                logger.warning(f"No photo for student {student.admission_number}")
                continue

            image_path = os.path.normpath(os.path.join(settings.MEDIA_ROOT, str(student.photo)))
            logger.debug(f"Processing student {student.admission_number}")
            logger.debug(f"Image path: {image_path}")

            if not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}")
                continue

            try:
                # Load and clean the image
                rgb_img = self.load_and_preprocess_image(image_path)
                if rgb_img is None:
                    logger.warning(f"Failed to load/process image for {student.admission_number}")
                    continue

                # Detect faces using InsightFace
                faces = self.app.get(rgb_img)
                if not faces:
                    logger.warning(f"No faces detected for {student.admission_number}")
                    continue

                # Use the first face found (assuming one face per student image)
                face = faces[0]
                if face.det_score < 0.5:  # Minimum detection confidence
                    logger.warning(f"Face detection confidence too low: {face.det_score}")
                    continue

                # Store the embedding and student info
                self.known_face_embeddings.append(face.embedding)
                self.known_face_ids.append(student.id)
                self.known_admission_numbers.append(student.admission_number)
                success_count += 1
                logger.info(f"Processed student {student.admission_number}")

            except Exception as e:
                logger.error(f"Error processing {student.admission_number}: {str(e)}", exc_info=True)

        logger.info(f"Loaded {success_count} face embeddings from {len(students)} students")
        return success_count > 0

    def is_ready(self):
        """Check if the system is ready for recognition"""
        return self.app is not None and len(self.known_face_embeddings) > 0

    def process_webcam_frame(self, frame):
        """Process a frame from webcam and compare with stored faces using InsightFace"""
        if not self.is_ready():
            logger.warning("System not ready for recognition")
            return []

        try:
            # Convert frame to supported format
            supported_frame = self.ensure_supported_format(frame)
            if supported_frame is None:
                logger.warning("Failed to convert webcam frame to supported format")
                return []

            # Clean the frame
            cleaned_frame = self.clean_image(supported_frame)
            if cleaned_frame is None:
                logger.warning("Failed to clean webcam frame")
                return []

            # Detect faces in the frame using InsightFace
            faces = self.app.get(cleaned_frame)
            recognized_data = []

            for face in faces:
                if face.det_score < 0.5:  # Minimum detection confidence
                    continue

                # Compare with known faces
                if not self.known_face_embeddings:
                    continue

                # Convert embeddings to numpy arrays
                known_embeddings = np.array(self.known_face_embeddings)
                current_embedding = np.array(face.embedding).reshape(1, -1)

                # Calculate cosine similarity
                similarity_scores = np.dot(known_embeddings, current_embedding.T).flatten()
                best_match_index = np.argmax(similarity_scores)
                best_match_score = similarity_scores[best_match_index]

                if best_match_score >= self.min_confidence:
                    recognized_data.append({
                        'id': self.known_face_ids[best_match_index],
                        'admission_number': self.known_admission_numbers[best_match_index],
                        'confidence': float(best_match_score),
                        'face_location': [
                            int(face.bbox[1]),  # top
                            int(face.bbox[2]),  # right
                            int(face.bbox[3]),  # bottom
                            int(face.bbox[0])  # left
                        ]
                    })
                    logger.info(f"Recognized {self.known_admission_numbers[best_match_index]} "
                              f"with confidence {best_match_score:.2f}")

            return recognized_data

        except Exception as e:
            logger.error(f"Error in process_webcam_frame: {str(e)}", exc_info=True)
            return []

    def recognize_face(self, frame):
        """Alias for process_webcam_frame for backward compatibility"""
        return self.process_webcam_frame(frame)

    def mark_attendance(self, recognized_data):
        """Mark attendance for recognized students"""
        from .models import Attendance
        from datetime import date

        if not recognized_data:
            return []

        today = date.today()
        marked = []

        for data in recognized_data:
            # Check if attendance already marked today
            existing = Attendance.objects.filter(
                student_id=data['id'],
                date=today
            ).exists()

            if not existing:
                try:
                    Attendance.objects.create(
                        student_id=data['id'],
                        date=today,
                        status='Present',
                        confidence=data['confidence']
                    )
                    marked.append(data['admission_number'])
                    logger.info(f"Marked attendance for {data['admission_number']}")
                except Exception as e:
                    logger.error(f"Error marking attendance: {str(e)}", exc_info=True)

        return marked