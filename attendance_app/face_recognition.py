import cv2
import numpy as np
import os
import logging
from django.conf import settings
from PIL import Image
from insightface.app import FaceAnalysis

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
        try:
            # Configure the model directory explicitly
            import insightface
            model_dir = os.path.join(settings.BASE_DIR, 'insightface_models')
            os.environ['INSIGHTFACE_HOME'] = model_dir

            # Create the directory if it doesn't exist
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            logger.info(f"InsightFace model directory: {model_dir}")

            # Check if model files exist
            required_model_files = self._check_model_files(model_dir)

            if not required_model_files:
                logger.warning("Required model files not found. Attempting to download models.")
                # Force download the models
                self._download_models(model_dir)

        except Exception as e:
            logger.error(f"Error setting up model directory: {str(e)}")

        model_priority = [
            {'name': 'buffalo_l', 'det_size': (640, 640)},
            {'name': 'buffalo_s', 'det_size': (640, 640)},
            {'name': 'buffalo_sc', 'det_size': (320, 320)},
            # Fallback to CPU version if others fail
            {'name': 'buffalo_l', 'det_size': (640, 640), 'ctx_id': -1},
        ]

        for model in model_priority:
            try:
                ctx_id = model.get('ctx_id', 0)  # Default to GPU 0, -1 for CPU
                logger.info(f"Attempting to initialize FaceAnalysis with model: {model['name']}, ctx_id: {ctx_id}")

                # Initialize with explicit model location
                self.app = FaceAnalysis(
                    name=model['name'],
                    root=os.environ.get('INSIGHTFACE_HOME')
                )

                # Tell InsightFace to download missing models
                self.app.prepare(ctx_id=ctx_id, det_size=model['det_size'])

                # Test with dummy image to verify working
                test_img = np.zeros((100, 100, 3), dtype=np.uint8)
                test_faces = self.app.get(test_img)
                logger.info(f"Successfully initialized with model: {model['name']}")
                return
            except Exception as e:
                logger.error(f"Failed to initialize with {model['name']}: {str(e)}", exc_info=True)
                self.app = None
                continue

        # Final fallback: Use OpenCV's Haar cascade for face detection if InsightFace fails
        try:
            logger.warning("All InsightFace models failed. Falling back to OpenCV Haar Cascade")
            cascade_path = os.path.join(cv2.__path__[0], 'data', 'haarcascade_frontalface_default.xml')

            if not os.path.exists(cascade_path):
                cascade_path = os.path.join(settings.BASE_DIR, 'static', 'haarcascade_frontalface_default.xml')

            if os.path.exists(cascade_path):
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                self.using_fallback = True
                logger.info("Successfully initialized OpenCV fallback")
                return
            else:
                logger.error(f"Haar cascade file not found at {cascade_path}")
                self.face_cascade = None
                self.using_fallback = False
        except Exception as e:
            logger.error(f"Error initializing OpenCV fallback: {str(e)}", exc_info=True)
            self.face_cascade = None
            self.using_fallback = False

        logger.error("All face detection initialization attempts failed")

    def _check_model_files(self, model_dir):
        """Check if the required model files exist"""
        # We need at least the detection model files
        det_model_folder = os.path.join(model_dir, 'models', 'buffalo_l')
        if not os.path.exists(det_model_folder):
            return False

        # Check for essential files like .param and .bin files
        param_files = [f for f in os.listdir(det_model_folder) if f.endswith('.param')]
        bin_files = [f for f in os.listdir(det_model_folder) if f.endswith('.bin')]

        return len(param_files) > 0 and len(bin_files) > 0

    def _download_models(self, model_dir):
        """Attempt to manually download the required models"""
        try:
            import subprocess

            # Create a simple Python script to force download the models
            script_path = os.path.join(settings.BASE_DIR, 'download_models.py')
            with open(script_path, 'w') as f:
                f.write("""
import os
import insightface
from insightface.app import FaceAnalysis
from insightface.utils import face_align

# Set model directory
os.environ['INSIGHTFACE_HOME'] = '{}'

# Initialize with download
app = FaceAnalysis(allow_download=True)
app.prepare(ctx_id=-1)  # Use CPU
print("Models downloaded successfully!")
                """.format(model_dir))

            # Run the script
            subprocess.run(['python', script_path], check=True)
            logger.info("Model download script executed successfully")

            # Remove the temporary script
            os.remove(script_path)

        except Exception as e:
            logger.error(f"Error downloading models: {str(e)}")

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
                logger.warning(f"Failed to convert {image_path} to supported format")
                return None

            # Clean the image
            cleaned_image = self.clean_image(supported_image)
            if cleaned_image is None:
                logger.warning(f"Image cleaning failed for {image_path}")
                return None

            return cleaned_image
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            return None

    def extract_opencv_face_embedding(self, face_img):
        """Extract a simple feature vector from face image as OpenCV fallback"""
        try:
            # Resize to standard size
            face_img_resized = cv2.resize(face_img, (100, 100))

            # Convert to grayscale
            gray = cv2.cvtColor(face_img_resized, cv2.COLOR_RGB2GRAY)

            # Apply histogram equalization
            equalized = cv2.equalizeHist(gray)

            # Create a simple feature vector (flatten pixel values)
            # In a real system you'd use a proper face recognition model here
            feature_vector = equalized.flatten().astype(np.float32)

            # Normalize the vector
            norm = np.linalg.norm(feature_vector)
            if norm > 0:
                feature_vector = feature_vector / norm

            return feature_vector
        except Exception as e:
            logger.error(f"Error extracting OpenCV face embedding: {str(e)}")
            return None

    def load_student_images(self):
        """Load and preprocess all student images"""
        from .models import Student

        if self.app is None and not hasattr(self, 'using_fallback'):
            logger.error("Cannot load student images - Face detection not initialized")
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
            logger.debug(f"Processing student {student.admission_number}, image path: {image_path}")

            if not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}")
                continue

            try:
                # Load and clean the image
                rgb_img = self.load_and_preprocess_image(image_path)
                if rgb_img is None:
                    logger.warning(f"Failed to load/process image for {student.admission_number}")
                    continue

                # Using InsightFace if available
                if self.app is not None:
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

                # OpenCV fallback
                elif hasattr(self, 'face_cascade') and self.face_cascade is not None:
                    # Convert to grayscale for face detection
                    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)

                    # Detect faces
                    faces = self.face_cascade.detectMultiScale(
                        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                    )

                    if len(faces) == 0:
                        logger.warning(f"No faces detected (OpenCV) for {student.admission_number}")
                        continue

                    # Use the first face
                    x, y, w, h = faces[0]
                    face_img = rgb_img[y:y + h, x:x + w]

                    # Get simple features
                    embedding = self.extract_opencv_face_embedding(face_img)
                    if embedding is None:
                        logger.warning(f"Failed to extract embedding for {student.admission_number}")
                        continue

                    self.known_face_embeddings.append(embedding)
                else:
                    logger.warning(f"No face detection method available for {student.admission_number}")
                    continue

                # Store student info regardless of detection method
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
        if self.app is not None:
            return len(self.known_face_embeddings) > 0
        elif hasattr(self, 'face_cascade') and self.face_cascade is not None:
            return len(self.known_face_embeddings) > 0
        return False

    def process_webcam_frame(self, frame):
        """Process a frame from webcam and compare with stored faces"""
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

            recognized_data = []

            # Using InsightFace if available
            if self.app is not None:
                # Detect faces in the frame using InsightFace
                faces = self.app.get(cleaned_frame)

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

            # OpenCV fallback
            elif hasattr(self, 'face_cascade') and self.face_cascade is not None:
                # Convert to grayscale for face detection
                gray = cv2.cvtColor(cleaned_frame, cv2.COLOR_RGB2GRAY)

                # Detect faces
                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )

                for (x, y, w, h) in faces:
                    # Get face region
                    face_img = cleaned_frame[y:y + h, x:x + w]

                    # Extract features
                    current_embedding = self.extract_opencv_face_embedding(face_img)
                    if current_embedding is None:
                        continue

                    # Compare with known faces
                    if not self.known_face_embeddings:
                        continue

                    # Calculate similarity
                    similarity_scores = []
                    for known_embedding in self.known_face_embeddings:
                        # Use normalized dot product (cosine similarity)
                        similarity = np.dot(known_embedding, current_embedding)
                        similarity_scores.append(similarity)

                    similarity_scores = np.array(similarity_scores)
                    best_match_index = np.argmax(similarity_scores)
                    best_match_score = similarity_scores[best_match_index]

                    # OpenCV fallback is less accurate, use a lower threshold
                    if best_match_score >= self.min_confidence * 0.8:
                        recognized_data.append({
                            'id': self.known_face_ids[best_match_index],
                            'admission_number': self.known_admission_numbers[best_match_index],
                            'confidence': float(best_match_score),
                            'face_location': [y, x + w, y + h, x]  # top, right, bottom, left
                        })
                        logger.info(f"Recognized {self.known_admission_numbers[best_match_index]} "
                                    f"with confidence {best_match_score:.2f} (OpenCV fallback)")

            return recognized_data

        except Exception as e:
            logger.error(f"Error in process_webcam_frame: {str(e)}", exc_info=True)
            return []

    def recognize_face(self, frame):
        """Alias for process_webcam_frame for backward compatibility"""
        return self.process_webcam_frame(frame)

    def mark_attendance(self, recognized_data, unit_id):
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
                unit_id=unit_id,
                date=today
            ).exists()

            if not existing:
                try:
                    Attendance.objects.create(
                        student_id=data['id'],
                        unit_id=unit_id,
                        is_present=True
                        # date and time are auto_now_add=True
                    )
                    marked.append(data['admission_number'])
                    logger.info(f"Marked attendance for {data['admission_number']} in unit {unit_id}")
                except Exception as e:
                    logger.error(f"Error marking attendance: {str(e)}", exc_info=True)

        return marked