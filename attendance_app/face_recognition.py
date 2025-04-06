import face_recognition
import numpy as np
import cv2
from django.conf import settings
import os
from PIL import Image


class FaceRecognition:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_ids = []
        self.known_admission_numbers = []
        self.min_confidence = 0.6  # Minimum confidence threshold for recognition

    def clean_image(self, image):
        """Apply preprocessing to clean and standardize the image"""
        try:
            # If image is already RGB (3 channels)
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Check if it's actually BGR (OpenCV default)
                if image.dtype == np.uint8:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return image

            # Convert grayscale to RGB
            if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            # Convert RGBA to RGB
            if len(image.shape) == 3 and image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

            # Ensure uint8 dtype
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)

            # Normalize lighting
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            image = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

            return image
        except Exception as e:
            print(f"Error in clean_image: {str(e)}")
            return image

    def load_and_preprocess_image(self, image_path):
        """Load and preprocess an image from file"""
        try:
            # Try with OpenCV first
            image = cv2.imread(image_path)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                print(f"Loaded with OpenCV. Shape: {image.shape}, dtype: {image.dtype}")
            else:
                # Fallback to PIL if OpenCV fails
                with Image.open(image_path) as pil_img:
                    if pil_img.mode != 'RGB':
                        pil_img = pil_img.convert('RGB')
                    image = np.array(pil_img)
                    print(f"Loaded with PIL. Shape: {image.shape}, dtype: {image.dtype}")

            # Clean the image
            cleaned_image = self.clean_image(image)
            if cleaned_image is None:
                print("Image cleaning failed")
                return None

            print(f"After cleaning. Shape: {cleaned_image.shape}, dtype: {cleaned_image.dtype}")
            return cleaned_image
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            return None

    def load_student_images(self):
        """Load and preprocess all student images"""
        from .models import Student
        students = Student.objects.all()

        print(f"Loading {len(students)} student images...")

        # Clear existing data
        self.known_face_encodings = []
        self.known_face_ids = []
        self.known_admission_numbers = []

        for student in students:
            if not student.photo:
                print(f"No photo for student {student.admission_number}")
                continue

            # FIXED: Proper path handling with normpath
            image_path = os.path.normpath(os.path.join(settings.MEDIA_ROOT, str(student.photo)))
            print(f"\nProcessing student {student.admission_number}")
            print(f"Image path: {image_path}")

            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                continue

            try:
                # Load and clean the image
                rgb_img = self.load_and_preprocess_image(image_path)
                if rgb_img is None:
                    print(f"Failed to load/process image for {student.admission_number}")
                    continue

                # Debug: Print image info before face detection
                print(
                    f"Debug: Image before face detection - shape={rgb_img.shape}, dtype={rgb_img.dtype}, min={rgb_img.min()}, max={rgb_img.max()}")

                # Debug: Save the cleaned image for inspection
                debug_dir = os.path.join(settings.MEDIA_ROOT, "debug")
                os.makedirs(debug_dir, exist_ok=True)
                debug_filename = f"cleaned_{student.admission_number.replace('/', '_')}.jpg"
                debug_path = os.path.join(debug_dir, debug_filename)
                cv2.imwrite(debug_path, cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
                print(f"Saved cleaned image to {debug_path}")

                # Try multiple face detection methods
                face_locations = self.detect_faces(rgb_img)
                if not face_locations:
                    print(f"No faces detected for {student.admission_number}")
                    continue

                # Get encodings for all faces found
                face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
                if face_encodings:
                    for encoding in face_encodings:
                        self.known_face_encodings.append(encoding)
                        self.known_face_ids.append(student.id)
                        self.known_admission_numbers.append(student.admission_number)
                    print(f"✓ Processed student {student.admission_number}")
                else:
                    print(f"No face encodings generated for {student.admission_number}")

            except Exception as e:
                print(f"Error processing {student.admission_number}: {str(e)}")
                import traceback
                traceback.print_exc()

        print(f"\nLoaded {len(self.known_face_encodings)} face encodings from {len(students)} students")

    def detect_faces(self, image):
        """Detect faces using multiple methods with fallbacks"""
        if image is None:
            print("No image provided for face detection")
            return []

        print(f"Detecting faces in image with shape {image.shape}, dtype {image.dtype}")

        # Make a copy of the image to avoid modifying the original
        img = image.copy()

        # Ensure proper format - face_recognition needs RGB uint8
        # First convert to uint8 if needed
        if img.dtype != np.uint8:
            print(f"Converting image from {img.dtype} to uint8")
            img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)

        # Then handle channel conversion
        if len(img.shape) == 2:  # Grayscale
            print("Converting grayscale to RGB")
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif len(img.shape) == 3:
            if img.shape[2] == 4:  # RGBA
                print("Converting RGBA to RGB")
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            elif img.shape[2] == 3:  # Could be BGR or RGB
                # OpenCV reads images as BGR, convert to RGB if from OpenCV
                if 'cv2' in str(img.__class__):
                    print("Converting BGR to RGB")
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Verify image format is correct
        if len(img.shape) != 3 or img.shape[2] != 3 or img.dtype != np.uint8:
            print(f"Image format is still incorrect: shape={img.shape}, dtype={img.dtype}")
            return []

        face_locations = []

        # Method 1: HOG (fastest)
        try:
            print("Trying HOG face detection...")
            face_locations = face_recognition.face_locations(img, model="hog")
            if face_locations:
                print(f"HOG found {len(face_locations)} faces")
                return face_locations
        except Exception as e:
            print(f"HOG detection failed: {str(e)}")

        # Method 2: CNN (more accurate but slower)
        try:
            print("Trying CNN face detection...")
            face_locations = face_recognition.face_locations(img, model="cnn")
            if face_locations:
                print(f"CNN found {len(face_locations)} faces")
                return face_locations
        except Exception as e:
            print(f"CNN detection failed: {str(e)}")

        # Method 3: Try with resized image
        try:
            print("Trying resized image face detection...")
            small_img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
            face_locations = face_recognition.face_locations(small_img)
            if face_locations:
                print(f"Resized image found {len(face_locations)} faces")
                # Scale locations back to original size
                return [(top * 2, right * 2, bottom * 2, left * 2) for (top, right, bottom, left) in face_locations]
        except Exception as e:
            print(f"Resized detection failed: {str(e)}")

        print("No faces detected with any method")
        return []

    def recognize_face(self, frame):
        """Alias for process_webcam_frame for backward compatibility"""
        return self.process_webcam_frame(frame)

    def process_webcam_frame(self, frame):
        """Process a frame from webcam and compare with stored faces"""
        if len(self.known_face_encodings) == 0:
            print("Warning: No known faces loaded")
            return []

        try:
            # Clean and preprocess the webcam frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cleaned_frame = self.clean_image(rgb_frame)

            if cleaned_frame is None:
                print("Failed to clean webcam frame")
                return []

            # Detect faces in the frame
            face_locations = self.detect_faces(cleaned_frame)
            if not face_locations:
                print("No faces detected in webcam frame")
                return []

            # Get encodings for detected faces
            face_encodings = face_recognition.face_encodings(cleaned_frame, face_locations)
            recognized_data = []

            for i, face_encoding in enumerate(face_encodings):
                # Compare with known faces
                face_distances = face_recognition.face_distance(
                    self.known_face_encodings,
                    face_encoding
                )

                if len(face_distances) == 0:
                    continue

                best_match_index = np.argmin(face_distances)
                best_match_distance = face_distances[best_match_index]
                confidence = 1 - best_match_distance

                if confidence >= self.min_confidence:
                    recognized_data.append({
                        'id': self.known_face_ids[best_match_index],
                        'admission_number': self.known_admission_numbers[best_match_index],
                        'confidence': float(confidence),
                        'face_location': face_locations[i]
                    })
                    print(
                        f"Recognized {self.known_admission_numbers[best_match_index]} with confidence {confidence:.2f}")

            return recognized_data

        except Exception as e:
            print(f"Error in process_webcam_frame: {str(e)}")
            return []

    def mark_attendance(self, recognized_data):
        """Mark attendance for recognized students"""
        from .models import Attendance
        from datetime import date

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
                    print(f"Marked attendance for {data['admission_number']}")
                except Exception as e:
                    print(f"Error marking attendance: {str(e)}")

        return marked