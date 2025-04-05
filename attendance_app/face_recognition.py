import face_recognition
import numpy as np
import cv2
from django.conf import settings
import os


class FaceRecognition:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_ids = []

    def load_student_images(self):
        from .models import Student
        students = Student.objects.all()
        for student in students:
            if student.photo:
                image_path = os.path.join(settings.MEDIA_ROOT, str(student.photo))
                try:
                    image = face_recognition.load_image_file(image_path)
                    face_encodings = face_recognition.face_encodings(image)
                    if face_encodings:
                        self.known_face_encodings.append(face_encodings[0])
                        self.known_face_ids.append(student.id)
                except Exception as e:
                    print(f"Error processing image for student {student.admission_number}: {e}")

    def recognize_face(self, frame):
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color
        rgb_small_frame = small_frame[:, :, ::-1]

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        recognized_ids = []

        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                recognized_ids.append(self.known_face_ids[best_match_index])

        return recognized_ids