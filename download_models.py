
import os
import insightface
from insightface.app import FaceAnalysis
from insightface.utils import face_align

# Set model directory
os.environ['INSIGHTFACE_HOME'] = 'C:\Users\kiama\PycharmProjects\attendance_system\insightface_models'

# Initialize with download
app = FaceAnalysis(allow_download=True)
app.prepare(ctx_id=-1)  # Use CPU
print("Models downloaded successfully!")
                