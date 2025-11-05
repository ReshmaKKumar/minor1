import os
import io
import time
import threading
import base64
import traceback
from datetime import datetime, date
from functools import wraps

from flask import (
    Flask, render_template, request, redirect, url_for, flash, jsonify,
    Response, session
)
from bson.objectid import ObjectId
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
import cv2
import csv

# --- MongoDB Connection Setup ---
from pymongo import MongoClient
import certifi

# --- Flask Config ---
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'super-secret-key')

# ✅ MongoDB Atlas Secure Connection
MONGO_URI = os.environ.get(
    "MONGO_URI",
    "mongodb+srv://dbflaskuser:flask123@cluster0.tvpcrxu.mongodb.net/?retryWrites=true&w=majority"
)
try:
    client = MongoClient(MONGO_URI, tls=True, tlsCAFile=certifi.where())
    client.admin.command("ping")
    print("✅ MongoDB Atlas connected successfully!")
except Exception as e:
    print("❌ MongoDB connection failed:", e)

db = client["attendanceDB"]
users_col = db.users
students_col = db.students
subjects_col = db.subjects
attendance_col = db.attendance
sessions_col = db.sessions

# --- YOLO Setup ---
YOLO_CFG = os.environ.get('YOLO_CFG', 'models/yolo.cfg')
YOLO_WEIGHTS = os.environ.get('YOLO_WEIGHTS', 'models/yolo.weights')
YOLO_NAMES = os.environ.get('YOLO_NAMES', 'models/coco.names')
YOLO_CONFIDENCE = float(os.environ.get('YOLO_CONF', 0.5))
YOLO_NMS_THRESH = float(os.environ.get('YOLO_NMS', 0.4))

yolo_net = None
yolo_output_layers = []
yolo_labels = []
if os.path.exists(YOLO_CFG) and os.path.exists(YOLO_WEIGHTS):
    try:
        yolo_net = cv2.dnn.readNetFromDarknet(YOLO_CFG, YOLO_WEIGHTS)
        yolo_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        yolo_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        if os.path.exists(YOLO_NAMES):
            with open(YOLO_NAMES, 'r') as f:
                yolo_labels = [l.strip() for l in f.readlines() if l.strip()]
        layer_names = yolo_net.getLayerNames()
        yolo_output_layers = [layer_names[i[0] - 1] for i in yolo_net.getUnconnectedOutLayers()]
        print("✅ YOLO loaded successfully!")
    except Exception as e:
        print("❌ Failed to load YOLO:", e)
        yolo_net = None
else:
    print("⚠️ YOLO config/weights not found at configured paths. YOLO disabled.")

# --- FaceNet Loader ---
USE_FACENET = False
FACENET_AVAILABLE = False
FACERECOG_AVAILABLE = False

try:
    from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
    import torch
    from torchvision import transforms
    FACENET_AVAILABLE = True
    USE_FACENET = True
    print("✅ facenet-pytorch module found.")
except Exception:
    try:
        import face_recognition
        FACERECOG_AVAILABLE = True
        print("✅ face_recognition fallback enabled.")
    except Exception:
        print("⚠️ No FaceNet or face_recognition available. Embedding will not work.")

facenet_model = None
preprocess_transform = None
if FACENET_AVAILABLE:
    try:
        facenet_model = InceptionResnetV1(pretrained='vggface2').eval()
        preprocess_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            fixed_image_standardization
        ])
        print("✅ FaceNet (facenet-pytorch) model loaded successfully!")
    except Exception as e:
        print("❌ Could not load facenet model:", e)
        FACENET_AVAILABLE = False
        USE_FACENET = False

# --- Utility functions ---
def login_required_teacher(fn):
    @wraps(fn)
    def wrapper(*a, **kw):
        if 'teacher_id' not in session:
            flash('Please login as teacher to access that page.')
            return redirect(url_for('login'))
        return fn(*a, **kw)
    return wrapper

def login_required_student(fn):
    @wraps(fn)
    def wrapper(*a, **kw):
        if 'student_id' not in session:
            flash('Please login as student to access that page.')
            return redirect(url_for('student_login'))
        return fn(*a, **kw)
    return wrapper

def compute_embedding_facenet(face_bgr):
    if not FACENET_AVAILABLE or facenet_model is None:
        raise RuntimeError("FaceNet not available")
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    tensor = preprocess_transform(face_rgb).unsqueeze(0)
    with torch.no_grad():
        emb = facenet_model(tensor)
    return emb[0].cpu().numpy().tolist()

def compute_embedding_facerec(face_bgr):
    if not FACERECOG_AVAILABLE:
        raise RuntimeError("face_recognition not available")
    import face_recognition
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    encs = face_recognition.face_encodings(rgb)
    return encs[0].tolist() if encs else None

def compute_embedding(face_bgr):
    if FACENET_AVAILABLE:
        return compute_embedding_facenet(face_bgr)
    elif FACERECOG_AVAILABLE:
        return compute_embedding_facerec(face_bgr)
    else:
        raise RuntimeError("No embedding backend available")

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

MATCH_THRESHOLD = float(os.environ.get('MATCH_THRESHOLD', 0.6))

# --- Camera Worker ---
class CameraWorker(threading.Thread):
    def __init__(self, source=0):
        super().__init__(daemon=True)
        self.source = source
        self.running = False
        self.cap = None
        self.active_session = None
        self.active_subject_id = None
        self.recognitions = []
        self.lock = threading.Lock()

    def run(self):
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            print("⚠️ Camera not available.")
            return
        self.running = True
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.1)
                    continue
                boxes = self._detect_with_yolo(frame) if yolo_net else self._detect_with_haar(frame)
                recs = []
                for (x1, y1, x2, y2, conf) in boxes:
                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop.size == 0:
                        continue
                    emb = None
                    try:
                        emb = compute_embedding(face_crop)
                    except Exception:
                        pass
                    recs.append({
                        'name': 'Unknown' if not emb else 'Face Detected',
                        'box': [x1, y1, x2, y2],
                        'timestamp': datetime.utcnow().isoformat()
                    })
                with self.lock:
                    self.recognitions = recs
                time.sleep(0.05)
        finally:
            if self.cap:
                self.cap.release()
            self.running = False

    def _detect_with_haar(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        return [(x, y, x + w, y + h, 1.0) for (x, y, w, h) in faces]

    def _detect_with_yolo(self, frame):
        (H, W) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        yolo_net.setInput(blob)
        outputs = yolo_net.forward(yolo_output_layers)
        boxes = []
        for out in outputs:
            for detection in out:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = float(scores[classID])
                if confidence > YOLO_CONFIDENCE:
                    center_x, center_y, w, h = (
                        int(detection[0] * W), int(detection[1] * H),
                        int(detection[2] * W), int(detection[3] * H)
                    )
                    x1, y1 = int(center_x - w/2), int(center_y - h/2)
                    boxes.append((x1, y1, x1 + w, y1 + h, confidence))
        return boxes

    def stop(self):
        self.running = False

    def get_recognitions(self):
        with self.lock:
            return list(self.recognitions)


cam_worker = CameraWorker(0)

# --- Routes ---
@app.route('/')
def index():
    if 'teacher_id' in session:
        return redirect(url_for('dashboard'))
    if 'student_id' in session:
        return redirect(url_for('student_dashboard'))
    return render_template('choose.html')


# --- Startup Health Summary ---
print("\n================= SYSTEM CHECK =================")
print("MongoDB:", "✅ Connected" if client else "❌ Not connected")
print("YOLO:", "✅ Loaded" if yolo_net else "⚠️ Not found")
print("FaceNet:", "✅ Loaded" if FACENET_AVAILABLE else ("✅ face_recognition fallback" if FACERECOG_AVAILABLE else "❌ None"))
print("================================================\n")


# --- Run Flask App ---
if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    admin = users_col.find_one({'username': 'admin'})
    if not admin:
        users_col.insert_one({
            'username': 'admin',
            'email': 'admin@example.com',
            'password_hash': generate_password_hash('admin123'),
            'created_at': datetime.utcnow()
        })
        print("👤 Default admin created (admin / admin123)")
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
