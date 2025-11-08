import threading
import time
import cv2
import numpy as np

# Try to import Ultralytics YOLOv8 if available
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False
    
from anti_spoof import is_live
from recognition import load_student_encodings, recognize_face, FACE_RECOG_AVAILABLE

# We'll lazily populate known encodings when requested
_KNOWN_ENCODINGS = None


class CameraWorker:
    """Background camera capture worker with frame skipping and simple face drawing.

    - Reads frames from the camera in a background thread.
    - Processes every `process_every_n` frames (default 3) to reduce CPU load.
    - Stores the last processed JPEG bytes in `last_frame` for streaming.
    """
    def __init__(self, src=0, process_every_n=3):
        self.src = src
        self.process_every_n = max(1, int(process_every_n))
        self.cap = None
        self.thread = None
        self.running = False
        self.lock = threading.Lock()
        self.last_frame = None
        self._frame_count = 0
        # simple Haar cascade for face drawing fallback
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # optional YOLO model
        self.yolo = None
        if YOLO_AVAILABLE:
            try:
                # 'yolov8n' is small and faster; for faces a custom model would be ideal
                self.yolo = YOLO('yolov8n.pt')
            except Exception:
                # if model file not available or fails to load, disable YOLO
                self.yolo = None
        # recognition results
        self.last_recognitions = []  # list of dicts: {box, student_id, name, distance, live, score}
        # attendance callback and session control
        self.attendance_callback = None
        self.active_subject = None
        self.active_session = None
        self._last_notify = {}  # student_id -> timestamp of last notify to throttle

    def start(self):
        if self.running:
            return
        self.cap = cv2.VideoCapture(self.src)
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
        self.thread = None

    def _run(self):
        while self.running:
            try:
                if not self.cap or not self.cap.isOpened():
                    # try to open if closed
                    self.cap = cv2.VideoCapture(self.src)
                    time.sleep(0.1)

                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue

                self._frame_count += 1
                # Only process every Nth frame
                if (self._frame_count % self.process_every_n) == 0:
                    # Prefer YOLO detection if available, otherwise Haar cascade
                    try:
                        if self.yolo is not None:
                            # Ultralytics YOLO returns results with boxes in xyxy format
                            try:
                                results = self.yolo(frame, imgsz=640, conf=0.35)
                            except Exception:
                                results = self.yolo.predict(frame, imgsz=640, conf=0.35)
                            # results[0].boxes.xyxy may be a tensor or numpy array
                            if results and len(results):
                                res = results[0]
                                boxes = []
                                if hasattr(res, 'boxes') and res.boxes is not None:
                                    for b in res.boxes:
                                        try:
                                            xyxy = b.xyxy[0].cpu().numpy() if hasattr(b.xyxy[0], 'cpu') else np.array(b.xyxy[0])
                                        except Exception:
                                            try:
                                                xyxy = np.array(b.xyxy)
                                            except Exception:
                                                continue
                                        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                                        boxes.append((x1, y1, x2 - x1, y2 - y1))
                                    # draw boxes and collect crops for liveness/recognition
                                    crops = []
                                    for (x, y, w, h) in boxes:
                                        cv2.rectangle(frame, (x, y), (x + w, y + h), (14, 165, 233), 2)
                                        # extract crop with padding
                                        x1 = max(0, x - 10)
                                        y1 = max(0, y - 10)
                                        x2 = min(frame.shape[1], x + w + 10)
                                        y2 = min(frame.shape[0], y + h + 10)
                                        crop = frame[y1:y2, x1:x2].copy()
                                        crops.append(((x1, y1, x2, y2), crop))
                                    # run liveness + recognition on crops
                                    self.last_recognitions = []
                                    global _KNOWN_ENCODINGS
                                    if _KNOWN_ENCODINGS is None and FACE_RECOG_AVAILABLE:
                                        try:
                                            # import here to avoid circular imports; app will provide Student and db session
                                            from app import db, Student
                                            _KNOWN_ENCODINGS = load_student_encodings(db.session, Student)
                                        except Exception:
                                            _KNOWN_ENCODINGS = []

                                    for (box, crop) in crops:
                                        live, score = is_live(crop)
                                        recognized = (None, None, None)
                                        if live and FACE_RECOG_AVAILABLE and _KNOWN_ENCODINGS:
                                            try:
                                                # face_recognition expects RGB
                                                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                                                recognized = recognize_face(rgb, _KNOWN_ENCODINGS)
                                            except Exception:
                                                recognized = (None, None, None)

                                        # encode crop snapshot
                                        snap_bytes = None
                                        try:
                                            ok, jpg = cv2.imencode('.jpg', crop, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                                            if ok:
                                                snap_bytes = jpg.tobytes()
                                        except Exception:
                                            snap_bytes = None

                                        rec = {
                                            'box': box,
                                            'live': bool(live),
                                            'liveness_score': float(score),
                                            'student_id': recognized[0],
                                            'name': recognized[1],
                                            'distance': recognized[2],
                                            'snapshot': snap_bytes
                                        }
                                        self.last_recognitions.append(rec)

                                        # Notify attendance callback if configured and subject active
                                        try:
                                            sid = rec.get('student_id')
                                            nowt = time.time()
                                            notify = False
                                            if self.attendance_callback and self.active_subject and sid:
                                                last_t = self._last_notify.get(sid)
                                                # throttle notifications per student to at most once per 60 seconds
                                                if not last_t or (nowt - last_t) > 60:
                                                    notify = True
                                                    self._last_notify[sid] = nowt
                                            if notify:
                                                try:
                                                    self.attendance_callback(rec, self.active_subject, self.active_session)
                                                except Exception:
                                                    pass
                                        except Exception:
                                            pass
                            else:
                                # no detections
                                pass
                        else:
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                            for (x, y, w, h) in faces:
                                cv2.rectangle(frame, (x, y), (x + w, y + h), (14, 165, 233), 2)
                    except Exception:
                        # if processing fails, continue silently
                        pass

                    # encode to JPEG and store
                    try:
                        ret2, jpg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                        if ret2:
                            with self.lock:
                                self.last_frame = jpg.tobytes()
                    except Exception:
                        # ignore encoding errors
                        pass

                # small sleep to avoid tight loop
                time.sleep(0.01)
            except Exception:
                time.sleep(0.05)

    def get_frame(self):
        with self.lock:
            return self.last_frame


# module-level singleton worker (import and use this from app)
cam_worker = CameraWorker(process_every_n=3)

if __name__ == '__main__':
    cam_worker.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        cam_worker.stop()
