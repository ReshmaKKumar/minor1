import json
import numpy as np
try:
    import face_recognition
    FACE_RECOG_AVAILABLE = True
except Exception:
    FACE_RECOG_AVAILABLE = False

# Optional FaceNet (facenet-pytorch) support
try:
    from facenet_pytorch import InceptionResnetV1
    from PIL import Image
    import torch
    import torchvision.transforms as T

    FACENET_AVAILABLE = True
    _facenet_model = None

    def _get_facenet_model():
        global _facenet_model
        if _facenet_model is None:
            # load pretrained weights (vggface2 is a good default)
            _facenet_model = InceptionResnetV1(pretrained='vggface2').eval()
        return _facenet_model

    _facenet_transform = T.Compose([
        T.Resize((160, 160)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    def facenet_embedding_from_rgb(numpy_rgb_image):
        """Convert an RGB numpy image (H,W,3) to a 512-d Facenet embedding (numpy array).

        Returns None on failure.
        """
        try:
            img = Image.fromarray(numpy_rgb_image.astype('uint8'), 'RGB')
            t = _facenet_transform(img).unsqueeze(0)  # 1x3x160x160
            model = _get_facenet_model()
            with torch.no_grad():
                emb = model(t)
            return emb[0].cpu().numpy()
        except Exception:
            return None
except Exception:
    FACENET_AVAILABLE = False


def load_student_encodings(db_session, Student):
    """Load encodings from DB into a list of (student_id, name, encoding)"""
    students = Student.query.all()
    encs = []
    for s in students:
        if s.face_encoding:
            try:
                data = json.loads(s.face_encoding)
                # Backwards compatibility:
                # - older entries store a list (face_recognition 128-d encoding)
                # - newer entries may store a dict {"face_recognition": [...], "facenet": [...]}.
                if isinstance(data, dict):
                    # prefer facenet encoding if present, else face_recognition
                    if FACENET_AVAILABLE and 'facenet' in data and data['facenet']:
                        encs.append((s.id, s.name, np.array(data['facenet'])))
                    elif 'face_recognition' in data and data['face_recognition']:
                        encs.append((s.id, s.name, np.array(data['face_recognition'])))
                    else:
                        # fall back to any list entries
                        for k, v in data.items():
                            try:
                                encs.append((s.id, s.name, np.array(v)))
                                break
                            except Exception:
                                continue
                elif isinstance(data, list):
                    for e in data:
                        encs.append((s.id, s.name, np.array(e)))
                else:
                    # unknown format - skip
                    continue
            except Exception:
                continue
    return encs


def recognize_face(face_rgb, known_encodings, threshold=0.6):
    """Recognize using face_recognition compare_faces if available.
    known_encodings: list of (student_id, name, encoding)
    Returns best match or None
    """
    # If there are no known encodings, nothing to do
    if not known_encodings:
        return None, None, None

    # Decide which embedding method to use based on known encoding dimensionality.
    # If any known encoding has length 512 and FaceNet is available, use FaceNet.
    # Otherwise fall back to face_recognition (128-d) if available.
    try:
        dims = [len(k[2]) for k in known_encodings if hasattr(k[2], '__len__')]
        use_facenet = False
        use_face_recog = False
        if dims:
            if 512 in dims and FACENET_AVAILABLE:
                use_facenet = True
            if 128 in dims and FACE_RECOG_AVAILABLE:
                use_face_recog = True
        else:
            # default to face_recognition if installed
            use_face_recog = FACE_RECOG_AVAILABLE

        # Try face_recognition matching first if available and compatible
        if use_face_recog:
            try:
                encs = face_recognition.face_encodings(face_rgb)
                if encs:
                    enc = encs[0]
                    dists = []
                    for sid, name, ke in known_encodings:
                        try:
                            if len(ke) != len(enc):
                                continue
                            d = float(np.linalg.norm(ke - enc))
                        except Exception:
                            d = 1e6
                        dists.append((d, sid, name))
                    if dists:
                        dists.sort()
                        best = dists[0]
                        if best[0] <= threshold:
                            return best[1], best[2], float(best[0])
            except Exception:
                pass

        # If FaceNet available and compatible known encodings, try it
        if use_facenet and FACENET_AVAILABLE:
            try:
                emb = facenet_embedding_from_rgb(face_rgb)
                if emb is None:
                    return None, None, None
                dists = []
                for sid, name, ke in known_encodings:
                    try:
                        if len(ke) != len(emb):
                            continue
                        d = float(np.linalg.norm(ke - emb))
                    except Exception:
                        d = 1e6
                    dists.append((d, sid, name))
                if dists:
                    dists.sort()
                    best = dists[0]
                    # default threshold for facenet (512-d) is typically ~1.0
                    facenet_threshold = 1.0
                    if best[0] <= facenet_threshold:
                        return best[1], best[2], float(best[0])
            except Exception:
                pass

    except Exception:
        return None, None, None

    return None, None, None
