import cv2
import numpy as np


def edge_density_score(gray):
    edges = cv2.Canny(gray, 100, 200)
    return float(np.sum(edges) / 255) / (gray.shape[0] * gray.shape[1])


def laplacian_var_score(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def histogram_contrast_score(gray):
    # normalized variance of histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist_norm = hist / (hist.sum() + 1e-9)
    mean = np.dot(np.arange(256), hist_norm)
    var = np.dot((np.arange(256) - mean) ** 2, hist_norm)
    return float(var)


def is_live(face_bgr, debug=False):
    """Lightweight liveness heuristic using texture and edge measures.

    Returns (bool, score) where higher score indicates more likely live.
    This is not foolproof — recommended to use as a filter, not final gate.
    """
    try:
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    except Exception:
        return False, 0.0

    ed = edge_density_score(gray)
    lv = laplacian_var_score(gray)
    hc = histogram_contrast_score(gray)

    # normalize scores to roughly comparable ranges (heuristic)
    # scale laplacian and histogram down
    lv_n = np.tanh(lv / 100.0)
    hc_n = np.tanh(hc / 1000.0)

    score = 0.5 * ed + 0.3 * lv_n + 0.2 * hc_n

    if debug:
        return score >= 0.18, score
    return float(score >= 0.18), float(score)
