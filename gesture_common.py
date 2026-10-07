"""
커스텀 제스처 수집/학습/추론에서 같이 쓰는 함수 모음
- 손 랜드마커 생성
- 랜드마크 → 학습용 특징(feature) 벡터 변환
- 손 그리기
"""
from pathlib import Path

import numpy as np
import cv2
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

HERE = Path(__file__).parent
HAND_MODEL_PATH = HERE / "hand_landmarker.task"
DATA_PATH = HERE / "data" / "custom_gestures.csv"
CLASSIFIER_PATH = HERE / "custom_gesture_model.joblib"

NUM_FEATURES = 21 * 3  # 21개 점 × (x, y, z)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]
FINGERTIPS = (4, 8, 12, 16, 20)


def create_hand_landmarker(num_hands=1, min_conf=0.5):
    if not HAND_MODEL_PATH.exists():
        raise FileNotFoundError(f"모델 파일이 없습니다: {HAND_MODEL_PATH}")
    options = vision.HandLandmarkerOptions(
        # 경로에 한글이 있으면 MediaPipe가 파일을 못 열어서 바이트로 직접 전달
        base_options=mp_python.BaseOptions(model_asset_buffer=HAND_MODEL_PATH.read_bytes()),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=num_hands,
        min_hand_detection_confidence=min_conf,
        min_hand_presence_confidence=min_conf,
        min_tracking_confidence=min_conf,
    )
    return vision.HandLandmarker.create_from_options(options)


def landmarks_to_features(landmarks, handedness, image_w, image_h):
    """
    손 21개 점을 위치·크기·좌우와 무관한 63차원 벡터로 변환
      1) 화면 비율 보정: x에 (가로/세로) 곱하기 → 손 모양이 찌그러지지 않게
      2) 손목(0번)을 원점으로 이동 → 화면 어디에 있든 같은 값
      3) 손목~가장 먼 점 거리로 나누기 → 카메라와의 거리(손 크기)와 무관
      4) 왼손이면 x를 뒤집기 → 오른손/왼손을 같은 제스처로 취급
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    pts[:, 0] *= image_w / image_h
    pts -= pts[0]
    scale = np.linalg.norm(pts[:, :2], axis=1).max()
    if scale > 0:
        pts /= scale
    if handedness == "Left":
        pts[:, 0] *= -1
    return pts.flatten()


def draw_hand(frame, landmarks):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (255, 255, 255), 2)
    for idx, p in enumerate(pts):
        color = (0, 0, 255) if idx in FINGERTIPS else (0, 255, 0)
        cv2.circle(frame, p, 5, color, -1)
    return pts
