"""
MediaPipe Gesture Recognizer 웹캠 실시간 손 제스처 인식
설치: pip install -U mediapipe opencv-python
모델: gesture_recognizer.task 를 이 파일과 같은 폴더에 둘 것
      (https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task)
실행: python webcam_gesture.py
키:   q / ESC  종료
인식 제스처: Closed_Fist, Open_Palm, Pointing_Up, Thumb_Down, Thumb_Up, Victory, ILoveYou
"""
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = Path(__file__).with_name("gesture_recognizer.task")
CAMERA_INDEX = 0       # 기본 웹캠
NUM_HANDS = 2          # 최대 검출 손 개수
MIN_CONF = 0.5         # 검출/추적 신뢰도 임계값
MIN_GESTURE_SCORE = 0.5  # 이보다 낮으면 제스처 표시 안 함

# 제스처 이름 → 화면 표시 텍스트 (OpenCV는 한글/이모지 출력 불가라 영문)
GESTURE_TEXT = {
    "Closed_Fist": "Fist",
    "Open_Palm": "Open Palm",
    "Pointing_Up": "Pointing Up",
    "Thumb_Down": "Thumbs Down",
    "Thumb_Up": "Thumbs Up",
    "Victory": "Victory",
    "ILoveYou": "I Love You",
}

# 21개 랜드마크 연결 (손목-엄지-검지-중지-약지-새끼)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]
FINGERTIPS = (4, 8, 12, 16, 20)


def draw_result(frame, result):
    h, w = frame.shape[:2]
    for i, landmarks in enumerate(result.hand_landmarks):
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (255, 255, 255), 2)
        for idx, p in enumerate(pts):
            color = (0, 0, 255) if idx in FINGERTIPS else (0, 255, 0)
            cv2.circle(frame, p, 5, color, -1)

        hand = result.handedness[i][0].category_name
        gesture = result.gestures[i][0] if result.gestures and result.gestures[i] else None
        if gesture and gesture.category_name in GESTURE_TEXT and gesture.score >= MIN_GESTURE_SCORE:
            label = f"{hand}: {GESTURE_TEXT[gesture.category_name]} {gesture.score:.2f}"
            color = (0, 255, 255)
        else:
            label = f"{hand}: -"
            color = (180, 180, 180)

        x0 = min(p[0] for p in pts)
        y0 = min(p[1] for p in pts)
        cv2.putText(frame, label, (x0, max(y0 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"모델 파일이 없습니다: {MODEL_PATH}")

    options = vision.GestureRecognizerOptions(
        # 경로에 한글이 있으면 MediaPipe가 파일을 못 열어서 바이트로 직접 전달
        base_options=mp_python.BaseOptions(model_asset_buffer=MODEL_PATH.read_bytes()),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=NUM_HANDS,
        min_hand_detection_confidence=MIN_CONF,
        min_hand_presence_confidence=MIN_CONF,
        min_tracking_confidence=MIN_CONF,
    )

    # Windows에서는 CAP_DSHOW가 웹캠 여는 속도가 빠름
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"웹캠({CAMERA_INDEX})을 열 수 없습니다.")

    start = time.monotonic()
    prev = start
    with vision.GestureRecognizer.create_from_options(options) as recognizer:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # 거울 모드

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            # VIDEO 모드는 단조 증가하는 타임스탬프(ms)가 필요
            timestamp_ms = int((time.monotonic() - start) * 1000)
            result = recognizer.recognize_for_video(mp_image, timestamp_ms)

            draw_result(frame, result)

            now = time.monotonic()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now
            cv2.putText(frame, f"FPS {fps:.1f}  hands {len(result.hand_landmarks)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("MediaPipe Gesture Recognizer", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
