"""
학습한 커스텀 제스처로 웹캠 실시간 추론 + 인식된 제스처에 맞는 이모지 표시
설치: pip install -U mediapipe opencv-python scikit-learn pillow
실행: python webcam_custom_gesture.py
      (train_gesture.py 로 custom_gesture_model.joblib 을 먼저 만들어야 함)
키:   q / ESC  종료
"""
import time
from collections import Counter, deque

import cv2
import joblib
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gesture_common import CLASSIFIER_PATH, create_hand_landmarker, draw_hand, landmarks_to_features

CAMERA_INDEX = 0
NUM_HANDS = 2
MIN_PROB = 0.7      # 이보다 확률이 낮으면 "?" 로 표시
SMOOTH_FRAMES = 5   # 최근 N프레임 다수결로 결과 흔들림 줄이기

# 라벨 → 표시할 이모지 (라벨을 추가하면 여기에도 추가)
EMOJI = {
    "korean_heart": "\u2764",      # ❤
    "ok": "\U0001F44C",            # 👌
}
EMOJI_FONT = "C:/Windows/Fonts/seguiemj.ttf"  # Windows 컬러 이모지 폰트
EMOJI_SIZE = 96


def render_emoji(char, size=EMOJI_SIZE):
    """OpenCV는 이모지를 못 그려서 Pillow로 투명 배경 BGRA 이미지를 만들어 둔다"""
    font = ImageFont.truetype(EMOJI_FONT, size)
    l, t, r, b = font.getbbox(char)
    img = Image.new("RGBA", (r - l, b - t), (0, 0, 0, 0))
    ImageDraw.Draw(img).text((-l, -t), char, font=font, embedded_color=True)
    img = img.crop(img.getbbox())  # 투명 여백 제거
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)


def overlay(frame, img, cx, bottom):
    """BGRA 이미지를 (cx, bottom) 기준 가운데-아래 정렬로 알파 합성 (화면 밖은 잘라냄)"""
    h, w = img.shape[:2]
    x1, y1 = cx - w // 2, bottom - h
    fx1, fy1 = max(x1, 0), max(y1, 0)
    fx2, fy2 = min(x1 + w, frame.shape[1]), min(y1 + h, frame.shape[0])
    if fx1 >= fx2 or fy1 >= fy2:
        return
    crop = img[fy1 - y1:fy2 - y1, fx1 - x1:fx2 - x1]
    alpha = crop[:, :, 3:4].astype(np.float32) / 255
    roi = frame[fy1:fy2, fx1:fx2]
    roi[:] = (crop[:, :, :3] * alpha + roi * (1 - alpha)).astype(np.uint8)


def main():
    if not CLASSIFIER_PATH.exists():
        raise FileNotFoundError(f"학습된 모델이 없습니다: {CLASSIFIER_PATH}\n먼저 train_gesture.py 를 실행하세요.")
    bundle = joblib.load(CLASSIFIER_PATH)
    model, labels = bundle["model"], bundle["labels"]
    print("인식 가능한 제스처:", labels)
    emoji_images = {label: render_emoji(char) for label, char in EMOJI.items()}

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"웹캠({CAMERA_INDEX})을 열 수 없습니다.")

    # 손(Left/Right)별 최근 예측 기록
    history = {"Left": deque(maxlen=SMOOTH_FRAMES), "Right": deque(maxlen=SMOOTH_FRAMES)}
    start = time.monotonic()
    prev = start
    with create_hand_landmarker(num_hands=NUM_HANDS) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # 거울 모드
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, int((time.monotonic() - start) * 1000))

            for i, landmarks in enumerate(result.hand_landmarks):
                hand = result.handedness[i][0].category_name
                features = landmarks_to_features(landmarks, hand, w, h)
                probs = model.predict_proba([features])[0]
                best = probs.argmax()
                pred = labels[best] if probs[best] >= MIN_PROB else "?"
                history[hand].append(pred)
                smoothed = Counter(history[hand]).most_common(1)[0][0]

                pts = draw_hand(frame, landmarks)
                x0 = min(p[0] for p in pts)
                y0 = min(p[1] for p in pts)
                color = (180, 180, 180) if smoothed in ("?", "none") else (0, 255, 255)
                cv2.putText(frame, f"{hand}: {smoothed} {probs[best]:.2f}", (x0, max(y0 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                # 인식된 제스처의 이모지를 손 위쪽(라벨 글자 위)에 표시
                if smoothed in emoji_images:
                    cx = (min(p[0] for p in pts) + max(p[0] for p in pts)) // 2
                    overlay(frame, emoji_images[smoothed], cx, y0 - 35)

            now = time.monotonic()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now
            cv2.putText(frame, f"FPS {fps:.1f}  hands {len(result.hand_landmarks)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("Custom Gesture", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
