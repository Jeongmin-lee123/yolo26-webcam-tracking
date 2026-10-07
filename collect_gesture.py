"""
커스텀 제스처 데이터 수집 (웹캠 → 손 랜드마크 → CSV 저장)
설치: pip install -U mediapipe opencv-python scikit-learn
실행: python collect_gesture.py --label 제스처이름 [--count 300]
      예) python collect_gesture.py --label ok
          python collect_gesture.py --label rock
          python collect_gesture.py --label none   ← 아무 동작 아닌 손 (오인식 줄이기용, 추천)
키:   SPACE    녹화 시작/일시정지 (녹화 중에는 매 프레임 저장)
      q / ESC  종료
저장: data/custom_gestures.csv 에 누적 저장 (같은 라벨로 여러 번 실행해도 이어서 쌓임)
"""
import argparse
import csv
import time

import cv2
import mediapipe as mp

from gesture_common import DATA_PATH, NUM_FEATURES, create_hand_landmarker, draw_hand, landmarks_to_features

CAMERA_INDEX = 0


def count_existing(label):
    if not DATA_PATH.exists():
        return 0
    with DATA_PATH.open(newline="", encoding="utf-8") as f:
        return sum(1 for row in csv.reader(f) if row and row[0] == label)


def main():
    parser = argparse.ArgumentParser(description="커스텀 제스처 데이터 수집")
    parser.add_argument("--label", required=True, help="제스처 이름 (영문 권장, 예: ok, rock, call)")
    parser.add_argument("--count", type=int, default=300, help="이번 실행에서 모을 샘플 수 (기본 300)")
    args = parser.parse_args()
    label = args.label.strip()
    if not label.isascii():
        print("※ 라벨은 영문/숫자를 권장합니다 (OpenCV 화면에 한글이 표시되지 않음)")

    DATA_PATH.parent.mkdir(exist_ok=True)
    is_new_file = not DATA_PATH.exists()
    existing = count_existing(label)
    print(f"[{label}] 기존 샘플 {existing}개, 이번에 {args.count}개 수집")

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"웹캠({CAMERA_INDEX})을 열 수 없습니다.")

    recording = False
    saved = 0
    start = time.monotonic()
    with DATA_PATH.open("a", newline="", encoding="utf-8") as f, create_hand_landmarker() as landmarker:
        writer = csv.writer(f)
        if is_new_file:
            writer.writerow(["label"] + [f"f{i}" for i in range(NUM_FEATURES)])

        while saved < args.count:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # 거울 모드
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, int((time.monotonic() - start) * 1000))

            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                handedness = result.handedness[0][0].category_name
                draw_hand(frame, landmarks)
                if recording:
                    features = landmarks_to_features(landmarks, handedness, w, h)
                    writer.writerow([label] + [f"{v:.5f}" for v in features])
                    saved += 1

            # 상태 표시
            status = "REC" if recording else "PAUSED (SPACE to record)"
            color = (0, 0, 255) if recording else (0, 200, 255)
            cv2.putText(frame, f"label: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"{saved}/{args.count}  (total {existing + saved})",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            if not result.hand_landmarks:
                cv2.putText(frame, "no hand", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            # 진행 막대
            cv2.rectangle(frame, (0, h - 10), (int(w * saved / args.count), h), (0, 0, 255), -1)

            cv2.imshow("Collect Gesture", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord(" "):
                recording = not recording

    cap.release()
    cv2.destroyAllWindows()
    print(f"[{label}] {saved}개 저장 → 총 {existing + saved}개 ({DATA_PATH})")


if __name__ == "__main__":
    main()
