"""
MediaPipe Face Landmarker 웹캠 실시간 얼굴 랜드마크(478점) + 표정(Blendshape) 인식
설치: pip install -U mediapipe opencv-python
모델: face_landmarker.task 를 이 파일과 같은 폴더에 둘 것
      (https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task)
실행: python webcam_face.py
키:   q / ESC  종료
      m        얼굴 메시(삼각망) 표시 on/off
      b        표정(Blendshape) 점수 패널 on/off
"""
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = Path(__file__).with_name("face_landmarker.task")
CAMERA_INDEX = 0       # 기본 웹캠
NUM_FACES = 1          # 최대 검출 얼굴 개수
MIN_CONF = 0.5         # 검출/추적 신뢰도 임계값
TOP_BLENDSHAPES = 8    # 패널에 보여줄 표정 개수 (점수 높은 순)
BLINK_THRESHOLD = 0.5  # eyeBlink 점수가 이보다 크면 눈 감음
MOUTH_THRESHOLD = 0.3  # jawOpen 점수가 이보다 크면 입 벌림

C = vision.FaceLandmarksConnections
# (연결 목록, BGR 색상, 두께)
CONTOURS = [
    (C.FACE_LANDMARKS_FACE_OVAL, (220, 220, 220), 1),
    (C.FACE_LANDMARKS_LEFT_EYEBROW, (0, 200, 255), 2),
    (C.FACE_LANDMARKS_RIGHT_EYEBROW, (0, 200, 255), 2),
    (C.FACE_LANDMARKS_LEFT_EYE, (0, 255, 0), 1),
    (C.FACE_LANDMARKS_RIGHT_EYE, (0, 255, 0), 1),
    (C.FACE_LANDMARKS_LEFT_IRIS, (255, 128, 0), 2),
    (C.FACE_LANDMARKS_RIGHT_IRIS, (255, 128, 0), 2),
    (C.FACE_LANDMARKS_LIPS, (0, 0, 255), 2),
]


def draw_face(frame, landmarks, show_mesh):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    if show_mesh:
        for c in C.FACE_LANDMARKS_TESSELATION:
            cv2.line(frame, pts[c.start], pts[c.end], (90, 90, 90), 1)
    for connections, color, thickness in CONTOURS:
        for c in connections:
            cv2.line(frame, pts[c.start], pts[c.end], color, thickness)
    return pts


def blendshape_scores(result, i):
    if not result.face_blendshapes:
        return {}
    return {b.category_name: b.score for b in result.face_blendshapes[i]}


def draw_status(frame, scores, pts):
    """눈 깜빡임 / 입 벌림 / 미소 상태를 얼굴 위에 표시"""
    states = []
    # 화면을 좌우 반전했으므로 모델의 Left/Right가 화면 기준 반대
    if scores.get("eyeBlinkLeft", 0) > BLINK_THRESHOLD and scores.get("eyeBlinkRight", 0) > BLINK_THRESHOLD:
        states.append("Blink")
    elif scores.get("eyeBlinkLeft", 0) > BLINK_THRESHOLD or scores.get("eyeBlinkRight", 0) > BLINK_THRESHOLD:
        states.append("Wink")
    if scores.get("jawOpen", 0) > MOUTH_THRESHOLD:
        states.append("Mouth Open")
    if (scores.get("mouthSmileLeft", 0) + scores.get("mouthSmileRight", 0)) / 2 > 0.5:
        states.append("Smile")

    x0 = min(p[0] for p in pts)
    y0 = min(p[1] for p in pts)
    text = " | ".join(states) if states else "-"
    cv2.putText(frame, text, (x0, max(y0 - 10, 60)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)


def draw_blendshape_panel(frame, scores):
    """점수 높은 표정 TOP N을 막대그래프로 표시"""
    top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top = [kv for kv in top if kv[0] != "_neutral"][:TOP_BLENDSHAPES]
    x, y = frame.shape[1] - 290, 50
    cv2.rectangle(frame, (x - 10, y - 25), (frame.shape[1] - 5, y + 22 * len(top)), (0, 0, 0), -1)
    for name, score in top:
        cv2.putText(frame, name, (x, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.rectangle(frame, (x + 150, y - 8), (x + 150 + int(score * 120), y + 6), (0, 200, 255), -1)
        y += 22


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"모델 파일이 없습니다: {MODEL_PATH}")

    options = vision.FaceLandmarkerOptions(
        # 경로에 한글이 있으면 MediaPipe가 파일을 못 열어서 바이트로 직접 전달
        base_options=mp_python.BaseOptions(model_asset_buffer=MODEL_PATH.read_bytes()),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=NUM_FACES,
        min_face_detection_confidence=MIN_CONF,
        min_face_presence_confidence=MIN_CONF,
        min_tracking_confidence=MIN_CONF,
        output_face_blendshapes=True,  # 표정 점수 52종 출력
    )

    # Windows에서는 CAP_DSHOW가 웹캠 여는 속도가 빠름
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"웹캠({CAMERA_INDEX})을 열 수 없습니다.")

    show_mesh = False
    show_panel = True
    start = time.monotonic()
    prev = start
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # 거울 모드

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            # VIDEO 모드는 단조 증가하는 타임스탬프(ms)가 필요
            timestamp_ms = int((time.monotonic() - start) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            for i, landmarks in enumerate(result.face_landmarks):
                pts = draw_face(frame, landmarks, show_mesh)
                scores = blendshape_scores(result, i)
                draw_status(frame, scores, pts)
                if show_panel and i == 0:
                    draw_blendshape_panel(frame, scores)

            now = time.monotonic()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now
            cv2.putText(frame, f"FPS {fps:.1f}  faces {len(result.face_landmarks)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("MediaPipe Face Landmarker", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("m"):
                show_mesh = not show_mesh
            elif key == ord("b"):
                show_panel = not show_panel

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
