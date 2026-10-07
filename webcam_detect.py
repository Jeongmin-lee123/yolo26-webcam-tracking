"""
YOLO26 웹캠 실시간 Segmentation + Object Tracking
설치: pip install -U ultralytics opencv-python lap
실행: python webcam_detect.py
키:   q / ESC  종료
      m        마스크 표시 on/off
      t        이동 궤적 표시 on/off
"""
import time
from collections import defaultdict, deque

import cv2
from ultralytics import YOLO

MODEL_PATH = "yolo26n-seg.pt"  # segmentation 모델 (n/s/m/l/x, 처음 실행 시 자동 다운로드)
TRACKER = "bytetrack.yaml"     # "bytetrack.yaml" 또는 "botsort.yaml"
CAMERA_INDEX = 0               # 기본 웹캠
CONF_THRESHOLD = 0.5           # 신뢰도 임계값
IMG_SIZE = 640
TRAIL_LENGTH = 30              # 궤적에 남길 프레임 수


def main():
    model = YOLO(MODEL_PATH)

    # Windows에서는 CAP_DSHOW가 웹캠 여는 속도가 빠름
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"웹캠({CAMERA_INDEX})을 열 수 없습니다.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    show_masks = True
    show_trails = True
    trails = defaultdict(lambda: deque(maxlen=TRAIL_LENGTH))  # track_id -> 중심점 목록
    seen_ids = set()

    prev_time = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("프레임을 읽지 못했습니다.")
                break

            # persist=True: 프레임 간 트랙 ID 유지
            results = model.track(frame, persist=True, tracker=TRACKER,
                                  conf=CONF_THRESHOLD, imgsz=IMG_SIZE, verbose=False)
            result = results[0]

            # 박스/라벨/트랙 ID/마스크가 그려진 프레임
            annotated = result.plot(masks=show_masks)

            boxes = result.boxes
            if boxes.id is not None:
                active_ids = set()
                for box, track_id in zip(boxes, boxes.id.int().tolist()):
                    active_ids.add(track_id)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    trails[track_id].append(((x1 + x2) // 2, (y1 + y2) // 2))

                    # 새로 등장한 객체만 콘솔에 출력
                    if track_id not in seen_ids:
                        seen_ids.add(track_id)
                        cls_name = model.names[int(box.cls[0])]
                        print(f"[NEW] id={track_id:<4d} {cls_name:>12s} {float(box.conf[0]):.2f}")

                # 사라진 트랙의 궤적 정리
                for tid in list(trails):
                    if tid not in active_ids:
                        del trails[tid]
            else:
                trails.clear()

            if show_trails:
                for points in trails.values():
                    for i in range(1, len(points)):
                        thickness = max(1, int(4 * i / len(points)))
                        cv2.line(annotated, points[i - 1], points[i], (0, 255, 255), thickness)

            # FPS 표시
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            cv2.putText(annotated, f"FPS: {fps:.1f}  Tracks: {len(trails)}  Total IDs: {len(seen_ids)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(annotated, f"[m] mask: {'ON' if show_masks else 'OFF'}  [t] trail: {'ON' if show_trails else 'OFF'}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("YOLO26 Segmentation + Tracking", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("m"):
                show_masks = not show_masks
            elif key == ord("t"):
                show_trails = not show_trails
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
