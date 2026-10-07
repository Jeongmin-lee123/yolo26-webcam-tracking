"""
학습한 커스텀 제스처 모델(custom_gesture_model.joblib)을 웹 데모용 JSON으로 변환
실행: python export_web_model.py
결과: docs/custom_gesture_model.json  (docs/app.js 가 브라우저에서 직접 계산)
      train_gesture.py 로 다시 학습했다면 이 스크립트도 다시 실행할 것
"""
import json

import joblib

from gesture_common import CLASSIFIER_PATH, HERE

OUT_PATH = HERE / "docs" / "custom_gesture_model.json"
ROUND = 6  # 소수점 자릿수 (파일 크기 줄이기)


def to_list(arr):
    return [[round(float(v), ROUND) for v in row] for row in arr] if arr.ndim == 2 \
        else [round(float(v), ROUND) for v in arr]


def main():
    bundle = joblib.load(CLASSIFIER_PATH)
    scaler, mlp = bundle["model"][0], bundle["model"][-1]
    if mlp.activation != "relu" or mlp.out_activation_ != "softmax":
        raise ValueError("relu + softmax MLP 만 지원합니다.")

    data = {
        "labels": [str(l) for l in bundle["labels"]],
        "mean": to_list(scaler.mean_),
        "scale": to_list(scaler.scale_),
        # layers[i] = { W: [입력][출력], b: [출력] }
        "layers": [{"W": to_list(W), "b": to_list(b)} for W, b in zip(mlp.coefs_, mlp.intercepts_)],
    }
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    print(f"저장 완료: {OUT_PATH} ({OUT_PATH.stat().st_size / 1024:.1f} KB), 라벨: {data['labels']}")


if __name__ == "__main__":
    main()
