"""
커스텀 제스처 분류기 학습 (data/custom_gestures.csv → custom_gesture_model.joblib)
설치: pip install -U scikit-learn
실행: python train_gesture.py
      (collect_gesture.py 로 2개 이상의 라벨을 먼저 수집해야 함)
"""
import csv
from collections import Counter

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gesture_common import CLASSIFIER_PATH, DATA_PATH

TEST_RATIO = 0.2   # 검증용으로 떼어둘 비율
RANDOM_SEED = 42


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"데이터가 없습니다: {DATA_PATH}\n먼저 collect_gesture.py 로 수집하세요.")
    labels, features = [], []
    with DATA_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # 헤더
        for row in reader:
            if row:
                labels.append(row[0])
                features.append([float(v) for v in row[1:]])
    return np.array(features, dtype=np.float32), np.array(labels)


def build_model():
    # 표준화 → 작은 신경망(MLP). 63차원 입력이라 CPU로 몇 초면 학습됨
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000,
                      early_stopping=True, random_state=RANDOM_SEED),
    )


def main():
    X, y = load_data()
    counts = Counter(y)
    print("=== 데이터 ===")
    for label, n in sorted(counts.items()):
        print(f"  {label:<15} {n}개")
    if len(counts) < 2:
        raise ValueError("라벨이 2개 이상 있어야 학습할 수 있습니다.")
    if min(counts.values()) < 20:
        print("※ 샘플이 적은 라벨이 있습니다. 라벨당 200개 이상을 권장합니다.")

    # 1) 일부를 떼어 성능 검증
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_RATIO, stratify=y, random_state=RANDOM_SEED)
    model = build_model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n=== 검증 결과 ===")
    print(classification_report(y_test, y_pred, digits=3))
    labels = sorted(counts)
    print("혼동 행렬 (행=정답, 열=예측):", labels)
    print(confusion_matrix(y_test, y_pred, labels=labels))

    # 2) 전체 데이터로 다시 학습해서 저장
    final_model = build_model()
    final_model.fit(X, y)
    joblib.dump({"model": final_model, "labels": list(final_model.classes_)}, CLASSIFIER_PATH)
    print(f"\n저장 완료: {CLASSIFIER_PATH}")


if __name__ == "__main__":
    main()
