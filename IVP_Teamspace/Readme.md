
# 📦 Door-Box: AI 기반 주거 공간 방문자 분석 블랙박스 프로젝트 (CatchCAM 기반)

> 웹캠 → NPU → 얼굴 검출 + 감정 분류 → Slack 실시간 알림  
> 단일 Python 파일로 통합된 실시간 데모 시스템


---

## 주요 기능

- 얼굴 인식 (YOLOv5s, NPU)
- 감정 분류 (FER+, MobileNetV2, NPU)
- 성별/연령대 분류 (MobileNetV3, CPU)
- Slack 실시간 경고 메시지 전송
- 스마트폰 앱 or PC 응용 프로그램을 통한 로그 확인인

---

## 📁 프로젝트 구조

```

DoorBox\_Project/
├─ run\_doorbox.py             # 메인 실행 파일 (전체 파이프라인)
├─ run\_doorbox\_live.py        # 실시간 UI 데모 실행 파일
├─ inference/                 # 인퍼런스 구성 모듈
│   ├─ 1\_yolov5\_face\_detect.py
│   ├─ 2\_cropper.py
│   ├─ 3\_emotion\_infer.py
│   ├─ 4\_gender\_age\_infer.py
│   ├─ 5\_result\_packager.py
│   ├─ 6\_slack\_trigger.py
│   └─ run\_inference.py
├─ models/pt/                 # .pth 모델 (gender, age)
├─ outputs/                   # 결과 저장 (.json, clips, faces 등)
├─ notifier/                  # Slack 연동
├─ requirements.txt
└─ README.md

````

---

## 실행 방법법

```bash
# 기본 파이프라인 실행
python run_doorbox.py

# 실시간 데모 실행
python run_doorbox_live.py
````

---

## 알림 예시 (Slack)

```
[DoorBox] (주의)방문자가 인식되었습니다.
- 시각: 2025-06-06 16:30
- 감정: anger
- 성별: 여자
- 연령대: 30대
```

---

## 환경 안내내

* Python 3.8+
* Kneron SDK (`kneron-sdk`)
* OpenCV, PyTorch, Pillow, Requests

---

