
# 📦 Door-Box: AI 기반 주거 공간 방문자 분석 블랙박스 프로젝트 (CatchCAM 기반)

> 웹캠 → CatchCAM NPU → 얼굴 검출 + 감정 분류 → 로컬 CPU → 성별/연령대 분류 → Slack 실시간 알림 & 클립 저장

---

## 주요 기능

- 얼굴 인식 (YOLOv5s, NPU)
- 감정 분류 (FER+, MobileNetV2, NPU)
- 성별/연령대 분류 (MobileNetV3, CPU)
- Slack 실시간 알림 전송 
- 스마트폰 앱 or PC 응용 프로그램을 통한 로그 확인
- Google Sheet & Slack API 사용용

---

## 📁 프로젝트 구조

```jsx
DoorBox_Project/
├── models/
│   ├── nef/
│   │   └── combined.nef              # YOLOv5s + MobileNetV2 결합 모델
│   └── pth/
│       ├── mobilenetv3_age.pth       # 연령대 분류 모델
│       └── mobilenetv3_gender.pth    # 성별 분류 모델

├── inference/                        
│   ├── 1_yolov5_face_detect.py            # 얼굴 검출 (NPU)
│   ├── 2_cropper.py                       # 얼굴 crop 이미지 저장
│   ├── 3_emotion_infer.py                 # 감정 분류 (NPU)
│   ├── 4_gender_age_infer.py              # 성별/연령대 분류 (CPU)
│   ├── 5_result_packager.py               # 감정+성별+연령 결과 저장
│   └── run_inference.py              # 전체 인퍼런스 흐름 통합 (UI 없음)

├── UI/                              
│   ├── slack_URL.py                  # Webhook URL 정의
│   └── slack_UI.py                   # Slack 메시지 포맷 + 전송 함수

├── outputs/                          # 출력 결과 저장
│   ├── faces/                          # crop된 얼굴 이미지
│   ├── clips/                          # 5초 영상 클립
│   ├── logs/                           # 로그 텍스트
│   └── result.json                     # Slack 전송용 결과 파일

├── res/
│   └── firmware/
│       └── KL630/
│           └── kp_firmware.tar       # CatchCAM용 펌웨어 파일

├── run_doorbox.py                   # 통합 실행 파일 (UI 없음)
├── run_doorbox_live.py              # 실시간 UI 데모용 실행 파일
├── requirements.txt
└── README.md

```

---

## 실행 방법

```bash
# 기본 파이프라인 실행
python run_doorbox.py

# 실시간 데모 실행
python run_doorbox_live.py
````

---

## 알림 예시 (Slack)

```
[Door-Box] (주의)방문자가 인식되었습니다.
- 시각: 2025-06-06 16:30
- 감정: anger
- 성별: 여자
- 연령대: 30대
```

---

## 환경 안내

* Python 3.8+
* Kneron SDK (`kneron-sdk`)
* OpenCV, PyTorch, Pillow, Requests

---

