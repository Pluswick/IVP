
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
├─ models/
│   └─ nef/
│       └─ combined.nef    # Yolov5s + MobileNetV2 컴바인 모델.nef
│   └─ pth/
│       ├─ mobilenetv3_age.pth
│       └─ mobilenetv3_gender.pth

├─ config/
│   ├─ emotion_labels.json              # 감정 softmax 매핑
│   ├─ slack_config.py                  # Webhook URL 등
│   └─ gender_age_labels.json           # 성별/연령 매핑 (선택)

├─ inference/                           # 인퍼런스 관련 코드들
│   ├─ run_inference.py                 # 메인 인퍼런스 파이프라인
│   ├─ 1_yolov5_face_detect.py            # 얼굴 검출 (NPU)
│   ├─ 2_cropper.py                       # 얼굴 crop
│   ├─ 3_emotion_infer.py                 # 감정 분류 (NPU)
│   ├─ 4_gender_age_infer.py              # 성별/연령 분류 (CPU)
│   ├─ 5_result_packager.py               # 감정+성별+연령 → result.json 저장
│   └─ 6_slack_trigger.py                 # result.json → Slack 전송

├─ outputs/
│   ├─ faces/                           # crop 저장
│   ├─ clips/                           # 5초 영상 저장
│   ├─ logs/                            # log 텍스트
│   └─ result.json                      # Slack 전송용 정보 저장 (최종)

├─ res/
│   └─ firmware/
│       └─ KL630/
│           └─ kp_firmware.tar

├─ run_doorbox.py                       # 실제 실행하는 통합 진입 파일
├─ run_doorbox_live.py                  # 실시간 UI 데모용 실행 파일 (미완성)
├─ requirements.txt
└─ README.md

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

