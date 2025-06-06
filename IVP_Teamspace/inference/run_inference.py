import kp
from datetime import datetime
from inference.1_yolov5_face_detect import detect_faces
from inference.2_cropper import crop_faces
from inference.3_emotion_infer import infer_emotion
from inference.4_gender_age_infer import infer_gender_age
from inference.5_result_packager import save_result
from UI.slack_UI import process_detection

MODEL_ID_YOLO = 22222
MODEL_ID_EMOTION = 11111

device_group = None  # 디바이스 연결 상태를 전역으로 유지
def run_all_inference(frame):
    global device_group

    # 0. CatchCAM 연결
    try:
        if device_group is None:
            print("[INFO] CatchCAM 연결 중...")
            device_group = kp.core.connect_devices()[0]
            kp.device.set_timeout(device_group, 10000)
            print("[INFO] CatchCAM 연결 성공.")
    except Exception as e:
        print(f"[ERROR] CatchCAM 연결 실패: {e}")
        return None, None

    # 1. YOLOv5 얼굴 검출
    try:
        print("[INFO] YOLOv5 얼굴 검출 시작...")
        _, boxes = detect_faces(device_group, MODEL_ID_YOLO, frame)
        if not boxes:
            print("[WARNING] 얼굴이 감지되지 않았습니다.")
            return None, None
        print(f"[INFO] 얼굴 검출 성공: {len(boxes)}명")
    except Exception as e:
        print(f"[ERROR] YOLO 얼굴 검출 실패: {e}")
        return None, None

    # 2. 얼굴 Crop (가장 큰 얼굴 1개만 처리)
    try:
        print("[INFO] 얼굴 Crop 수행 중...")
        cropped_faces = crop_faces(frame, boxes)
        face_img, face_path = cropped_faces[0]  # 첫 번째 얼굴 사용
        print(f"[INFO] 얼굴 Crop 완료: {face_path}")
    except Exception as e:
        print(f"[ERROR] 얼굴 Crop 실패: {e}")
        return None, None

    # 3. 감정 분류 (NPU)
    try:
        print("[INFO] 감정 분류 시작...")
        emotion = infer_emotion(device_group, MODEL_ID_EMOTION, face_img)
        print(f"[INFO] 감정 분류 결과: {emotion}")
    except Exception as e:
        print(f"[ERROR] 감정 분류 실패: {e}")
        emotion = "unknown"

    # 4. 성별/연령 분류 (로컬 CPU 실행)
    try:
        print("[INFO] 성별/연령 분류 시작...")
        gender, age = infer_gender_age(face_path)
        print(f"[INFO] 성별: {gender}, 연령대: {age}")
    except Exception as e:
        print(f"[ERROR] 성별/연령 분류 실패: {e}")
        gender, age = "unknown", "unknown"

    # 5. 결과 저장 (result.json)
    try:
        result = {
            "emotion": emotion,
            "gender": gender,
            "age": age
        }
        save_result(result)
        print("[INFO] 최종 결과 저장 완료.")
    except Exception as e:
        print(f"[ERROR] 결과 저장 실패: {e}")
        return None, None

    # 6. Slack 알림 전송
    try:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        process_detection(now, emotion, gender, age)
        print("[INFO] Slack 알림 전송 완료.")
    except Exception as e:
        print(f"[ERROR] Slack 전송 실패: {e}")

    return result, boxes[0]
