import kp
import time

from inference.1_yolov5_face_detect import detect_faces
from inference.2_cropper import crop_faces
from inference.3_emotion_infer import infer_emotion
from inference.4_gender_age_infer import infer_gender_age
from inference.5_result_packager import save_result
from inference.6_slack_trigger import send_slack_notification

MODEL_ID_YOLO = 211
MODEL_ID_EMOTION = 211  # YOLO+감정 모델이 컴바인된 경우 동일 ID

# 인퍼런스 파이프라인 메인
if __name__ == "__main__":
    device_group = kp.core.connect_devices()[0]
    kp.device.set_timeout(device_group, 10000)

    print("[1단계] YOLO 얼굴 검출 실행 중...")
    frame, boxes = detect_faces(device_group, MODEL_ID_YOLO)

    if not boxes:
        print("[종료] 얼굴이 감지되지 않았습니다.")
    else:
        print(f"[2단계] {len(boxes)}개의 얼굴 감지됨. Crop 진행 중...")
        cropped_faces = crop_faces(frame, boxes)

        for face_img, path in cropped_faces:
            print(f"[3단계] 감정 분류 중... ({path})")
            emotion = infer_emotion(device_group, MODEL_ID_EMOTION, face_img)

            print("[4단계] 성별/연령대 분류 중...")
            gender, age = infer_gender_age(path)

            print("[5단계] 결과 저장 중...")
            result_path = save_result(emotion, gender, age)

            print("[6단계] Slack 알림 전송 중...")
            send_slack_notification()

    kp.core.disconnect_devices()
