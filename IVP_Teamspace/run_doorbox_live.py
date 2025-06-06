"""
run_doorbox_live.py
DoorBox 실시간 시연용 UI + Slack 알림 포함 버전
"""

import cv2
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


def draw_result(frame, bbox, result_dict):
    x1, y1, x2, y2 = bbox
    label = f"{result_dict['emotion']} / {result_dict['gender']} / {result_dict['age']}"
    color = (0, 255, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def process_frame(frame, device_group, sent_flag):
    try:
        _, boxes = detect_faces(device_group, MODEL_ID_YOLO, frame)
        if not boxes:
            return frame, sent_flag

        cropped_faces = crop_faces(frame, boxes)
        face_img, face_path = cropped_faces[0]
        emotion = infer_emotion(device_group, MODEL_ID_EMOTION, face_img)
        gender, age = infer_gender_age(face_path)

        result = {"emotion": emotion, "gender": gender, "age": age}
        save_result(result)

        if not sent_flag:
            now = datetime.now().strftime("%Y-%m-%d %H:%M")
            process_detection(now, emotion, gender, age)
            sent_flag = True

        draw_result(frame, boxes[0], result)
        return frame, sent_flag

    except Exception as e:
        print(f"[ERROR] 프레임 처리 실패: {e}")
        return frame, sent_flag


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] 웹캠 열기 실패")
        return

    print("[INFO] 실시간 DoorBox 시연 시작 (q: 종료)")
    sent_flag = False

    try:
        device_group = kp.core.connect_devices()[0]
        kp.device.set_timeout(device_group, 10000)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame, sent_flag = process_frame(frame, device_group, sent_flag)
            cv2.imshow("DoorBox Live", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        kp.core.disconnect_devices()
        print("[INFO] 종료됨")


if __name__ == "__main__":
    main()
