# run_doorbox_live.py
# 실시간 UI 기반 DoorBox 데모 실행 파일

import cv2
import kp
import time

from inference.1_yolov5_face_detect import detect_faces
from inference.2_cropper import crop_faces
from inference.3_emotion_infer import infer_emotion
from inference.4_gender_age_infer import infer_gender_age

MODEL_ID_YOLO = 22222
MODEL_ID_EMOTION = 11111


def draw_results(frame, box, emotion, gender, age):
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"{emotion}, {gender}, {age}"
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def main():
    print("📺 실시간 DoorBox UI 실행")
    cap = cv2.VideoCapture(0)
    device_group = kp.core.connect_devices()[0]
    kp.device.set_timeout(device_group, 10000)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_copy = frame.copy()
            frame_resized = cv2.resize(frame_copy, (640, 640))

            # 얼굴 검출
            boxes = detect_faces(device_group, MODEL_ID_YOLO, frame_resized)

            if boxes:
                cropped_faces = crop_faces(frame_resized, boxes)

                for (crop_img, path), box in zip(cropped_faces, boxes):
                    emotion = infer_emotion(device_group, MODEL_ID_EMOTION, crop_img)
                    gender, age = infer_gender_age(path)
                    draw_results(frame_resized, box, emotion, gender, age)

            cv2.imshow("DoorBox Live", frame_resized)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        kp.core.disconnect_devices()
        print("✅ 실시간 UI 종료")


if __name__ == "__main__":
    main()
