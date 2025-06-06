import os
import cv2
from datetime import datetime

CROP_SAVE_DIR = "./outputs/faces"

# YOLO 바운딩 박스를 기반으로 얼굴을 crop하여 저장하고, crop 이미지 리스트 반환
def crop_faces(frame, box_list):
    os.makedirs(CROP_SAVE_DIR, exist_ok=True)
    cropped_faces = []

    for idx, box in enumerate(box_list):
        x1 = max(int(box.x1), 0)
        y1 = max(int(box.y1), 0)
        x2 = min(int(box.x2), frame.shape[1])
        y2 = min(int(box.y2), frame.shape[0])

        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"face_{timestamp}_{idx}.jpg"
        path = os.path.join(CROP_SAVE_DIR, filename)
        cv2.imwrite(path, face_crop)
        cropped_faces.append((face_crop, path))

    return cropped_faces
