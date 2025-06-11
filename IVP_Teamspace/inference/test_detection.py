import os
import sys
import json
import cv2
import platform
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from PIL import Image
from datetime import datetime

import kp

# YOLO 후처리용 외부 유틸 경로
sys.path.append("D:/IVP_git/IVP/IVP_Teamspace/CatchCAM/kneron_plus/python/example")
from utils.ExamplePostProcess import post_process_yolo_v5

# Slack UI 경로
sys.path.append("D:/IVP_git/IVP/IVP_Teamspace/UI")
from slack_UI import process_detection

# 모델 ID (detection.nef 에 정의된 ID)
MODEL_ID_YOLO = None

# 연결된 CatchCAM USB 포트 ID
detection_usb_port_id = 17

# 펌웨어 및 NEF 모델 경로
SCPU_FW_PATH   = "D:/IVP_git/IVP/IVP_Teamspace/CatchCAM/kneron_plus/res/firmware/KL630/kp_firmware.tar"
SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
DETECTION_NEF_PATH = os.path.join(SCRIPT_DIR, "..", "models", "nef", "detection_630.nef")

# 성별/연령 분류 모델 경로
AGE_MODEL_PATH    = "D:\IVP_git\IVP\IVP_Teamspace\models\pth\mobilenetv3_age.pth"
GENDER_MODEL_PATH = "D:\IVP_git\IVP\IVP_Teamspace\models\pth\mobilenetv3_gender.pth"

# 출력 폴더 및 결과 JSON 경로
CROP_SAVE_DIR = "D:\IVP_git\IVP\IVP_Teamspace\outputs/faces"
RESULT_PATH   = "D:\IVP_git\IVP\IVP_Teamspace\outputs/result.json"
os.makedirs(CROP_SAVE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)

# 전역 상태
device_group = None
model_loaded = False

# 성별/연령 분류용 transform 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])


def infer_gender_age(image_path):
    """저장된 얼굴 이미지에 대해 성별 및 연령대 추론을 수행합니다."""
    print("[디버그] 성별/연령 분류 시작:", image_path)
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0)

    # 1) state_dict만 불러오고
    gender_sd = torch.load(GENDER_MODEL_PATH, map_location='cpu')
    age_sd    = torch.load(AGE_MODEL_PATH,    map_location='cpu')

    # 2) torchvision의 MobileNetV3-Small 구조를 즉석 생성
    gender_model = models.mobilenet_v3_small(pretrained=False)
    # classifier 마지막 레이어 재구성 (기존 1000->2)
    in_feat = gender_model.classifier[-1].in_features
    gender_model.classifier[-1] = nn.Linear(in_feat, 2)

    age_model    = models.mobilenet_v3_small(pretrained=False)
    in_feat2 = age_model.classifier[-1].in_features
    age_model.classifier[-1] = nn.Linear(in_feat2, 8)

    # 3) state_dict 로드
    gender_model.load_state_dict(gender_sd)
    age_model.load_state_dict(age_sd)

    # 4) eval 모드 전환
    gender_model.eval()
    age_model.eval()

    # 5) 추론
    with torch.no_grad():
        gender_out = gender_model(tensor)
        age_out    = age_model(tensor)

    gender_idx = gender_out.argmax().item()
    age_idx    = age_out.argmax().item()

    GENDER_CLASSES = ["남자", "여자"]
    AGE_CLASSES    = ["9세 이하", "10대", "20대", "30대", "40대", "50대", "60대", "70세 이상"]

    gender = GENDER_CLASSES[gender_idx] if gender_idx < len(GENDER_CLASSES) else "Unknown"
    age    = AGE_CLASSES[age_idx]       if age_idx    < len(AGE_CLASSES)    else "Unknown"

    print(f"[디버그] 성별/연령 분류 결과: {gender}, {age}")
    return gender, age

def save_result(gender, age):
    """추론 결과를 JSON 파일로 저장합니다."""
    print("[디버그] 결과 JSON 저장 시작")
    result = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "gender": gender,
        "age_range": age
    }
    with open(RESULT_PATH, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("[디버그] 결과 JSON 저장 완료:", RESULT_PATH)
    return RESULT_PATH

def init_device():
    """
    CatchCAM 디바이스 그룹을 초기화하고,
    펌웨어 및 detection.nef 모델을 한 번만 업로드합니다.
    """
    global device_group, model_loaded
    if device_group is not None and model_loaded:
        print("[디버그] 이미 초기화된 디바이스 사용")
        return device_group

    print("\n[디버그] init_device 호출")
    print("[정보] CatchCAM 연결 중...")
    device_group = kp.core.connect_devices(usb_port_ids=[detection_usb_port_id])
    print("[디버그] 장치 연결 성공, DeviceGroup:", device_group)

    kp.core.set_timeout(device_group=device_group, milliseconds=10000)
    print("[디버그] 타임아웃 설정 완료 (10000ms)")

    print("[정보] 펌웨어 업로드 중...")
    kp.core.load_firmware_from_file(device_group, scpu_fw_path=SCPU_FW_PATH, ncpu_fw_path="")
    print("[디버그] 펌웨어 업로드 완료")

    print("[정보] detection.nef 모델 로드 중...")
    print("[디버그] NEF 경로 존재 여부:", os.path.exists(DETECTION_NEF_PATH))
    model_desc = kp.core.load_model_from_file(
        device_group=device_group,
        file_path=DETECTION_NEF_PATH
    )
    print("[디버그] detection.nef 모델 로드 완료")

    # ───── 2) init_device()에서 모델 로드 후 실제 ID 저장 ─────
    global MODEL_ID_YOLO
    MODEL_ID_YOLO = model_desc.models[0].id   # ★ 실제 ID
    print(f"[디버그] 로드된 모델 ID: {MODEL_ID_YOLO}")

    print(f"[디버그] 로드된 모델 수: {len(model_desc.models)}")

    model_loaded = True
    print("[정보] 장치 초기화 완료.")
    return device_group

def process_frame(frame):
    """
    단일 프레임을 처리합니다:
    1. 얼굴 검출 → 2. 크롭 → 3. 성별/연령 추론
    4. 결과 저장 → 5. Slack 알림 → 6. UI 시각화
    """
    print("\n[디버그] process_frame 시작")
    dg = init_device()

    # 1. YOLO 얼굴 검출
    print("[디버그] YOLO 추론용 Descriptor 생성")
    resized = cv2.resize(frame, (640, 640))
    rgb565 = cv2.cvtColor(resized, cv2.COLOR_BGR2BGR565)
    yolo_desc = kp.GenericImageInferenceDescriptor(
        model_id=MODEL_ID_YOLO,
        inference_number=0,
        input_node_image_list=[
            kp.GenericInputNodeImage(
                image=rgb565,
                resize_mode=kp.ResizeMode.KP_RESIZE_ENABLE,
                padding_mode=kp.PaddingMode.KP_PADDING_CORNER,
                normalize_mode=kp.NormalizeMode.KP_NORMALIZE_KNERON
            )
        ]
    )
    print("[디버그] YOLO 입력 전송")
    kp.inference.generic_image_inference_send(dg, yolo_desc)

    print("[디버그] YOLO 결과 수신 대기")
    raw = kp.inference.generic_image_inference_receive(dg)
    print("[디버그] YOLO 결과 수신 완료")

    outputs = [
        kp.inference.generic_inference_retrieve_float_node(
            node_idx=i,
            generic_raw_result=raw,
            channels_ordering=kp.ChannelOrdering.KP_CHANNEL_ORDERING_CHW
        )
        for i in range(raw.header.num_output_node)
    ]
    print("[디버그] YOLO 출력 노드 추출 완료:", len(outputs), "nodes")

    print("[디버그] YOLO 후처리 시작")
    yolo_res = post_process_yolo_v5(
        inference_float_node_output_list=outputs,
        hardware_preproc_info=raw.header.hw_pre_proc_info_list[0],
        thresh_value=0.3,
        with_sigmoid=False
    )
    print("[디버그] YOLO 후처리 완료, 감지된 얼굴 수:", len(yolo_res.box_list))

    if not yolo_res.box_list:
        print("[경고] 얼굴 검출되지 않음. 프레임 반환")
        return frame

    # 2. 얼굴 크롭 및 저장
    print("[디버그] 얼굴 크롭 시작")
    box = yolo_res.box_list[0]
    x1, y1 = max(int(box.x1), 0), max(int(box.y1), 0)
    x2, y2 = min(int(box.x2), frame.shape[1]), min(int(box.y2), frame.shape[0])
    face_img = frame[y1:y2, x1:x2]

    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"face_{ts}.jpg"
    fpath = os.path.join(CROP_SAVE_DIR, fname)
    cv2.imwrite(fpath, face_img)
    print("[디버그] 얼굴 이미지 저장 완료:", fpath)

    # 3. 성별/연령 추론 (CPU)
    gender, age = infer_gender_age(fpath)

    # 4. 결과 저장 및 Slack 알림
    save_result(gender, age)
    print("[디버그] Slack 알림 전송 시작")
    process_detection(datetime.now().strftime("%H:%M"), gender, age)
    print("[디버그] Slack 알림 전송 완료")

    # 5. 결과 시각화
    print("[디버그] 시각화 오버레이 수행")
    label = f"{gender}/{age}"
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    print("[디버그] process_frame 종료")
    return frame

if __name__ == '__main__':
    print("\n[정보] 실시간 추론 시작 (ESC 키로 종료)")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if platform.system() == 'Windows' else 0)
    if not cap.isOpened():
        print("[오류] 카메라 열기 실패.")
        sys.exit(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[오류] 프레임 수신 실패.")
            break

        output_frame = process_frame(frame)
        cv2.imshow("DoorBox Inference", output_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            print("\n[정보] ESC 키 감지, 종료합니다.")
            break

    cap.release()
    cv2.destroyAllWindows()
