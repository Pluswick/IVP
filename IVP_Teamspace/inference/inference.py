"""
Door‑Box : inference.py (rev‑2025‑06‑16‑b)
"""

# --- 기본 라이브러리 ---------------------------------------------------------
from __future__ import annotations
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
from PIL import Image, ImageFont, ImageDraw
from datetime import datetime
from collections import deque   
import warnings

import kp

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# YOLO 후처리 유틸
sys.path.append("D:/IVP_git/IVP/IVP_Teamspace/CatchCAM/kneron_plus/python/example")
from utils.ExamplePostProcess import post_process_yolo_v5

# Slack UI
sys.path.append("D:/IVP_git/IVP/IVP_Teamspace/UI")
from slack_UI import process_detection

# --- 모델/경로 상수 -----------------------------------------------------------
MODEL_ID_YOLO    = 32769
MODEL_ID_EMOTION = 22222

# CatchCAM USB port
detection_usb_port_id = 17

# 펌웨어 및 NEF 경로
SCPU_FW_PATH   = "D:/IVP_git/IVP/IVP_Teamspace/CatchCAM/kneron_plus/res/firmware/KL630/kp_firmware.tar"
SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
NEF_MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "models", "nef", "combined_new_630.nef")

# 성별/연령 .pth 경로
AGE_MODEL_PATH    = "D:/IVP_git/IVP/IVP_Teamspace/models/pth/mobilenetv3_age.pth"
GENDER_MODEL_PATH = "D:/IVP_git/IVP/IVP_Teamspace/models/pth/mobilenetv3_gender.pth"

# 클래스 목록
AGE_CLASSES    = ["Under 9","10s","20s","30s","40s","50s","60s","Over 70s"]
GENDER_CLASSES = ["Male","Female"]
EMOTIONS       = ["neutral","happiness","surprise","sadness","anger","disgust","fear","contempt"]

# 출력 폴더
CAPTURE_SAVE_DIR = "D:/IVP_git/IVP/IVP_Teamspace/outputs/capture"
CLIP_SAVE_DIR = "D:/IVP_git/IVP/IVP_Teamspace/outputs/clips"   
RESULT_PATH   = "D:/IVP_git/IVP/IVP_Teamspace/outputs/result.json"
os.makedirs(CAPTURE_SAVE_DIR, exist_ok=True)
os.makedirs(CLIP_SAVE_DIR,  exist_ok=True)   
os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)

# 전역 상태
device_group = None
model_loaded = False

# 감정 분류 모델 입력 크기 (추후 수정되면 자동 반영)
EMO_H, EMO_W = 224, 224

# 성별/연령 분류용 transform
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# --- CatchCAM 디바이스 초기화 --------------------------------------------------

def init_device():
    """CatchCAM 디바이스 초기화·펌웨어·모델 업로드 (1회)"""
    global device_group, model_loaded, EMO_H, EMO_W
    if device_group is not None and model_loaded:
        return device_group

    print("[Debug] init_device 호출")
    print("[정보] CatchCAM 연결 중...")
    device_group = kp.core.connect_devices(usb_port_ids=[detection_usb_port_id])
    kp.core.set_timeout(device_group=device_group, milliseconds=10000)
    print("[정보] 펌웨어 업로드 중...")
    kp.core.load_firmware_from_file(device_group,
        scpu_fw_path=SCPU_FW_PATH, ncpu_fw_path="")
    print("[정보] NEF 모델 로드 중...")
    mdesc = kp.core.load_model_from_file(device_group,
        file_path=NEF_MODEL_PATH)
    print(f"[Debug] 로드된 모델 수: {len(mdesc.models)}")

    # 감정 모델 입력 크기 자동 추출 (두 번째 모델)
    shape = mdesc.models[1].input_nodes[0].shape_npu  # [N,C,H,W]
    EMO_H, EMO_W = shape[2], shape[3]
    print(f"[Debug] 감정 모델 입력 크기: {EMO_W}×{EMO_H}")

    model_loaded = True
    print("[정보] 장치 초기화 완료.")
    return device_group

# --- 프레임 단위 처리 ----------------------------------------------------------

def process_frame(frame):
    """
    1) YOLO 검출
    2) 단계별 UI: Detection/Cropped Face/Emotion Input
    3) 감정 분류 → 성별·연령 분류
    4) 최종 오버레이된 프레임, (emotion, gender, age) 반환
    """
    dg = init_device()

    # 1. YOLOv5 얼굴 검출
    rgb565 = cv2.cvtColor(frame, cv2.COLOR_BGR2BGR565)
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
    kp.inference.generic_image_inference_send(dg, yolo_desc)
    raw = kp.inference.generic_image_inference_receive(dg)
    outputs = [
        kp.inference.generic_inference_retrieve_float_node(
            node_idx=i, generic_raw_result=raw,
            channels_ordering=kp.ChannelOrdering.KP_CHANNEL_ORDERING_CHW
        ) for i in range(raw.header.num_output_node)
    ]
    yolo_res = post_process_yolo_v5(
        inference_float_node_output_list=outputs,
        hardware_preproc_info=raw.header.hw_pre_proc_info_list[0],
        thresh_value=0.3, with_sigmoid=False
    )

    # 단계별 UI: Detection
    det_vis = frame.copy()
    for b in yolo_res.box_list:
        cv2.rectangle(det_vis,
                      (int(b.x1),int(b.y1)),
                      (int(b.x2),int(b.y2)),
                      (0,255,0),2)
    cv2.imshow("1. Detection", det_vis)

    if not yolo_res.box_list:
        return frame, None, None, None

    # 2. 얼굴 크롭
    b = yolo_res.box_list[0]
    x1, y1 = max(int(b.x1),0), max(int(b.y1),0)
    x2, y2 = min(int(b.x2),frame.shape[1]), min(int(b.y2),frame.shape[0])
    face_img = frame[y1:y2, x1:x2]

    # 3. 감정 입력 리사이즈 & UI
    face_resized = cv2.resize(face_img, (EMO_W, EMO_H))
    h,w = face_resized.shape[:2]
    if w%2: face_resized = face_resized[:, :-1]
    if h%2: face_resized = face_resized[:-1, :]

    # 4. 감정 분류 (NPU)
    emo_rgb565 = cv2.cvtColor(face_resized, cv2.COLOR_BGR2BGR565)
    emo_desc = kp.GenericImageInferenceDescriptor(
        model_id=MODEL_ID_EMOTION,
        inference_number=0,
        input_node_image_list=[
            kp.GenericInputNodeImage(
                image=emo_rgb565,
                resize_mode=kp.ResizeMode.KP_RESIZE_DISABLE,
                padding_mode=kp.PaddingMode.KP_PADDING_DISABLE,
                normalize_mode=kp.NormalizeMode.KP_NORMALIZE_KNERON
            )
        ]
    )
    kp.inference.generic_image_inference_send(dg, emo_desc)
    emo_raw = kp.inference.generic_image_inference_receive(dg)
    emo_out = kp.inference.generic_inference_retrieve_float_node(
        0, emo_raw, kp.ChannelOrdering.KP_CHANNEL_ORDERING_CHW
    )
    scores = emo_out.ndarray.flatten().tolist()
    idx    = scores.index(max(scores))
    emotion = EMOTIONS[idx]

    # 5. 성별/연령 분류 (CPU)
    face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    tensor   = transform(face_pil).unsqueeze(0)

    gender_sd = torch.load(GENDER_MODEL_PATH, map_location='cpu')
    age_sd    = torch.load(AGE_MODEL_PATH,    map_location='cpu')
    gm = models.mobilenet_v3_small(pretrained=False)
    in_f = gm.classifier[-1].in_features
    gm.classifier[-1] = nn.Linear(in_f, 2)
    gm.load_state_dict(gender_sd); gm.eval()
    am = models.mobilenet_v3_small(pretrained=False)
    in_f2 = am.classifier[-1].in_features
    am.classifier[-1] = nn.Linear(in_f2, 8)
    am.load_state_dict(age_sd); am.eval()
    with torch.no_grad():
        g_out = gm(tensor); a_out = am(tensor)
    gender = GENDER_CLASSES[int(g_out.argmax())]
    age    = AGE_CLASSES[int(a_out.argmax())]

    # 6. 최종 오버레이된 프레임
    vis = frame.copy()
    cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0),2)
    cv2.putText(vis, f"{emotion}/{gender}/{age}",
                (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255,255,255), 2)
    #cv2.imshow("4. Result Overlay", vis)

    return vis, emotion, gender, age


def save_result(emotion, gender, age):
    """마지막 결과 JSON 저장"""
    result = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "emotion": emotion,
        "gender": gender,
        "age_range": age
    }
    with open(RESULT_PATH, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    print("[정보] 실시간 추론 시작 (ESC 키로 종료)")
    cap = cv2.VideoCapture(0,
        cv2.CAP_DSHOW if platform.system()=='Windows' else 0)
    if not cap.isOpened():
        print("[오류] 카메라 열기 실패."); sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_buf = deque(maxlen=int(fps*5))       

    last_emotion = last_gender = last_age = None

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        output_frame, emotion, gender, age = process_frame(frame)
        frame_buf.append(output_frame.copy())       

        if emotion is not None:
            last_emotion, last_gender, last_age = emotion, gender, age

        cv2.imshow("DoorBox Inference", output_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:                               
             break

    cap.release()
    cv2.destroyAllWindows()

    # --- 스냅숏 & 5초 클립 저장 ------------------------------
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    snap_path = os.path.join(CAPTURE_SAVE_DIR, f"face_{ts}.jpg")
    cv2.imwrite(snap_path, frame_buf[-1])
    print(f"[정보] 스냅숏 저장 완료 → {snap_path}")

    buffer_length = len(frame_buf)
    fps = buffer_length / 5  # 프레임 개수 / 5초
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    h, w = frame_buf[0].shape[:2]
    clip_path = os.path.join(CLIP_SAVE_DIR, f"clip_{ts}.mp4")
    vw = cv2.VideoWriter(clip_path, fourcc, fps, (w, h))
    for f in frame_buf: vw.write(f)
    vw.release()
    print(f"[정보] 5초 클립 저장 완료 → {clip_path}")
    # --------------------------------------------------------------

    if last_emotion is not None:
        save_result(last_emotion, last_gender, last_age)
        process_detection(datetime.now().strftime("%H:%M"),
                            last_emotion, last_gender, last_age)
        print("[정보] 마지막 결과 저장 및 Slack 전송 완료")