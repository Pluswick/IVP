import cv2
import json
import kp

# FER+ 기반 감정 라벨 정의 (MobileNetV2 기반)
EMOTION_LABELS = {
    "0": "neutral",
    "1": "happiness",
    "2": "surprise",
    "3": "sadness",
    "4": "anger",
    "5": "disgust",
    "6": "fear",
    "7": "contempt"
}

# 감정 분류 함수: crop 이미지를 NPU에 넣고 가장 높은 감정 라벨 반환
def infer_emotion(device_group, model_id, crop_img):
    rgb565 = cv2.cvtColor(crop_img, cv2.COLOR_BGR2BGR565)
    descriptor = kp.GenericImageInferenceDescriptor(
        model_id=model_id,
        inference_number=0,
        input_node_image_list=[
            kp.GenericInputNodeImage(
                image=rgb565,
                resize_mode=kp.ResizeMode.KP_RESIZE_DISABLE,
                padding_mode=kp.PaddingMode.KP_PADDING_DISABLE,
                normalize_mode=kp.NormalizeMode.KP_NORMALIZE_KNERON
            )
        ]
    )

    # 감정 추론 전송 및 수신
    kp.inference.generic_image_inference_send(device_group, descriptor)
    result = kp.inference.generic_image_inference_receive(device_group)
    out = kp.inference.generic_inference_retrieve_float_node(0, result, kp.ChannelOrdering.KP_CHANNEL_ORDERING_CHW)

    scores = out.tolist()
    best_idx = scores.index(max(scores))
    label = EMOTION_LABELS.get(str(best_idx), "Unknown")

    return label
