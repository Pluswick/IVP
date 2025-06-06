import cv2
import platform
import kp
import os
from utils.ExamplePostProcess import post_process_yolo_v5

# YOLO 얼굴 검출 결과를 리턴하는 함수
def detect_faces(device_group, model_id):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if platform.system() == 'Windows' else 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("카메라에서 프레임을 읽을 수 없습니다.")

    rgb565 = cv2.cvtColor(frame, cv2.COLOR_BGR2BGR565)
    descriptor = kp.GenericImageInferenceDescriptor(
        model_id=model_id,
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

    # 전송 및 결과 수신
    kp.inference.generic_image_inference_send(device_group, descriptor)
    raw_result = kp.inference.generic_image_inference_receive(device_group)

    # 출력 노드에서 float 결과 추출
    outputs = []
    for i in range(raw_result.header.num_output_node):
        node_out = kp.inference.generic_inference_retrieve_float_node(
            node_idx=i,
            generic_raw_result=raw_result,
            channels_ordering=kp.ChannelOrdering.KP_CHANNEL_ORDERING_CHW
        )
        outputs.append(node_out)

    # YOLO 후처리 (bbox 반환)
    yolo_result = post_process_yolo_v5(
        inference_float_node_output_list=outputs,
        hardware_preproc_info=raw_result.header.hw_pre_proc_info_list[0],
        thresh_value=0.3,
        with_sigmoid=False
    )

    cap.release()
    return frame, yolo_result.box_list
