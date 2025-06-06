import json
import requests
import os
from config.slack_config import SLACK_WEBHOOK_URL

RESULT_PATH = "./outputs/result.json"

# Slack 메시지 전송 함수
def send_slack_notification():
    if not os.path.exists(RESULT_PATH):
        print("[경고] result.json 파일이 존재하지 않습니다.")
        return

    with open(RESULT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    is_warning = data["emotion"] in ["anger", "fear"]

    message = (
        f"[DoorBox] {'(주의)' if is_warning else ''}방문자가 인식되었습니다.\n"
        f"- 시각: {data['timestamp']}\n"
        f"- 감정: {data['emotion']}\n"
        f"- 성별: {data['gender']}\n"
        f"- 연령대: {data['age_range']}\n"
        f"- 클립 보기: (나중에 링크추가)"
    )

    payload = {"text": message}
    response = requests.post(SLACK_WEBHOOK_URL, json=payload)

    if response.status_code == 200:
        print("Slack 알림 전송 완료")
    else:
        print(f"Slack 전송 실패: {response.status_code} - {response.text}")
