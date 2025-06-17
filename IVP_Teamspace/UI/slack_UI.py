from slack_sdk.webhook import WebhookClient
from slack_URL import SLACK_WEBHOOK_URL
from google_sheet import log_to_google_sheet
from datetime import datetime

# Webhook 클라이언트 생성
webhook = WebhookClient(SLACK_WEBHOOK_URL)

def send_slack_message(text):
    """Slack으로 메시지 전송"""
    response = webhook.send(text=text)
    print(f"[Slack] 응답 상태: {response.status_code}")  # 200이면 성공

def process_detection(time, emotion, gender, age):
    danger_emotions = {"anger", "fear"}
    warning_prefix = "(주의)" if emotion.lower() in danger_emotions else ""

    msg = (
        f"[Door-Box] {warning_prefix}방문자가 인식되었습니다.\n"
        f"- 시각: {time}\n"
        f"- 감정: {emotion}\n"
        f"- 성별: {gender}\n"
        f"- 연령대: {age}"
    )

    # Slack + Sheets 실행
    send_slack_message(msg)
    log_to_google_sheet(time, emotion, gender, age, warning_prefix)


