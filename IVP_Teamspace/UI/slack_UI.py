from slack_sdk.webhook import WebhookClient
from slack_URL import SLACK_WEBHOOK_URL

# Webhook 클라이언트 생성
webhook = WebhookClient(SLACK_WEBHOOK_URL)

def send_slack_message(text):
    """Slack으로 메시지 전송"""
    response = webhook.send(text=text)
    print(f"[Slack] 응답 상태: {response.status_code}")  # 200이면 성공

def process_detection(time, emotion, gender, age):
    """Slack 알림 형식 구성 및 전송"""
    danger_emotions = {"anger", "fear"}

    # (주의) 조건
    warning_prefix = "(주의)" if emotion.lower() in danger_emotions else ""

    msg = (
        f"[Door-Box] {warning_prefix}방문자가 인식되었습니다.\n"
        f"- 시각: {time}\n"
        f"- 감정: {emotion}\n"
        f"- 성별: {gender}\n"
        f"- 연령대: {age}\n"
        f"- 클립 보기: (나중에 링크추가)"
    )
    send_slack_message(msg)

