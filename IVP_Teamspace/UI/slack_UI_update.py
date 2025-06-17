from slack_UI import send_slack_message 
from google_sheet import log_to_google_sheet
from datetime import datetime

def process_detection(time, emotion, gender, age):
    danger_emotions = {"anger", "fear"}
    warning_prefix = "(주의)" if emotion.lower() in danger_emotions else ""

    msg = (
        f"[Door-Box] {warning_prefix}방문자가 인식되었습니다.\n"
        f"- 시각: {time}\n"
        f"- 감정: {emotion}\n"
        f"- 성별: {gender}\n"
        f"- 연령대: {age}\n"
        f"- 클립 보기: (나중에 링크 추가)"
    )

    # Slack + Sheets 실행
    send_slack_message(msg)
    log_to_google_sheet(time, emotion, gender, age, warning_prefix)

if __name__ == "__main__":
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    emotion = "anger"
    gender = "남자"
    age = "30대"

    process_detection(now, emotion, gender, age)

