import gspread
from oauth2client.service_account import ServiceAccountCredentials
import os

# 구글 인증 설정
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
cred_path = os.path.join(os.path.dirname(__file__), "credentials.json")
creds = ServiceAccountCredentials.from_json_keyfile_name(cred_path, scope)
client = gspread.authorize(creds)
worksheet = client.open("[Door-Box] 방문 기록").worksheet("감지기록")

def log_to_google_sheet(time, emotion, gender, age, warning_prefix):
    row = [time, emotion, gender, age, warning_prefix]
    try:
        worksheet.append_row(row)
        print(f"[Google Sheets] 기록 완료: {row}")
    except Exception as e:
        print(f"[Google Sheets] 기록 실패: {e}")
