import json
import os
from datetime import datetime

RESULT_PATH = "./outputs/result.json"

# 감정, 성별, 연령대, 시간 정보를 JSON으로 저장
def save_result(emotion, gender, age):
    result = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "emotion": emotion,
        "gender": gender,
        "age_range": age
    }

    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    with open(RESULT_PATH, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return RESULT_PATH
