import torch
import torchvision.transforms as transforms
from PIL import Image
import os

AGE_MODEL_PATH = "./models/pt/mobilenetv3_age.pth"
GENDER_MODEL_PATH = "./models/pt/mobilenetv3_gender.pth"

AGE_CLASSES = ["9세 이하", "10대", "20대", "30대", "40대", "50대", "60대", "70세 이상"]
GENDER_CLASSES = ["남자", "여자"]

# transform 정의
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

# 모델 로드 함수
def load_model(path):
    model = torch.load(path, map_location=torch.device('cpu'))
    model.eval()
    return model

# 성별 및 연령 분류 함수
def infer_gender_age(image_path):
    transform = get_transform()
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0)

    gender_model = load_model(GENDER_MODEL_PATH)
    age_model = load_model(AGE_MODEL_PATH)

    with torch.no_grad():
        gender_out = gender_model(tensor)
        age_out = age_model(tensor)

        gender_idx = gender_out.argmax().item()
        age_idx = age_out.argmax().item()

        gender = GENDER_CLASSES[gender_idx] if gender_idx < len(GENDER_CLASSES) else "Unknown"
        age = AGE_CLASSES[age_idx] if age_idx < len(AGE_CLASSES) else "Unknown"

    return gender, age
