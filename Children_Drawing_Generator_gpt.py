import openai
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO
import os
import re
import logging

# OpenAI API Key 설정
openai.api_key = ''

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 텍스트 프롬프트를 기반으로 이미지 생성
def generate_images(prompt, num_images=1):
    response = openai.Image.create(
        prompt=prompt,
        n=num_images,
        size="1024x1024"
    )
    image_urls = [data['url'] for data in response['data']]
    return image_urls

# 고유한 파일 이름 생성 함수
def get_unique_filename(directory, base_name, extension):
    # 현재 디렉토리에서 같은 이름의 파일 중 가장 큰 숫자를 찾음
    max_number = 0
    for filename in os.listdir(directory):
        if filename.startswith(base_name) and filename.endswith(f'.{extension}'):
            match = re.search(r'_(\d+)\.', filename)
            if match:
                number = int(match.group(1))
                if number > max_number:
                    max_number = number
    # 새로운 파일 이름 생성
    unique_name = f"{base_name}_{max_number + 1}.{extension}"
    return unique_name

# 이미지 다운로드 및 저장
def download_and_save_images(image_urls, save_dir='C:/Users/user/Desktop/coding/Children_Drawing_Generator/images/gpt'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for idx, url in enumerate(image_urls):
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        
        # 고유한 파일 이름 생성
        base_name = 'image'
        extension = 'png'
        file_name = get_unique_filename(save_dir, base_name, extension)
        img.save(os.path.join(save_dir, file_name))

# 예시 프롬프트 사용
prompt = "a picture drawn by a child aged five to nine"
image_urls = generate_images(prompt, num_images=1)

# 생성된 이미지 다운로드 및 저장
download_and_save_images(image_urls)