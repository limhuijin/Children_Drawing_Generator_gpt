import openai
import requests
from PIL import Image
from io import BytesIO
import os
import re
import logging
import time

# OpenAI API Key 설정
openai.api_key = ''

# 로깅 설정 (콘솔 및 파일에 로그 저장)
log_filename = 'image_generation.log'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(), logging.FileHandler(log_filename)])

# 텍스트 프롬프트를 기반으로 이미지 생성
def generate_images(prompt, num_images, retries=3):
    for attempt in range(retries):
        try:
            logging.info(f"Generating {num_images} images for prompt: '{prompt}'")
            response = openai.Image.create(
                prompt=prompt,
                n=num_images,
                size="1024x1024"
            )
            image_urls = [data['url'] for data in response['data']]
            return image_urls
        except openai.error.APIError as e:
            logging.error(f"OpenAI API error: {e}")
            if attempt < retries - 1:
                logging.info(f"Retrying... (Attempt {attempt + 2}/{retries})")
                time.sleep(5)
            else:
                logging.error("Failed to generate images after several attempts.")
                raise

# 고유한 파일 이름 생성 함수
def get_unique_filename(directory, base_name, extension):
    max_number = 0
    for filename in os.listdir(directory):
        if filename.startswith(base_name) and filename.endswith(f'.{extension}'):
            match = re.search(r'_(\d+)\.', filename)
            if match:
                number = int(match.group(1))
                if number > max_number:
                    max_number = number
    unique_name = f"{base_name}_{max_number + 1}.{extension}"
    return unique_name

# 이미지 다운로드 및 저장
def download_and_save_images(image_urls, save_dir='images'):
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

        # 로그 남기기
        logging.info(f"Saved image as {file_name}")

# 텍스트 파일에서 프롬프트 읽기
prompt = '''
Crayon drawing, 5-year-old child, simple house, stick figures, three colors, basic shapes, simple and straightforward

'''

num_images = 1

try:
    image_urls = generate_images(prompt, num_images)
    # 생성된 이미지 다운로드 및 저장
    download_and_save_images(image_urls)
except Exception as e:
    logging.error(f"Failed to generate and save images: {e}")
