import openai
import cv2
import numpy as np
import os

# OpenAI API Key 설정
openai.api_key = ''

def analyze_image(image_path):
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        return "Unable to load image."

    # 이미지 크기
    height, width, _ = image.shape

    # 주요 색상 추출
    data = image.reshape((-1, 3))
    data = np.float32(data)

    # K-means 클러스터링을 사용하여 주요 색상 찾기
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 5
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    major_colors = centers[labels.flatten()]
    
    unique, counts = np.unique(labels, return_counts=True)
    major_colors = centers[unique[np.argsort(-counts)]]

    color_descriptions = []
    for color in major_colors:
        color_descriptions.append(f"RGB({color[0]}, {color[1]}, {color[2]})")

    # 이미지의 형태와 구성 분석 (여기서는 간단히 객체의 개수로 설명)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    num_objects = len(contours)

    # 이미지 설명 생성
    description = f"The image is {width} pixels wide and {height} pixels tall. "
    description += f"The major colors in the image are: {', '.join(color_descriptions)}. "
    description += f"There are approximately {num_objects} distinct objects in the image, which includes various shapes and sizes. "
    description += "The objects vary in size and shape, with some being circular, some rectangular, and others irregular. The objects are spread across the entire image, indicating a complex and detailed scene."

    # 파일 이름과 함께 설명 콘솔 출력
    print(f"Image Description for {os.path.basename(image_path)}:\n{description}\n")
    return description

def evaluate_image_description(description, image_path):
    # 평가 기준에 따라 점수 매기기
    score_prompt = (
        f"Based on the following image description, evaluate it according to these criteria with strict standards:\n\n"
        f"Description: {description}\n\n"
        "Criteria:\n"
        "1. 섬세함 (details): Provide a score between 1 to 5, with 5 being extremely detailed and 1 being very basic. "
        "Consider how well-defined the objects are, the level of fine details, and the overall clarity of the image.\n"
        "2. 스토리텔링 능력 (storytelling ability): Provide a score between 1 to 5, with 5 having a strong, clear narrative and 1 having no narrative. "
        "Consider how the elements in the image interact to tell a story or convey a message.\n"
        "3. 객체의 다양성 (variety of objects): Provide a score between 1 to 5, with 5 being highly varied and 1 being very monotonous. "
        "Consider the range of different objects and their uniqueness.\n"
        "4. 공간 활용 (use of space): Provide a score between 1 to 5, with 5 utilizing space perfectly and 1 leaving much empty space. "
        "Consider how well the space is used and whether the objects are well-distributed.\n"
        "5. 표현력 (expressiveness): Provide a score between 1 to 5, with 5 being highly expressive and 1 being very plain. "
        "Consider the emotional impact and the visual appeal of the image."
    )

    score_response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an image evaluator."},
            {"role": "user", "content": score_prompt}
        ],
        max_tokens=250
    )

    scores = score_response.choices[0].message['content'].strip()

    # 파일 이름과 함께 점수 콘솔 출력
    print(f"Scores for {os.path.basename(image_path)} based on the description:\n{scores}")
    return scores


def main(image_path):
    # 이미지 분석 및 설명 생성
    description = analyze_image(image_path)
    print(f"Image Description for {os.path.basename(image_path)}:\n{description}\n")

    # 이미지 설명 평가
    scores = evaluate_image_description(description, image_path)
    print(f"Scores for {os.path.basename(image_path)} based on the description:\n{scores}")

if __name__ == "__main__":
    # 로컬 이미지 경로
    image_paths = [
        'C:/Users/user/Desktop/coding/Children_Drawing_Generator/Children_Drawing_Generator_gpt/images/image_119.png',  
        'C:/Users/user/Desktop/coding/Children_Drawing_Generator/Children_Drawing_Generator_gpt/images/image_120.png',  
        'C:/Users/user/Desktop/coding/Children_Drawing_Generator/Children_Drawing_Generator_gpt/images/image_137.png',  
        'C:/Users/user/Desktop/coding/Children_Drawing_Generator/Children_Drawing_Generator_gpt/images/image_140.png'  
    ]
    
    for image_path in image_paths:
        main(image_path)
