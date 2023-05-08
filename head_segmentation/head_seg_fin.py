import os
import cv2
import numpy as np
from head_segmentation.segmentation_pipeline import HumanHeadSegmentationPipeline

# 인스턴스 생성 및 파일 경로 설정
pipeline = HumanHeadSegmentationPipeline()
input_folder = "C:/images/img"
output_folder = "C:/images/img2"

# 폴더 생성 (존재하지 않는 경우)
os.makedirs(output_folder, exist_ok=True)

# 이미지 파일 처리
for file_name in os.listdir(input_folder):
    file_path = os.path.join(input_folder, file_name)

    if os.path.isfile(file_path) and file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        try:
            # 이미지 로드 및 색상 공간 변환
            image = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)

            # 머리 부분 분리
            segmentation_map = pipeline.predict(image)

            # 분리된 머리 부분을 이미지에 적용
            segmented_region = image * cv2.cvtColor(segmentation_map, cv2.COLOR_GRAY2RGB)

            # 배경 흰색으로 채우기
            background = (1 - segmentation_map) * 255  # 배경 부분을 흰색으로 만들기
            background = cv2.cvtColor(background, cv2.COLOR_GRAY2RGB)
            result_image = segmented_region.astype(np.uint8) + background.astype(np.uint8)

            # 결과 이미지 저장
            output_path = os.path.join(output_folder, file_name)
            cv2.imwrite(output_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))

print("모든 이미지 처리가 완료되었습니다.")
