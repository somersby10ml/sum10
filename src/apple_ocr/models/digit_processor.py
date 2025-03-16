import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

from apple_ocr.preprocessing.image_preprocessing import preprocess_image
from apple_ocr.utils.model_manager import OCRModelManager


def get_transform():
    """학습 시 사용했던 전처리와 동일한 transform 반환"""
    return transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

def extract_digit_cells(image, cell_size=33, digit_region=(9, 23, 7, 19)):
    """이미지에서 숫자 영역을 포함하는 셀들을 추출"""
    # 이미지에서 숫자 영역이 포함된 부분을 crop (예시 좌표)
    cropped_image = image[74:400, 72:628]
    (height, width, _) = cropped_image.shape
    
    # 그리드 행/열 개수 계산
    rows = height // cell_size
    cols = width // cell_size
    
    cells = []
    for y in range(0, height, cell_size):
        for x in range(0, width, cell_size):
            # 각 셀 추출 (30x30 영역)
            cell = cropped_image[y:y+30, x:x+30]
            # 실제 숫자 영역: 세로 digit_region[0]~digit_region[1], 가로 digit_region[2]~digit_region[3]
            digit_img = cell[digit_region[0]:digit_region[1], digit_region[2]:digit_region[3]]
            cells.append(digit_img)
    
    rows = len(range(0, height, 33))
    cols = len(range(0, width, 33))
    return cells, rows, cols

def process_cells_to_tensors(cells, transform=None):
    """셀 이미지들을 텐서로 변환"""
    if transform is None:
        transform = get_transform()
    
    cell_tensors = []
    for cell in cells:
        # 기존에 정의한 preprocess_image 함수 사용
        pil_digit = preprocess_image(cell)
        tensor_img = transform(pil_digit)
        cell_tensors.append(tensor_img)
    
    return torch.stack(cell_tensors)

def predict_digits(batch_tensor, adjust_label=True):
    """텐서 배치를 모델에 입력하여 숫자 예측"""
    # 모델 가져오기
    model = OCRModelManager.get_model()
    device = OCRModelManager.get_device()
    
    # 모델에 입력
    batch_tensor = batch_tensor.to(device)
    with torch.no_grad():
        outputs = model(batch_tensor)
    
    # 예측: 각 셀의 예측 결과
    predictions = outputs.argmax(dim=1).cpu().numpy()
    
    # ImageFolder 매핑 때문에 실제 숫자 조정 (+1)
    if adjust_label:
        predictions = predictions + 1
    
    return predictions

def process_image_grid(image_path, cell_size=33, digit_region=(9, 23, 7, 19)):
    """이미지 경로로부터 숫자 그리드 추출 및 예측"""
    image = cv2.imread(image_path)
    cells, rows, cols = extract_digit_cells(image, cell_size, digit_region)
    transform = get_transform()
    batch_tensor = process_cells_to_tensors(cells, transform)
    predictions = predict_digits(batch_tensor)
    
    # 그리드 형태로 재구성
    result_grid = predictions.reshape((rows, cols))
    return result_grid