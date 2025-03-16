import cv2
import numpy as np
from PIL import Image


def remove_red_background(img):
    # 이미지가 BGR 포맷이라고 가정 (OpenCV 기본)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 빨간색 범위 설정 (HSV에서 빨간색은 두 영역으로 나뉨)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # 빨간 영역을 검은색으로 변경
    img[red_mask > 0] = [0, 0, 0]
    return img

def preprocess_image(digit_img):
    processed = remove_red_background(digit_img)
    # cvtColor의 상수는 cv2.COLOR_BGR2GRAY (오타 주의)
    gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    # PIL 이미지로 변환 후 transform 적용 (이미지 크기는 14x12)
    pil_digit = Image.fromarray(thresh)
    return pil_digit