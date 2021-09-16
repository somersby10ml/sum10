import cv2
import numpy


class Ocr:
    def __init__(self):
        self.numList = []

    def Init(self, image_path):
        number_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        (height, width) = number_image.shape

        for i in range(0, width, 12):
            self.numList.append(number_image[0:height, i:i + 12])
        return

    def GetNumber(self, src):
        for idx, val in enumerate(self.numList):
            tempDiff = cv2.subtract(src, val)
            all_zeros = not numpy.any(tempDiff)
            if all_zeros:
                return idx + 1
        return -1

