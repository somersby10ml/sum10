from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import NoSuchElementException

import numpy
import base64
import cv2

from sum10 import Sum10
from util.img import Ocr


class AppleGame:
    def __init__(self):
        """Start web driver"""
        chrome_options = webdriver.ChromeOptions()
        chrome_options.debugger_address = "127.0.0.1:9222"
        self.driver = webdriver.Chrome(options=chrome_options)
        self.driver.implicitly_wait(10)

        self.ocr = Ocr()
        self.ocr.Init("./image/number.png")
        # self.driver: selenium.webdriver = None

    def getImageFromCanvas(self) -> numpy.ndarray:
        """
        canvas image to numpy array
        :return: cv2 image array
        """
        try:
            canvas = self.driver.find_element_by_id('canvas')
            canvas_base64 = self.driver.execute_script("return arguments[0].toDataURL('image/png').substring(21);",                                  canvas)
            canvas_png = base64.b64decode(canvas_base64)
            nparr = numpy.frombuffer(canvas_png, numpy.uint8)
            img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img_np
        except NoSuchElementException as ex:
            print(ex.msg)
        return numpy.array([])

    def ImageToNumberArray(self, canvas_image) -> numpy.ndarray:
        """
        image to number
        :param canvas_image: canvas image
        :return: number 2d array
        """
        number_array = numpy.empty((0, 17), int)

        dst = cv2.cvtColor(canvas_image, cv2.COLOR_BGR2GRAY)
        dst = dst[74: 400, 72: 628]

        (height, width) = dst.shape
        number_line_array = []
        for y in range(0, height, 33):
            for x in range(0, width, 33):
                dst2 = dst[y:y + 33, x:x + 33]
                dst2 = dst2[9: 23, 7: 19]
                num = self.ocr.GetNumber(dst2)
                number_line_array.append(num)

            number_array = numpy.append(number_array, numpy.array([[*number_line_array]]), axis=0)
            number_line_array.clear()

        return number_array

    def Drag(self, x, y, width, height) -> None:
        """
        Selenium drag in cavas apple
        :param x: x index
        :param y: y index
        :param width: count
        :param height: count
        :return: None
        """
        canvas = self.driver.find_element_by_id('canvas')
        action = ActionChains(self.driver)

        start = (70, 71)

        x = start[0] + (33 * x)
        y = start[1] + (33 * y)

        wIndex = width
        hIndex = height

        action.move_to_element_with_offset(canvas, x, y)
        action.click_and_hold()
        action.move_by_offset(33 * wIndex, 33 * hIndex)
        action.pause(0.001)
        action.release().perform()

    def close(self):
        self.driver.quit()


if __name__ == '__main__':

#     a = numpy.array([[6, 0, 8, 0, 0, 0, 0, 5, 3, 0, 5, 9, 0, 0, 0, 9, 7]
# , [0, 0, 5, 0, 6, 7, 9, 0, 0, 0, 0, 9, 3, 0, 0, 5, 0]
# , [0, 6, 8, 0, 8, 6, 6, 9, 0, 0, 6, 5, 8, 8, 0, 0, 0]
# , [0, 0, 7, 0, 6, 9, 0, 5, 0, 0, 9, 4, 8, 1, 0, 0, 0]
# , [0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 7, 8, 6, 2, 0, 6, 0]
# , [0, 8, 0, 0, 5, 0, 0, 1, 0, 8, 4, 0, 0, 9, 0, 8, 0]
# , [0, 1, 0, 0, 0, 0, 8, 5, 0, 6, 2, 7, 8, 5, 0, 0, 1]
# , [0, 0, 0, 0, 9, 0, 4, 0, 0, 8, 9, 9, 0, 6, 0, 8, 4]
# , [0, 0, 0, 0, 9, 0, 2, 0, 0, 9, 0, 0, 0, 0, 0, 8, 4]
# , [8, 0, 0, 1, 4, 0, 2, 7, 0, 2, 0, 9, 5, 2, 0, 4, 3]])
#
#     calc = Sum10()
#     aa = calc.calc(a)
#     # print(aa)
#     print(a)
#     exit()

    Game = AppleGame()
    npimage = Game.getImageFromCanvas()  # get image
    numbary_array = Game.ImageToNumberArray(npimage)  # ocr (get number from image)

    calc = Sum10()
    while True:
        x, y, w, h = calc.calc(numbary_array)
        if x == 0 and y == 0 and w == 0 and h == 0:
            break

        Game.Drag(x, y, w, h)
    print(numbary_array)
    print('finish')
    Game.close()


