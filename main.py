from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from concurrent.futures import ThreadPoolExecutor

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
            canvas = self.driver.find_element(By.ID,'canvas')
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
        canvas = self.driver.find_element(By.ID, 'canvas')
        canvas_left = (canvas.size['width'] / 2 - 70) 
        canvas_top = (canvas.size['height'] / 2 - 70)


        start = (canvas_left, canvas_top)
        x =  (33 * x) - start[0]
        y =  (33 * y) - start[1] 
        wIndex =  (width * 33) 
        hIndex =  (height * 33)
        action = ActionChains(self.driver)

        action.move_to_element_with_offset(canvas, x, y)
        action.click_and_hold()  
        action.move_by_offset(wIndex, hIndex)
        
        action.pause(0.01)
        action.release().perform()

    def close(self):
        self.driver.quit()

if __name__ == '__main__':

    Game = AppleGame()
    npimage = Game.getImageFromCanvas()  # get image
    numbary_array = Game.ImageToNumberArray(npimage)  # ocr (get number from image)

    calc = Sum10()
    while True:
        x, y, w, h = calc.calc(numbary_array)
        if x == 0 and y == 0 and w == 0 and h == 0:
            break

        Game.Drag(x, y, w, h)
    print('finish')
    Game.close()


