import cv2
import numpy as np
import matplotlib.pyplot as plt


class CarPlateLocation:

    def __init__(self, img_path):

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        self.origin_img = img.copy()

        # resize image
        width = 500
        height = width * img.shape[0] / img.shape[1]
        self.img = cv2.resize(img, (width, int(height)),
                              interpolation=cv2.INTER_CUBIC)

    def _find_rectangle(self, contour):
        """
        find a rectagnle for a contour
        """
        x, y = [], []

        for p in contour:
            x.append(p[0][0])
            y.append(p[0][1])

        return [min(x), min(y), max(x), max(y)]

    def _detect_by_color(self, blocks):
        """
        transfrom to hsv space and detect plate by the white bg and blue number
        """

        lower_blue = np.array([110, 100, 100], dtype=np.uint8)
        upper_blue = np.array([130, 255, 255], dtype=np.uint8)
        sensitivity = 135
        lower_white = np.array([0, 0, 255 - sensitivity])
        upper_white = np.array([255, sensitivity, 255])

        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        proportion_list = list()

        for b in blocks:
            # print(b)
            block = hsv[b[1]:b[3], b[0]:b[2]]

            blue_mask = cv2.inRange(block, lower_blue, upper_blue)
            white_mask = cv2.inRange(block, lower_white, upper_white)

            # mask = (blue_mask | white_mask)
            blue_value = np.sum(blue_mask / 255)
            white_value = np.sum(white_mask / 255)

            total = np.size(block)
            valid = blue_value * 0.7 + white_value * 0.3

            proportion = valid / total
            proportion_list.append(proportion)

        sorted_index = np.argsort(- np.array(proportion_list))

        # return the maximum or can return a list then detect by other means
        return blocks[sorted_index[0]]

    def _find_all_blocks(self, img, wh_factor=1.5):
        """
        find all blocks in the processed image
        :param img: the processed image
        :param wh_factor: the scale factor for width and height
        :return: all blocks in the image, using rect to represent
        """
        # obtain contours
        contours, hierarchy = cv2.findContours(
            img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        blocks = list()
        for c in contours:
            block = self._find_rectangle(c)
            width = block[2] - block[0]
            height = block[3] - block[1]

            if width > 20 and height > 10 and width > height * wh_factor:
                blocks.append(block)

        return blocks

    def process(self):
        """
        process image to locate the plate
        """

        # denoising
        blur_img = cv2.GaussianBlur(self.img, (5, 5), 0)
        # gray scale
        gray_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2GRAY)
        # equalize
        equ_img = cv2.equalizeHist(gray_img)
        # # binary
        # bi_img = cv2.adaptiveThreshold(
        #     equ_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, 0)
        # edge detection
        canny_img = cv2.Canny(equ_img, 150, 200)

        # ------------------- Important Part -------------------
        # closing
        kernel = np.ones((10, 30), np.uint8)
        closing_img = cv2.morphologyEx(canny_img, cv2.MORPH_CLOSE, kernel)
        # opening
        opening_img = cv2.morphologyEx(closing_img, cv2.MORPH_OPEN, kernel)
        # opening again
        kernel = np.ones((20, 40), np.uint8)
        opening_img1 = cv2.morphologyEx(opening_img, cv2.MORPH_OPEN, kernel)
        # -------------------------------------------------------

        blocks = self._find_all_blocks(opening_img1)

        return self._detect_by_color(blocks)

    def show_result(self, block):
        print(block)
        target = self.img[block[1]:block[3], block[0]:block[2]]

        plt.imshow(cv2.cvtColor(target, cv2.COLOR_BGR2RGB))
        plt.show()
        # cv2.imshow('target', target)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def cut_plate(self, block, mode=1):
        """
        using the foregound to cut and obtain better block area
        :param mode: 1 for simple way. 2 for a slight bette way.
        :param block: the foreground area
        :return: the rectifed block
        """
        img = self.img.copy()
        rect = block.copy()
        # change to x_start, y_start, width, height
        rect[2] = rect[2] - rect[0]
        rect[3] = rect[3] - rect[1]
        rect = tuple(rect)

        mask = np.zeros(img.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)  # bg model
        fgd_model = np.zeros((1, 65), np.float64)  # fd model

        cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

        mask1 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        y_index, x_index = np.where(mask1 > 0)

        if mode == 1:
            target = [min(x_index), min(y_index), max(x_index), max(y_index)]
        else:
            blocks = self._find_all_blocks(mask1, 2)
            if len(blocks) == 0:
                target = [min(x_index), min(y_index), max(x_index), max(y_index)]
            elif len(blocks) == 1:
                target = blocks[0]
            else:
                target = self._detect_by_color(blocks)

        return target

    def get_cropped_plate(self, block):
        """
        Crop the license plate from the original (un-resized) image
        :param block: plate coordinates
        :return: the plate array cropped according to the blocks
        """
        origin_block = []
        for b in block:
            b *= (self.origin_img.shape[1] / 500)
            origin_block.append(int(b))

        target = self.origin_img[origin_block[1]:origin_block[3], origin_block[0]:origin_block[2]]
        # cv2.imshow('Cropped plate', target)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return target


if __name__ == '__main__':
    test = CarPlateLocation('./images/raw/YLT33W.jpg')

    inital_block = test.process()

    # for some plates, the cutting is not good
    target_block = test.cut_plate(inital_block)
    test.show_result(target_block)