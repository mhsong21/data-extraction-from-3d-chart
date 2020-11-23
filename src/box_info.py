import numpy as np
from PIL import Image
import pytesseract as ocr
from enum import Enum
import cv2


class LineType(Enum):
    VERTICAL = 1
    HORIZONTAL = 2
    NORMAL = 3


OUTLINE_TH = 5


class LineInfo:
    def __init__(self, axis, delta_threshold=3):
        x0, y0, x1, y1 = tuple(map(float, axis))
        self.start = [x0, y0]
        self.end = [x1, y1]
        self.length = np.sqrt((x1-x0)**2 + (y1-y0)**2)

        if np.abs(x1 - x0) <= delta_threshold:
            a = -1
            b = (x0+x1)/2
            self.type = LineType.VERTICAL
        elif np.abs(y1 - y0) <= delta_threshold:
            a = 0
            b = (y0+y1)/2
            self.type = LineType.HORIZONTAL
        else:
            a = (y1 - y0) / (x1 - x0)
            b = -a*(x0+x1)/2 + (y0+y1)/2
            self.type = LineType.NORMAL

        # y = ax + b
        self.equation = [a, b]

    def __str__(self):
        a, b = self.equation
        return "{} y = {}x + {}, ({}) -> ({})".format(
            self.type, a, b, self.start, self.end)

    def line_dist(self, box):
        xc, yc = box.pos
        a, b = self.equation
        x0, y0 = self.start
        x1, y1 = self.end

        if self.type == LineType.VERTICAL:
            tmp = max(np.abs(yc - y1), np.abs(yc - y0))
            if tmp > OUTLINE_TH + self.length:
                dist = 1000
            else:
                dist = np.abs(xc - x0)
        elif self.type == LineType.HORIZONTAL:
            tmp = max(np.abs(xc - x1), np.abs(xc - x0))
            if tmp > OUTLINE_TH + self.length:
                dist = 1000
            else:
                dist = np.abs(yc - y0)
        else:
            px = (xc + a*yc - a*b) / (a**2+1)
            py = a*px + b
            tmp = max(np.sqrt((px-x0)**2 + (py-y0)**2),
                      np.sqrt((px-x1)**2 + (py-y1)**2))
            if tmp > OUTLINE_TH + self.length:
                dist = 1000
            else:
                dist = np.abs(a*xc - yc + b) / np.sqrt(a**2 + 1)

        return dist


def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


class BoxInfo:
    def __init__(self, raw_data=None, pos=None, boxlen=None):
        if raw_data is None:
            xc, yc = pos
            lx, ly = boxlen
            dx = lx / 2
            dy = ly / 2
            box = np.array([xc - dx, yc - dy, xc + dx, yc + dy])
            self.box = box
            self.pos = pos
            return

        raw_data = np.array(raw_data).astype(float)
        x0, y0, x1, y1, x2, y2, x3, y3 = raw_data

        xc = np.average(raw_data[::2])
        yc = np.average(raw_data[1::2])

        self.box = np.array([x0, y0, x2, y2])
        self.pos = np.array([xc, yc])

    def length(self):
        x0, y0, x2, y2 = self.box
        return [x2-x0, y2-y0]

    def ocr(self, igs, isNumeric=False, scalar=5):
        test_imgs = []
        x0, y0, x2, y2 = self.box.astype(int)
        if x0 < 0 or y0 < 0 or x2 < 0 or y2 < 0:
            return None, False

        igs_bin = thresholding(igs)

        box_igs = igs[y0:y2, x0:x2]
        box_igs_bin = igs_bin[y0:y2, x0:x2]

        box_img = Image.fromarray(box_igs)
        W = box_img.width
        H = box_img.height

        if W <= 0 or H <= 0:
            return None, False

        test_imgs.append(box_img)

        binary_img = Image.fromarray(box_igs_bin)
        test_imgs.append(binary_img)

        resize = box_img.resize((W*scalar, H*scalar))
        test_imgs.append(resize)

        resize_binary = binary_img.resize((W*scalar, H*scalar))
        test_imgs.append(resize_binary)

        binary_resize = thresholding(np.array(resize))
        test_imgs.append(Image.fromarray(binary_resize))

        # cv2.imshow('asd', np.array(test_imgs[0]))
        # cv2.moveWindow('asd', 500, 0)
        # cv2.waitKey(0)
        # cv2.destroyWindow('asd')

        num_config = '--psm 7 -c tessedit_char_whitelist=0123456789'
        text_config = '--psm 7'

        for i, img in enumerate(test_imgs):
            result = ocr.image_to_string(img, config=num_config)
            try:
                result = int(result)
                isNumeric = True
            except ValueError:
                if isNumeric:
                    return None, False
                result = ocr.image_to_string(img, config=text_config).rstrip()

            if result is not None:
                return result, isNumeric

        return None, False
