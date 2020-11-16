import numpy as np
from PIL import Image
import pytesseract as ocr
from enum import Enum
import cv2
import matplotlib.pyplot as plt


class LineType(Enum):
    VERTICAL = 1
    HORIZONTAL = 2
    NORMAL = 3


class LineInfo:
    def __init__(self, axis, delta_threshold=3):
        x0, y0, x1, y1 = tuple(map(float, axis))
        self.start = [x0, y0]
        self.end = [x1, y1]

        if (x1 - x0) <= delta_threshold:
            a = -1
            b = (x0+x1)/2
            self.type = LineType.VERTICAL
        elif (y1 - y0) <= delta_threshold:
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

        if self.type == LineType.VERTICAL:
            dist = np.abs(xc - x0)
        elif self.type == LineType.HORIZONTAL:
            dist = np.abs(yc - y0)
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

    def ocr(self, igs, scalar=5):
        test_imgs = []
        x0, y0, x2, y2 = self.box.astype(int)
        dp = 0

        igs_bin = thresholding(igs)

        box_igs = igs[y0:y2, x0:x2+dp]
        box_igs_bin = igs_bin[y0:y2, x0:x2+dp]

        box_img = Image.fromarray(box_igs)
        W = box_img.width
        H = box_img.height
        test_imgs.append(box_img)

        binary_img = Image.fromarray(box_igs_bin)
        test_imgs.append(binary_img)

        resize = box_img.resize((W*scalar, H*scalar))
        test_imgs.append(resize)

        resize_binary = binary_img.resize((W*scalar, H*scalar))
        test_imgs.append(resize_binary)

        binary_resize = thresholding(np.array(resize))
        test_imgs.append(Image.fromarray(binary_resize))

        # fig, axes = plt.subplots(nrows=2, ncols=3)
        # axes[0][0].imshow(test_imgs[0], cmap='gray')
        # axes[0][1].imshow(test_imgs[1], cmap='gray')
        # axes[0][2].imshow(test_imgs[2], cmap='gray')
        # axes[1][0].imshow(test_imgs[3], cmap='gray')
        # axes[1][1].imshow(test_imgs[4], cmap='gray')
        # plt.show()

        config_str = '--psm 7 -c tessedit_char_whitelist=0123456789'

        for i, img in enumerate(test_imgs):
            result = ocr.image_to_string(img, config=config_str)
            try:
                result = int(result)
            except ValueError:
                result = None

            if result is not None:
                return result
