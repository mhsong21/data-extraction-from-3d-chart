import cv2
from PIL import Image, ImageDraw
import pytesseract as ocr
import matplotlib.pyplot as plt
import numpy as np

def vert(igs, x_s, y_s, dy, origin):
    y0, x0 = origin

    box_igs = igs[y_s:y0+dy, x_s:x0]
    box_img = Image.fromarray(box_igs)
    boxes = get_boxes(box_img)
    draw_boxes(box_img, boxes)
    for box in boxes:
        print(box)

def get_boxes(img):
    # boxes = ocr.image_to_boxes(img, config='outputbase digits')
    boxes = ocr.image_to_boxes(img)
    boxes = boxes.split('\n')[:-1]
    boxes = [x.split() for x in boxes]
    return boxes


def draw_boxes(img, boxes):
    draw = ImageDraw.Draw(img)
    H, W = img.height, img.width
    for box in boxes:
        v = box[0]
        x0, y0, x1, y1, _ = list(map(int, box[1:]))
        draw.rectangle([x0, H-y0, x1, H-y1], outline='black')
    plt.imshow(img, cmap='gray')
    plt.show()

# thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def ocr_test():
    img = Image.open('./data/excel.png').convert('L')

    img = img.resize((img.width*3, img.height*3))
    igs = np.array(img)

    igs = thresholding(igs)

    t_img = Image.fromarray(igs)
    boxes = get_boxes(t_img)
    print(boxes)
    draw_boxes(t_img, boxes)


    # excel
    origin = (354*3, 64*3)
    x_s = 100
    y_s = 400
    dy = 100

    # matlab
    # origin = (339*3, 33*3)
    # x_s = 0
    # y_s = 0
    # dy = 50
    vert(igs, x_s, y_s, dy, origin)

    # boxes = get_boxes(img_new)

    # draw = ImageDraw.Draw(img_new)
    # H, W = img_new.height, img_new.width
    # for box in boxes:
    #     print(box)
    #     v = box[0]
    #     x0, y0, x1, y1, _ = list(map(int, box[1:]))
    #     draw.rectangle([x0, H-y0, x1, H-y1], outline='black')
    
