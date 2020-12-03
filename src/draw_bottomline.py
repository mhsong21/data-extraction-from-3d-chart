import math
import numpy as np
import ocr
import Hcolor
import axis
import sys
import cv2
from PIL import Image
from box_info import LineInfo


# INPUT
# axis_points : list of (start,end). Ordered in z, 계열, rest (from axis.py)
# degrees : list of degrees of each axis. Ordered in z, 계열, rest (from axis.py)
# num_colors : number of colors i.e. number of 계열 (from ocr.py)
# delta_box : list of delta distance bewteen ticks. Ordered in z, 계열, rest(from ocr.py)


def draw_bottomline(axis_points, degrees, num_colors, delta_box):
    start = axis_points[2][0]
    end = axis_points[2][1]
    bottom_line = [(start, end)]
    for i in range(num_colors-1):
        dx = int(round(delta_box[i] * math.cos(-degrees[1]/180*math.pi)))
        dy = int(round(delta_box[i] * math.sin(-degrees[1]/180*math.pi)))
        jump = (dx, -dy)
        start = tuple(map(sum, zip(start, jump)))
        end = tuple(map(sum, zip(end, jump)))
        bottom_line.append((start, end))
        #print("delta_box, degree, jump: ", delta_box, degrees[1], jump)
    #for i in range(num_colors-1):
    #    start = tuple(map(sum, zip(start, jump)))
    #    end = tuple(map(sum, zip(end, jump)))
    #    bottom_line.append((start, end))

    return bottom_line


def main(filename, axis_points, degrees, dbox):
    img = cv2.imread('./data/' + filename + '.png', cv2.IMREAD_COLOR)
    result, background, num_colors, bar_colors, bottomline_interval = Hcolor.color_find(img)

    bottom_line = draw_bottomline(axis_points, degrees, num_colors, bottomline_interval)

    for lines in bottom_line:
        print(lines)
        img = cv2.line(img, lines[0], lines[1], (0, 0, 255), 1)
    
    axis.show_wait_destroy("img_with_bottomlines", img)

    lineinfos = [LineInfo(x[0] + x[1]) for x in bottom_line]
    return lineinfos
