import math
import numpy as np
import ocr
import Hcolor
import axis
import sys
import cv2
from PIL import Image

####INPUT
# axis_points : list of (start,end). Ordered in z, 계열, rest (from axis.py)
# degrees : list of degrees of each axis. Ordered in z, 계열, rest (from axis.py)
# num_colors : number of colors i.e. number of 계열 (from ocr.py)
# delta_box : list of delta distance bewteen ticks. Ordered in z, 계열, rest(from ocr.py)

def draw_bottomline(axis_points, degrees, num_colors, delta_box):
    dx = (int) (delta_box[1] * math.cos(-degrees[1]/180*math.pi))
    dy = (int) (delta_box[1] * math.sin(-degrees[1]/180*math.pi))
    jump = (dx, -dy)
    print("delta_box, degree, jump: ",delta_box[1], degrees[1], jump )
    start = axis_points[2][0]
    end = axis_points[2][1]
    bottom_line = [(start,end)]
    for i in range(num_colors-1):
        start = tuple(map(sum,zip(start, jump)))
        end = tuple(map(sum,zip(end, jump)))
        bottom_line.append((start,end))
    
    return bottom_line

def main():
    filename = 'Matlab3'
    img_in = Image.open('../data/matlab/' + filename + '.png').convert('RGB')
    img = np.array(img_in)
    result, background, num_colors, bar_colors = Hcolor.color_find(img)
    axis_points, degrees = axis.main("../data/matlab/", filename + '.png', '../result/' )
    delta_box = ocr.tick_to_value('../data/matlab/' + filename + '.png', '../CRAFT-pytorch/result/res_Matlab3.txt', (axis_points[0], axis_points[1]), (4,5))
    bottom_line = draw_bottomline(axis_points, degrees, num_colors, (0,66,0))
    
    for lines in bottom_line:
        print(lines)
        img = cv2.line(img, lines[0], lines[1], (0, 0, 255), 1)
    axis.show_wait_destroy("img_with_bottomlines", img)

if __name__ == '__main__':
    main()
    