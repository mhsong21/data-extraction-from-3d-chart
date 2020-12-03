import cv2
import numpy as np
from PIL import Image
import matplotlib
import sys
import scipy.ndimage as ndimage
import Hcolor
import os

def template_matching(img, res, template, min_interval, vertex):
    ####
    # codes for matching the template
    ####

    src = res
    dst = img.copy()
    result = cv2.matchTemplate(src, template, cv2.TM_CCOEFF_NORMED)
    # print(min_interval)
    # using threshold
    threshold = 0.9
    loc = np.where(result >= threshold)
    
    while len(loc[0]) == 0:
        threshold = threshold - 0.03
        loc = np.where(result >= threshold)
    
    # elements of coord[0] are x-coordinates, coord[1] are y-coordinates
    coord = np.zeros((2, len(loc[0])))
    
    i = 0
    for pt in zip(*loc[::-1]):
        coord[0][i] = pt[0]
        coord[1][i] = pt[1]
        i += 1
    coord = coord.astype(int)
    # np.sort(coord.view('i8,i8'), order=['f1'], axis=0).view(np.int)
    # Still, coord has a problem that it has nearby(overlapping) templates.
    # get rid of nearby templates and save it on the coordinate

    coordinate = coord

    for i in range(coord.shape[1]):
        for j in range(coord.shape[1]):
            if coord[0][j] - 3 <= coord[0][i] <= coord[0][j] + 3:
                if result[coord[1][j]][coord[0][j]] > result[coord[1][i]][coord[0][i]]:
                    coordinate[:, i] = coord[:, j]
    coordinate = np.unique(coordinate, axis=1)

    if coordinate.shape[1] == 0:
        coordinate = np.zeros((0,0))
        return dst, coordinate, min_interval

    if coordinate.shape[1] > 1:
        for i in range(coordinate.shape[1]-1):
            if (coordinate[0][i+1] - coordinate[0][i]) < min_interval:
                min_interval = coordinate[0][i+1] - coordinate[0][i]
    original_min_interval = min_interval

    # to avoid the situation if one of every two or three heads are detected. 
    min_interval = min_interval // 6
    
    if min_interval == 0:
        min_interval = 5
    x_coord = coordinate[0][0]

    threshold = 0.25
    if min_interval < 5000:
        while x_coord+vertex[0] < result.shape[1]:
            for i in range(x_coord-vertex[0]//2, x_coord+vertex[0]//2):
                if np.amax(result[:, i]) > threshold:
                    y_coord = np.argmax(result[:, i])
                    coordinate = np.append(
                        coordinate, [[i], [y_coord]], axis=1)
            x_coord += min_interval    
            
        x_coord = coordinate[0][0]
        while x_coord-vertex[0] > 0:
            for i in range(x_coord-vertex[0]//2, x_coord+vertex[0]//2):
                if np.amax(result[:, i]) > threshold:
                    y_coord = np.argmax(result[:, i])
                    coordinate = np.append(
                        coordinate, [[i], [y_coord]], axis=1)
            x_coord -= min_interval
    coord = coordinate

    for i in range(coord.shape[1]):
        for j in range(coord.shape[1]):
            if coord[0][j] - min(template.shape[1], min_interval*5) <= coord[0][i] <= coord[0][j] + min(template.shape[1], min_interval*5):
                if result[coord[1][j]][coord[0][j]] >= result[coord[1][i]][coord[0][i]]:
                    coordinate[:, i] = coord[:, j]
    coordinate = np.unique(coordinate, axis=1)

    if coordinate.shape[1] > 1:
        for i in range(coordinate.shape[1]-1):
            if (coordinate[0][i+1] - coordinate[0][i]) < original_min_interval and (coordinate[0][i+1] - coordinate[0][i]) > min(template.shape[1], min_interval):
                original_min_interval = coordinate[0][i+1] - coordinate[0][i]
    min_interval = original_min_interval
    
    # dst is the image which is marked where the template is matched
    h, w, _ = template.shape
    for i in range(coordinate.shape[1]):
        cv2.rectangle(dst, (coordinate[0][i], coordinate[1][i]),
                      (coordinate[0][i] + w, coordinate[1][i] + h), (0, 0, 0), 2)
    
    for i in range(coordinate.shape[1]):
        coordinate[:,i] += vertex

    return dst, coordinate, min_interval


def template_finding(img, res, bar_colors, maxsize_template):
    ####
    # codes for finding a template (head)
    ####
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # color_filtered_img = cv2.imread("./Matlab8/color_0.png", cv2.IMREAD_GRAYSCALE)
    # assume the deteceted color gray scale value
    colorValue_H = bar_colors[0]
    colorValue_S = bar_colors[1]

    colorArea = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    #colorArea = cv2.medianBlur(colorArea, 3)
    #colorArea = np.where(hsv[:, :, 0] == colorValue_H,
    #                     hsv[:, :, 0], 0)  # 진환이가 만들어준 남은 이미지들로 바꾸기
    

    x = 0
    y = 0
    # no legend, start from index 50 to avoid matching to chart title
    for j in range(0, colorArea.shape[0]):
        #if np.amax(colorArea[j, :]) == colorValue_H:
        if np.amax(colorArea[j, :]) != 0:
            x = np.where(colorArea[j, :] != 0)
            x = x[0][0]
            y = j
            break
    #print(x,y)
    # the coordinate of head top point is saved in x, y
    # x -= 1
    # y -= 1

    # for calculating the slope of x-axis and y-axis of chart,
    # I put manually the stating and end points of axes
    # x-axis: x_start / x_end is the starting / ending points of x-axis respectively.
    # y-axis: y_start / y_end is the starting / ending points of y-axis respectively.
    
    x_temp = x
    y_temp = y
    x_max = x - 1

    for i in range(0,colorArea.shape[0]):
        x_temp = x
        y_temp = y + i
        while colorArea[y_temp][x_temp] != 0:
            x_temp += 1
        if x_temp - 1 >= x_max:
            x_max = x_temp - 1
        else:
            break 

    x_alongy = x_max
    y_alongy = y_temp


    x_temp = x
    y_temp = y
    x_min = x + 1

    for i in range(0,colorArea.shape[0]):
        x_temp = x
        y_temp = y + i
        while colorArea[y_temp][x_temp] != 0:
            x_temp -= 1
        if x_temp + 1 <= x_min:
            x_min = x_temp + 1
        else:
            break 

    x_alongx = x_min
    y_alongx = y_temp

    template_coord = np.array([[x, y], [x_alongy, y_alongy], [x_alongx, y_alongx], [
                              x_alongx + x_alongy - x, y_alongx + y_alongy - y]])
    template = res[np.amin(template_coord[:, 1]): np.amax(
        template_coord[:, 1]) + 1, np.amin(template_coord[:, 0]): np.amax(template_coord[:, 0]) + 1, :]
    if template.shape[1] < maxsize_template.shape[1] or template.shape[0] > res.shape[0] * 0.2:
        template = np.where(maxsize_template[:,:] == [0,0,0], [0,0,0], res[y][x])
    template = template.astype(np.uint8)
    #np.all([maxsize_template[:,:,0] == 0,maxsize_template[:,:,1] == 0,maxsize_template[:,:,2] == 0]) 
    vertex = np.array([x_alongy - x, template.shape[0]])
    #print(template.shape)

    # template: head template image
    # vertex: coordinate difference between the bottom vertex and the template left-top vertex
    return template, vertex
    
def edge_enhance(igs):
    new_igs = np.copy(igs)
    kernel = np.ones((3,3), np.uint8)
    new_igs = cv2.erode(igs, kernel)

    return new_igs

def run(filename):
    if not os.path.isdir('./temp'):
        os.mkdir('./temp')
    folder = './temp/' + filename
    if not os.path.isdir(folder):
        os.mkdir(folder)
    # img_in = Image.open('data/' + filename +'.png').convert('RGB')
    # img_in.show()
    # img = np.array(img_in)
    img = cv2.imread('data/' + filename + '.png', cv2.IMREAD_COLOR)
    dst = img.copy()
    result, background, number_colors, bar_colors, _ = Hcolor.color_find(
        img)  # , number_colors)
    back = Image.fromarray(background)
    back.save("color_divided/" + filename + "background.png")

    template_coordinate = list()
    min_interval = 5000

    for i, igs in enumerate(result):
        result[i] = edge_enhance(igs)
    template = np.zeros((0,0,3))
    for i in range(number_colors):
        res = Image.fromarray(result[i])
        image_name = "color_divided/"+filename+"color_%i.png" % i
        res.save(image_name)
        
        template, vertex = template_finding(
            img, result[i], bar_colors[i], template)
        #image_name = folder+"/template_%i.png" % i
        #cv2.imwrite(image_name, template)

        _, _, min_interval = template_matching(
            img, result[i], template, min_interval, vertex)

    for i in range(number_colors):
        template, vertex = template_finding(
            img, result[i], bar_colors[i], template)
        image_name = folder+"/template_%i.png" % i
        cv2.imwrite(image_name, template)
        #template = cv2.imread(folder+"/template_%i.png" % i)
        matched_image, coord, min_interval = template_matching(
            img, result[i], template, min_interval, vertex)
        template_coordinate.append(coord)
        image_name = folder+"/matched_image_%i.png" % i
        cv2.imwrite(image_name, matched_image)

        for i in range(coord.shape[1]):
            coord[:,i] -= vertex
        h, w, _ = template.shape
        for i in range(coord.shape[1]):
            cv2.rectangle(dst, (coord[0][i], coord[1][i]),
                        (coord[0][i] + w, coord[1][i] + h), (0, 0, 0), 2)

    cv2.imshow("Head Detection", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # template_coordinate is the bottom vertex coordinate of the detected head
    print(template_coordinate)
    return template_coordinate


if __name__ == '__main__':
    filename = sys.argv[1]
    run(filename)