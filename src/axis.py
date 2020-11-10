import math

import cv2
import numpy as np


def show_wait_destroy(window_name, img):
    cv2.imshow(window_name, img)
    cv2.moveWindow(window_name, 500, 0)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)


def make_kernel_by_theta(theta, length=15):
    if theta == 0:
        return np.ones((1, length))
    elif theta == -90:
        return np.ones((length, 1))
    
    row = math.ceil(np.abs(length * np.sin(theta * np.pi / 180)))
    col = math.ceil(np.abs(length * np.cos(theta * np.pi / 180)))
    print(theta, row, col)

    start_index = 0
    if 0 < theta < 45:
        step = float(col) / float(row)
        kernel = np.zeros((row, col), np.uint8)
        for i in range(row):
            index = step * i
            index_ceiling = math.ceil(index)
            kernel[i, start_index:index_ceiling+1] = 1
            start_index = index_ceiling + 1
    elif 45 <= theta < 90:
        step = float(row) / float(col)
        kernel = np.zeros((row, col), np.uint8)
        for i in range(col):
            index = step * i
            index_ceiling = math.ceil(index)
            kernel[start_index:index_ceiling+1, i] = 1
            start_index = index_ceiling + 1
    elif -45 <= theta < 0:
        step = float(col) / float(row)
        kernel = np.zeros((row, col), np.uint8)
        for i in range(row):
            index = step * i
            index_ceiling = math.ceil(index)
            kernel[row-1-i, start_index:index_ceiling+1] = 1
            start_index = index_ceiling + 1
    elif -90 < theta < -45:
        step = float(row) / float(col)
        kernel = np.zeros((row, col), np.uint8)
        for i in range(col):
            index = step * i
            index_ceiling = math.ceil(index)
            kernel[start_index:index_ceiling+1, col-1-i] = 1
            start_index = index_ceiling + 1
    print(kernel)
    return kernel


def make_kernel(row, col, rightdown=True):
    start_index = 0
    if rightdown:
        if col > row:
            step = float(col) / float(row)
            kernel = np.zeros((row, col), np.uint8)
            for i in range(row):
                index = step * i
                index_ceiling = math.ceil(index)
                kernel[i, start_index:index_ceiling+1] = 1
                start_index = index_ceiling + 1
        else:
            step = float(row) / float(col)
            kernel = np.zeros((row, col), np.uint8)
            for i in range(col):
                index = step * i
                index_ceiling = math.ceil(index)
                kernel[start_index:index_ceiling+1, i] = 1
                start_index = index_ceiling + 1
    else:
        if col > row:
            step = float(col) / float(row)
            kernel = np.zeros((row, col), np.uint8)
            for i in range(row):
                index = step * i
                index_ceiling = math.ceil(index)
                kernel[row-1-i, start_index:index_ceiling+1] = 1
                start_index = index_ceiling + 1
        else:
            step = float(row) / float(col)
            kernel = np.zeros((row, col), np.uint8)
            for i in range(col):
                index = step * i
                index_ceiling = math.ceil(index)
                kernel[start_index:index_ceiling+1, col-1-i] = 1
                start_index = index_ceiling + 1
    return kernel


def make_kernel_backup(row, col, rightdown=True):
    if rightdown:
        if col > row:
            step = float(col) / float(row)
            kernel = np.zeros((row, col), np.uint8)
            for i in range(row):
                index = step * i
                index_floor = math.floor(index)
                index_ceiling = math.ceil(index)
                kernel[i, index_floor] = 1
                kernel[i, index_ceiling] = 1
        else:
            step = float(row) / float(col)
            kernel = np.zeros((row, col), np.uint8)
            for i in range(col):
                index = step * i
                index_floor = math.floor(index)
                index_ceiling = math.ceil(index)
                kernel[index_floor, i] = 1
                kernel[index_ceiling, i] = 1
    else:
        if col > row:
            step = float(col) / float(row)
            kernel = np.zeros((row, col), np.uint8)
            for i in range(row):
                index = step * i
                index_floor = math.floor(index)
                index_ceiling = math.ceil(index)
                kernel[row-1-i, index_floor] = 1
                kernel[row-1-i, index_ceiling] = 1
        else:
            step = float(row) / float(col)
            kernel = np.zeros((row, col), np.uint8)
            for i in range(col):
                index = step * i
                index_floor = math.floor(index)
                index_ceiling = math.ceil(index)
                kernel[index_floor, col-1-i] = 1
                kernel[index_ceiling, col-1-i] = 1
    return kernel


def extract_chart(img, num_colors):
    # Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist( [hsv], [0, 1], None, [180, 256], [0, 180, 0, 256] )
    
    H = np.sum(hist, axis= 1)
    most_colors = np.argpartition(H, -(num_colors+1))
    most_colors.astype(np.uint8)
    
    ## Background image
    #Assumtion: Background image consist with only white 
    chart_msk = cv2.inRange(hsv, (1, 0,0), (180,255,255))
    chart = cv2.bitwise_and(img, img, mask=chart_msk)
    return chart


def remove_color(img, num_colors):
    # Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist( [hsv], [0, 1], None, [180, 256], [0, 180, 0, 256] )
    
    H = np.sum(hist, axis= 1)
    most_colors = np.argpartition(H, -(num_colors+1))
    most_colors.astype(np.uint8)
    
    ## Background image
    #Assumtion: Background image consist with only white 
    chart = cv2.inRange(hsv, (1, 0,0), (180,255,255))
    removed = cv2.bitwise_or(img, img, mask=~chart)
    removed_gray = cv2.cvtColor(removed, cv2.COLOR_BGR2GRAY)
    removed_gray = removed_gray + chart

    show_wait_destroy("removed_gray", removed_gray)

    return removed_gray


def find_frequent_degree(img, edges):
    threshold = 10
    minLineLength = 70
    maxLineGap = 30

    lines = cv2.HoughLinesP(edges, 1, np.pi/360, threshold, minLineLength, maxLineGap)
    print(len(lines))

    degree_resolution = 180
    theta_votes = np.zeros(degree_resolution)
    for line in lines:
        x1,y1,x2,y2 = line[0]
        cv2.line(img,(x1,y1),(x2,y2),(0,255,0),1)
        slope = (float)(y2 - y1) / (float)(x2 - x1 + 1e-12)
        theta = np.arctan(slope) * 180 / np.pi
        theta = int(np.around(theta))
        if theta == 90:
            theta = -90
        theta_votes[theta+90] += 1

    # non-maximal suppression
    suppression_result = np.zeros(180)
    for i in range(degree_resolution):
        start_index = max(i-2, 0)
        end_index = min(i+2, degree_resolution - 1)
        if max(theta_votes[start_index:end_index+1]) <= theta_votes[i]:
            suppression_result[i] = theta_votes[i]
    sorted_indices = np.argsort(suppression_result)[::-1]

    show_wait_destroy("img", img)
    return sorted_indices[0] - 90, sorted_indices[1] - 90, sorted_indices[2] - 90


def find_z_axis(gray):
    gray = cv2.bitwise_not(gray)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                cv2.THRESH_BINARY, 15, -2)

    kernel = np.ones((20, 1))

    result = cv2.erode(bw, kernel)
    result = cv2.dilate(result, kernel)
    result_npy = np.array(result)

    show_wait_destroy("result", result)

    h, w = result.shape

    # find leftmost point
    for j in range(w):
        if np.sum(result_npy[:, j]) > 0:
            for i in range(h):
                if result_npy[i, j] > 0:
                    point1 = (j, i)  # (x, y)
                    break
            for i in range(h):
                if result_npy[h-1-i, j] > 0:
                    point2 = (j, h-1-i)  # (x, y)
                    break
            break

    return point1, point2


def find_x_axis(gray):
    gray = cv2.bitwise_not(gray)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                cv2.THRESH_BINARY, 15, -2)

    kernel = make_kernel(9, 15)
    kernel = make_kernel_by_theta(33)

    result = cv2.erode(bw, kernel)
    result = cv2.dilate(result, kernel)
    result_npy = np.array(result)

    show_wait_destroy("result", result)

    h, w = result.shape

    # find bottommost point
    for i in range(h):
        if np.sum(result_npy[h-1-i]) > 0:
            for j in range(w):
                if result_npy[h-1-i, j] > 0:
                    point1 = (j, h-1-i)  # (x, y)
                    break
            break
    
    # find leftmost point
    for j in range(w):
        if np.sum(result_npy[:, j]) > 0:
            for i in range(h):
                if result_npy[h-1-i, j] > 0:
                    point2 = (j, h-1-i)  # (x, y)
                    break
            break

    return point1, point2


def find_y_axis(gray):
    gray = cv2.bitwise_not(gray)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                cv2.THRESH_BINARY, 15, -2)

    kernel = make_kernel(5, 15, rightdown=False)
    kernel = make_kernel_by_theta(-21)

    result = cv2.erode(bw, kernel)
    result = cv2.dilate(result, kernel)
    result_npy = np.array(result)

    show_wait_destroy("bw", bw)
    show_wait_destroy("result", result)

    h, w = result.shape

    # find bottommost point
    for i in range(h):
        if np.sum(result_npy[h-1-i]) > 0:
            for j in range(w):
                if result_npy[h-1-i, j] > 0:
                    point1 = (j, h-1-i)  # (x, y)
                    break
            break

    # find rightmost point
    for j in range(w):
        if np.sum(result_npy[:, w-1-j]) > 0:
            for i in range(h):
                if result_npy[h-1-i, w-1-j] > 0:
                    point2 = (w-1-j, h-1-i)  # (x, y)
                    break
            break

    return point1, point2


def main(image_path):
    img = cv2.imread(image_path)
    """
    chart = extract_chart(img, 3)
    chart_gray = cv2.cvtColor(chart, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(chart_gray, 50, 150, apertureSize = 3)
    show_wait_destroy("edges", edges)
    find_frequent_degree(img, edges)
    return
    """

    gray = remove_color(img, 3)
    bw_not = cv2.bitwise_not(gray)
    show_wait_destroy("bw_not", bw_not)
    d1, d2, d3 = find_frequent_degree(img, bw_not)

    z_axis = find_z_axis(gray)
    x_axis = find_x_axis(gray)
    y_axis = find_y_axis(gray)

    print(z_axis)
    print(x_axis)
    print(y_axis)

    cv2.line(img, z_axis[0], z_axis[1], (0, 0, 255), 1)
    cv2.line(img, x_axis[0], x_axis[1], (0, 0, 255), 1)
    cv2.line(img, y_axis[0], y_axis[1], (0, 0, 255), 1)
    show_wait_destroy("img", img)


if __name__ == "__main__":
    main('../data/matlab_three.png')
    # main('./data/excel_one.png')
    # print(make_kernel(5, 17, False))
