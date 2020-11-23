import math
import os

import cv2
import numpy as np
from PIL import Image


def show_wait_destroy(window_name, img):
    cv2.imshow(window_name, img)
    cv2.moveWindow(window_name, 500, 0)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)


def find_point(axis_img, first_direction, second_direction):
    h, w = axis_img.shape
    x = -1
    y = -1

    if first_direction == "bottommost":
        for i in reversed(range(h)):
            if np.sum(axis_img[i]) > 0:
                y = i
                break
        candidate = axis_img[i]
    elif first_direction == "uppermost":
        for i in range(h):
            if np.sum(axis_img[i]) > 0:
                y = i
                break
        candidate = axis_img[i]
    elif first_direction == "leftmost":
        for j in range(w):
            if np.sum(axis_img[:, j]) > 0:
                x = j
                break
        candidate = axis_img[:, j]
    elif first_direction == "rightmost":
        for j in reversed(range(w)):
            if np.sum(axis_img[:, j]) > 0:
                x = j
                break
        candidate = axis_img[:, j]
    else:
        assert False, "Invalid first direction"

    if second_direction == "bottommost":
        for i in reversed(range(h)):
            if candidate[i] > 0:
                y = i
                break
    elif second_direction == "uppermost":
        for i in range(h):
            if candidate[i] > 0:
                y = i
                break
    elif second_direction == "leftmost":
        for j in range(w):
            if candidate[j] > 0:
                x = j
                break
    elif second_direction == "rightmost":
        for j in reversed(range(w)):
            if candidate[j] > 0:
                x = j
                break
    else:
        assert False, "Invalid first direction"

    if (y != -1) and (x != -1):
        "x or y is not initialized. x : {}, y : {}".format(x, y)
    return x, y


def make_kernel(theta, length=25):
    if theta == 0:
        print("theta : 0")
        return np.ones((1, 30))
    elif theta == -90:
        print("theta : -90")
        return np.ones((30, 1))

    row = math.ceil(np.abs(length * np.sin(theta * np.pi / 180)))
    col = math.ceil(np.abs(length * np.cos(theta * np.pi / 180)))
    print("theta : ", theta)
    print("kernel width : ", col)
    print("kernel height : ", row)

    start_index = 0
    if 0 < theta < 45:
        step = float(col) / float(row)
        kernel = np.zeros((row, col), np.uint8)
        for i in range(1, row+1):
            index = step * i
            index_ceiling = min(math.ceil(index), col)
            kernel[i-1, start_index:index_ceiling] = 1
            start_index = index_ceiling
    elif 45 <= theta < 90:
        step = float(row) / float(col)
        kernel = np.zeros((row, col), np.uint8)
        for i in range(1, col+1):
            index = step * i
            index_ceiling = min(math.ceil(index), row)
            kernel[start_index:index_ceiling, i-1] = 1
            start_index = index_ceiling
    elif -45 <= theta < 0:
        step = float(col) / float(row)
        kernel = np.zeros((row, col), np.uint8)
        for i in range(1, row+1):
            index = step * i
            index_ceiling = min(math.ceil(index), col)
            kernel[row-i, start_index:index_ceiling] = 1
            start_index = index_ceiling
    elif -90 < theta < -45:
        step = float(row) / float(col)
        kernel = np.zeros((row, col), np.uint8)
        for i in range(1, col+1):
            index = step * i
            index_ceiling = min(math.ceil(index), row)
            kernel[start_index:index_ceiling, col-i] = 1
            start_index = index_ceiling
    print(kernel)
    return kernel


def remove_color(img, num_colors):
    # Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    ## Background image
    #Assumtion: Background image consist with only white 
    back = cv2.inRange(hsv, (0, 0, 0), (180, 0, 255))
    removed = cv2.bitwise_or(img, img, mask=back)
    removed_gray = cv2.cvtColor(removed, cv2.COLOR_BGR2GRAY)
    removed_gray = removed_gray + ~back

    show_wait_destroy("removed_gray", removed_gray)

    return removed_gray


def find_frequent_degree(img, edges):
    threshold = 10
    minLineLength = 70
    maxLineGap = 30

    lines = cv2.HoughLinesP(edges, 1, np.pi/360,
                            threshold, minLineLength, maxLineGap)
    print("detected lines in 'find_frequent_degree' function : ", len(lines))

    degree_resolution = 180
    theta_votes = np.zeros(degree_resolution)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
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
    for i in range(5):
        suppression_result[180-1-i] = 0
    print("suppression_result : ", suppression_result)
    sorted_indices = np.argsort(suppression_result)[::-1]

    show_wait_destroy("img", img)
    return sorted_indices[0] - 90, sorted_indices[1] - 90, sorted_indices[2] - 90


def find_axis(gray, degree):
    gray = cv2.bitwise_not(gray)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY, 15, -2)
    kernel = make_kernel(degree)

    result = cv2.erode(bw, kernel)
    result = cv2.dilate(result, kernel)
    result_npy = np.array(result)

    show_wait_destroy("result", result)

    if degree == 0:
        point1 = find_point(result_npy, "bottommost", "leftmost")
        point2 = find_point(result_npy, "bottommost", "rightmost")
    elif degree == -90:
        point1 = find_point(result_npy, "leftmost", "bottommost")
        point2 = find_point(result_npy, "leftmost", "uppermost")
    elif 0 < degree < 90:
        point1 = find_point(result_npy, "bottommost", "rightmost")
        point2 = find_point(result_npy, "leftmost", "uppermost")
    elif -90 < degree < 0:
        point1 = find_point(result_npy, "bottommost", "leftmost")
        point2 = find_point(result_npy, "rightmost", "uppermost")
    return point1, point2


def main(folder_path, img_filename, result_folder_path="../result/"):
    image_path = folder_path + img_filename
    # img = cv2.imread(image_path)
    # img = cv2.resize(img, dsize=(600, 600))
    img_in = Image.open(image_path).convert('RGB')
    img = np.array(img_in)

    gray = remove_color(img, 3)
    bw_not = cv2.bitwise_not(gray)
    show_wait_destroy("bw_not", bw_not)
    degrees = find_frequent_degree(img, bw_not)

    axis_points = []
    for degree in degrees:
        axis_points.append(find_axis(gray, degree))

    for i in range(len(axis_points)):
        print("axis", i, ": ", axis_points[i])

    cv2.line(img, axis_points[0][0], axis_points[0][1], (0, 0, 255), 1)
    cv2.line(img, axis_points[1][0], axis_points[1][1], (0, 0, 255), 1)
    cv2.line(img, axis_points[2][0], axis_points[2][1], (0, 0, 255), 1)
    show_wait_destroy("img", img)
    BGR_to_RGB = np.zeros(img.shape)
    BGR_to_RGB[:, :, 0] = img[:, :, 2]
    BGR_to_RGB[:, :, 1] = img[:, :, 1]
    BGR_to_RGB[:, :, 2] = img[:, :, 0]
    img_pil = Image.fromarray(BGR_to_RGB.astype(np.uint8))
    img_pil.save(result_folder_path + img_filename)
    return axis_points


def iterate_data(folder_path):
    img_list = os.listdir(folder_path)
    img_list = [
        img_file_name for img_file_name in img_list if img_file_name.endswith(".png")]
    for img_file_name in img_list:
        main(folder_path, img_file_name)


if __name__ == "__main__":
    # main('../data/matlab_three.png')
    iterate_data("../data/matlab/")
    # main('../data/', 'Matlab10.png')
    # main('../data/Matlab1.png')
    # print(make_kernel(5, 17, False))
