import cv2
import numpy as np
from PIL import Image
import matplotlib

def template_matching():
    ####
    # codes for matching the template
    ####
    src = cv2.imread("./Matlab8/color_1.png", cv2.IMREAD_COLOR)
    template = cv2.imread("./Matlab8/template_1.png", cv2.IMREAD_COLOR)
    dst = cv2.imread("./Matlab8/color_1.png")

    result = cv2.matchTemplate(src, template, cv2.TM_CCOEFF_NORMED)
    
    # using threshold
    threshold = 0.8
    loc = np.where( result >= threshold)
    # elements of coord[0] are x-coordinates, coord[1] are y-coordinates
    coord = np.zeros((2,len(loc[0])))
    i = 0
    
    for pt in zip(*loc[::-1]):
        coord[0][i] = pt[0]
        coord[1][i] = pt[1]
        i += 1
    coord = coord.astype(int)
    #np.sort(coord.view('i8,i8'), order=['f1'], axis=0).view(np.int)
    # Still, coord has a problem that it has nearby(overlapping) templates.
    # get rid of nearby templates and save it on the coordinate
    
    coordinate = coord
    for i in range(coord.shape[1]):
        for j in range(coord.shape[1]):
            if coord[0][j] - 3 <= coord[0][i] <= coord[0][j] + 3:
                if result[ coord[1][j] ][ coord[0][j] ] > result[ coord[1][i] ][ coord[0][i] ]:
                    coordinate[:,i] = coord[:,j]
    coordinate = np.unique(coordinate, axis=1)
    
     # dst is the image which is marked where the template is matched

    h, w, _ = template.shape
    for i in range(coordinate.shape[1]):
            cv2.rectangle(dst, (coordinate[0][i], coordinate[1][i]), (coordinate[0][i] + w, coordinate[1][i] + h), (0,0,255), 1)

    cv2.imshow("dst", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    return dst, coordinate

    
def template_finding():
    ####
    # codes for finding a template (head)
    ####
    color_filtered_img = cv2.imread("./Matlab8/color_1.png", cv2.IMREAD_GRAYSCALE)
    # assume the deteceted color gray scale value
    colorValue = color_filtered_img[108][274]

    cv2.imshow("colorfiltered", color_filtered_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    colorArea = np.where(color_filtered_img == colorValue, color_filtered_img, 0)

    cv2.imshow("colorArea", colorArea)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    x = 0
    y = 0
    # no legend, start from index 50 to avoid matching to chart title 
    for j in range(50, colorArea.shape[0]):
        if np.amax(colorArea[j,50:]) == colorValue:
            a = np.where(colorArea[j,:] != 0)
            a = a[0][0]
            if np.amin(colorArea[ j+2 : j+4 , a: a+2]) != 0: 
                y = j
                x = a
                break
    # the coordinate of head top point is saved in x, y
    x -= 1
    y -= 1
    # for calculating the slope of x-axis and y-axis of chart, 
    # I put manually the stating and end points of axes
    # x-axis: x_start / x_end is the starting / ending points of x-axis respectively.
    # y-axis: y_start / y_end is the starting / ending points of y-axis respectively.
    x_start = np.array([572, 562]) 
    x_end = np.array([653, 366])
    y_start = np.array([572, 562])
    y_end = np.array([55, 488])

    x_slope = (x_end[1] - x_start[1]) / (x_end[0] - x_start[0])
    y_slope = (y_end[1] - y_start[1]) / (y_end[0] - y_start[0])

    if y_slope < 0:
        temp = y_slope
        y_slope = x_slope
        x_slope = y_slope
    print(x_slope, y_slope)
    # along parrel line to x-axis, find the point where grayscale value of adjacent 5 pixels up and down. 
    for i in range(0, colorArea.shape[1]):
        if np.amax(colorArea[y - int(i*x_slope) - 5 : y - int(i*x_slope) + 5, x - i]) == 0 :
            x_alongx = x - i
            y_alongx = y - int(i*x_slope)
            break        
    
    # along parrel line to y-axis, find the point where grayscale value of adjacent 5 pixels up and down.
    for i in range(0, colorArea.shape[1]):
        if np.amax(colorArea[y + int(i*y_slope) - 5 : y + int(i*y_slope) + 5, x + i]) == 0 :
            x_alongy = x + i
            y_alongy = y + int(i*y_slope)
            break 
    print(x,y)
    print(x_alongx,y_alongx)
    print(x_alongy,y_alongy)
    template_coord = np.array([[x,y], [x_alongy,y_alongy], [x_alongx,y_alongx], [x_alongx + x_alongy - x, y_alongx + y_alongy - y]])

    rgb_image = cv2.imread('./Matlab8/color_1.png')
    template = rgb_image[np.amin(template_coord[:,1]) : np.amax(template_coord[:,1]) + 1, np.amin(template_coord[:,0]): np.amax(template_coord[:,0]) + 1, :]
    return template

if __name__ == '__main__':
    template = template_finding()
    cv2.imwrite('./Matlab8/template_1.png',template)
    matched_image, template_coord = template_matching()
    cv2.imwrite('./Matlab8/matched_image_1.png',matched_image)
    print(template_coord)