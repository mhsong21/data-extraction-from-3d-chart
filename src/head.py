import cv2
import numpy as np
import matplotlib

def template_make(c_in):
    ####
    # codes for making a template manually
    ####
    
    rgb_image = cv2.imread('chart1.png')
    '''
    cv2.imshow('rgb_image',rgb_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    cv2.imwrite('template.png',rgb_image[np.amin(c_in[:,1]) : np.amax(c_in[:,1]) + 1, np.amin(c_in[:,0]): np.amax(c_in[:,0]) + 1, :])
    


def template_matching():
    ####
    # codes for matching the template
    ####
    src = cv2.imread("chart1.png", cv2.IMREAD_GRAYSCALE)
    template = cv2.imread("template.png", cv2.IMREAD_GRAYSCALE)
    dst = cv2.imread("chart1.png")

    result = cv2.matchTemplate(src, template, cv2.TM_CCOEFF_NORMED)
    
    # using threshold
    threshold = 0.6
    h, w = template.shape
    loc = np.where( result >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(dst, pt, (pt[0] + w, pt[1] + h), (0,0,255), 1)
    cv2.imshow("dst", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    '''
    # not using threshold, just one matching
    cv2.imwrite('result.png', result*255)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
    x, y = minLoc
    h, w = template.shape

    dst = cv2.rectangle(dst, (x, y), (x +  w, y + h) , (0, 0, 255), 1)
    cv2.imshow("dst", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    
def template_finding():
    ####
    # codes for finding a template (head)
    ####
    src = cv2.imread("chart1.png", cv2.IMREAD_GRAYSCALE)
    # assume that the deteceted color BGR is [168 38 61], gray scale is 59
    colorValue = 59
    colorArea = np.where(src == colorValue, src, 0)
    
    x = 0
    y = 0
    # Since legend contains blue color, to avoid these, I chose range starting from index 100
    for j in range(100, colorArea.shape[0]):
        if np.amax(colorArea[j,:]) == colorValue:
            y = j
            x = np.where(colorArea[j,:] != 0)
            x = x[0][0]
            break
    # the coordinate of blue head top point is saved in x, y
    
    # for calculating the slope of x-axis and y-axis of chart, 
    # I put manually the stating and end points of axes
    # x-axis: (370, 561) (453, 431)
    # y-axis: (370, 561) (41,348)
    x_start = np.array([370, 561]) 
    x_end = np.array([453, 531])
    y_start = np.array([370, 561])
    y_end = np.array([41, 348])
    
    x_slope = np.arctan2(x_start[1] - x_end[1], x_start[0] - x_end[0])
    y_slope = np.arctan2(y_end[1] - y_start[1], y_end[0] - y_start[0])

    # along parrel line to x-axis, find the point where grayscale value of adjacent 5 pixels up and down. 
    for i in range(0, colorArea.shape[1]):
        if np.amax(colorArea[y - int(i*np.tan(x_slope)) - 2 : y - int(i*np.tan(x_slope)) + 3, x - i]) == 0 :
            x_alongx = x - i
            y_alongx = y - int(i*np.tan(x_slope))
            break        

    # along parrel line to y-axis, find the point where grayscale value of adjacent 5 pixels up and down.
    for i in range(0, colorArea.shape[1]):
        if np.amax(colorArea[y + int(i*np.tan(y_slope)) - 2 : y + int(i*np.tan(y_slope)) + 3, x + i]) == 0 :
            x_alongy = x + i
            y_alongy = y + int(i*np.tan(y_slope))
            break 

    template_coord = np.array([[x,y], [x_alongy,y_alongy], [x_alongx,y_alongx], [x_alongx + x_alongy - x, y_alongx + y_alongy - y]])
    return template_coord


def line_detection_1():
    ####
    # codes for detecting lines
    ####
    src = cv2.imread("chart1.png")
    dst = src.copy()
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 5000, 1500, apertureSize = 7, L2gradient = True)
    cv2.imshow("canny",canny)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    lines = cv2.HoughLines(canny, 1.0, np.pi / 45, 200, srn = 0, stn = 0, min_theta = -np.pi, max_theta = np.pi)

    for i in lines:
        rho, theta = i[0][0], i[0][1]
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a*rho, b*rho

        scale = src.shape[0] + src.shape[1]

        x1 = int(x0 + scale * -b)
        y1 = int(y0 + scale * a)
        x2 = int(x0 - scale * -b)
        y2 = int(y0 - scale * a)

        cv2.line(dst, (x1, y1), (x2, y2), (0, 0, 255), 1)
        #cv2.circle(dst, (x0, y0), 3, (255, 0, 0), 5, cv2.FILLED)

    cv2.imshow("dst", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()    

def line_detection_2():
    ####
    # codes for detecting lines
    ####
    src = cv2.imread("chart1.png")
    dst = src.copy()
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 5000, 1500, apertureSize = 7, L2gradient = True)
    cv2.imshow("canny",canny)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    lines = cv2.HoughLinesP(canny, 1.0, np.pi / 180, 100, minLineLength = 5, maxLineGap = 10)
    print(lines)
    for i in lines:
        cv2.line(dst, (i[0][0], i[0][1]), (i[0][2], i[0][3]), (0, 0, 255), 1)

    cv2.imshow("dst", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    #template_matching()
    #line_detection_2()
    template_coord = template_finding()
    template_make(template_coord)
    template_matching()