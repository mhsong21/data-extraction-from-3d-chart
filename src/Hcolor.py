import cv2
import numpy as np
import sys
import scipy.ndimage as ndimage
from PIL import Image
from matplotlib import pyplot as plt


def color_find(img):  # , num_colors):
    result = list()
    # Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])    
    whites = np.unravel_index(hist.argmax(), hist.shape)
    hist[whites] = 0
    maxi = ndimage.maximum_filter(hist, size=(1, 1))
    M = np.max(maxi)
    val = np.unique(maxi)
    cnt = 0
    color_val = list()
    colors_temp = list()
    colors = list()

    print(val)
    for v in val:
        if v >= 0.08 * M and v >= 1000:
            cnt += 1
            color_val.append(v)
    print(color_val)
    for cvl in color_val:
        index = np.where(hist == cvl)
        print(index)
        colors_temp.append((index[0][0], index[1][0]))
    num_colors = cnt

    for i in range(cnt-1):
        isduplicated = False
        for j in range(i+1, cnt):
            if abs(colors_temp[i][0] - colors_temp[j][0]) + abs(colors_temp[i][1] - colors_temp[j][1]) < 5:
                num_colors -= 1
                isduplicated = True
                break
        if isduplicated is False:
            colors.append(colors_temp[i])

    colors.append(colors_temp[cnt-1])

    print("number: ", num_colors)
    print(colors)

    # H = np.sum(hist, axis= 1)
    # most_colors = np.argpartition(H, -(num_colors+1));
    # most_colors.astype(np.uint8)
    ##visualize H histogram
    #plt.plot(H)
    #plt.show()
    
    ## Background image
    #Assumtion: Background image consist with only white 
    back = cv2.inRange(hsv, (0, 0, 0), (180, 0, 255))
    background = cv2.bitwise_and(img, img, mask=back)
    # backshow = Image.fromarray(background)                    
    # backshow.show()
    # define range of most colors in HSV
    # for i in range(num_colors):
    #     clr_low = (int(most_colors[-2-i]-1), 0, 0)
    #     clr_up = (int(most_colors[-2-i]+1),255,255)
    #     mask = cv.inRange(hsv, clr_low, clr_up)
    #     res = cv.bitwise_and(img,img, mask= mask)
    #     result.append(res+background)

    # res_out = Image.fromarray(res + background)
    # res_out.show()
    color_order = []
    for i in range(num_colors):
        clr_low = (int(colors[i][0]-3), int(colors[i][1]-3), 0)
        clr_up = (int(colors[i][0]+3), int(colors[i][1]+3), 255)
        print(clr_low, clr_up)
        mask = cv2.inRange(hsv, clr_low, clr_up)
        height, width = mask.shape
        maxy = 0
        maxx = 0
        for y in range(height):
            for x in reversed(range(width)):
                if mask[y,x] == 255 and maxy < y:
                    maxy = y
                    maxx = x
        res = cv2.bitwise_and(img, img, mask=mask)
        color_order.append((maxy,maxx,res,colors[i]))
    color_sort = sorted(color_order, key = lambda x:x[0])
    color_sort = color_sort[::-1]
    bottomline_interval = 0
    bottomline_interval = bottomline_interval +  ((color_sort[0][0] - color_sort[num_colors-1][0])**2 + (color_sort[0][1] - color_sort[num_colors-1][1])**2)**0.5
    bottomline_interval = bottomline_interval/(num_colors-1)
    result = [x[2] for x in color_sort]
    colors = [x[3] for x in color_sort]
    # result : color bars
    # background : background with no bars
    # num_colors : number of color bars
    # colors : Bars color value in list of tuple (H,S)
    print(bottomline_interval)
    return result, background, num_colors, colors, bottomline_interval


def main():
    filename = sys.argv[1]
    img_in = Image.open('data/' + filename + '.png').convert('RGB')
    # img_in.show()
    img = np.array(img_in)
    result, background, number_colors, bar_colors, bottomline_interval = color_find(img)
    # , number_colors)
    back = Image.fromarray(background)
    back.save("color_divided/" + filename + "background.png")
    for i in range(number_colors):
        res = Image.fromarray(result[i])
        image_name = "color_divided/"+filename+"color_%i.png" % i
        res.save(image_name)


if __name__ == '__main__':
    main()
