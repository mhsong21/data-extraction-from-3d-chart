import cv2 as cv
import numpy as np
import sys
from PIL import Image
from matplotlib import pyplot as plt

def color_find(img, num_colors):
    result = list()
    # Convert BGR to HSV
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    hist = cv.calcHist( [hsv], [0, 1], None, [180, 256], [0, 180, 0, 256] )
    
    H = np.sum(hist, axis= 1)
    most_colors = np.argpartition(H, -(num_colors+1));
    most_colors.astype(np.uint8)
    
    ##visualize H histogram
    plt.plot(H)
    plt.show()
    
    ## Background image
    #Assumtion: Background image consist with only white 
    back = cv.inRange(hsv, (1, 0,0), (180,255,255))
    background = cv.bitwise_and(img, img, mask= ~back)
    # backshow = Image.fromarray(background)                    
    # backshow.show()
    # define range of most colors in HSV
    for i in range(num_colors):
        clr_low = (int(most_colors[-2-i]-1), 0, 0)
        clr_up = (int(most_colors[-2-i]+1),255,255)
        mask = cv.inRange(hsv, clr_low, clr_up)
        res = cv.bitwise_and(img,img, mask= mask)
        result.append(res+background)

        # res_out = Image.fromarray(res + background)
        # res_out.show()

    return result, background

def main():
    filename = sys.argv[1]
    img_in = Image.open('data/' + filename).convert('RGB')
    img_in.show()
    img = np.array(img_in)
    number_colors = 3
    result, background = color_find(img, number_colors)
    back = Image.fromarray(background)
    back.save("color_divided/background.png")
    for i in range(number_colors):
        res = Image.fromarray(result[i])
        image_name = "color_divided/color_%i.png" %i
        res.save(image_name)



if __name__ == '__main__':
    main()