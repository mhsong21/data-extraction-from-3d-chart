import ocr
import axis
import head
import sys
import draw_bottomline
import subprocess
import os
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


def save_predict_as_csv(x_label, y_label, data, save_path):
    df = pd.DataFrame(data, header=x_label, index=y_label)
    df.to_csv(save_path)


def main(filename):
    chart_name = filename + '.png'
    folder_path = os.path.join(os.path.abspath(sys.path[0]), "../data/")
    chart_path = folder_path + chart_name
    craft_model_path = os.path.join(
        os.path.abspath(sys.path[0]), "../craft_mlt_25k.pth")
    subprocess.call(["python", "CRAFT-pytorch/test.py",
                     "--test_folder=" + folder_path,
                     "--trained_model=" + craft_model_path,
                     "--file_name=" + chart_name])
    box_path = os.path.join(os.path.abspath(
        sys.path[0]), '../result/res_' + filename + '.txt')

    axis_points, degrees = axis.axis(folder_path, chart_name)
    axis_list = [x[0] + x[1] for x in axis_points]

    result, dbox = ocr.tick_to_value(chart_path, box_path, axis_list)

    zaxis = result[0]
    tick_val = 0.0
    tick_px = float(dbox[0])
    for i in range(len(zaxis)-1):
        z1 = zaxis[i+1]
        z0 = zaxis[i]
        if z1 is not None and z0 is not None:
            tick_val = float(z1 - z0)
            break
    bottom_line = draw_bottomline.main(filename, axis_points, degrees, dbox)
    template_coordinate = head.run(filename, axis_points)

    print("=============================")
    print("=============================")
    print("=============================")
    print("=============================")
    print("=============================")
    print("=============================")
    print(result)

    print(tick_px, tick_val)
    print(template_coordinate)

    x_len = len(bottom_line)
    y_len = len(template_coordinate[0][0])

    coord_map = np.empty((x_len, y_len, 2))
    data_map = np.empty((x_len, y_len))
    for i, line in enumerate(bottom_line):
        x_list = template_coordinate[i][0]
        y_list = template_coordinate[i][1]
        for j, point in enumerate(zip(x_list, y_list)):
            h_px = line.height(point)
            h_val = h_px * tick_val / tick_px
            coord_map[i, j] = point
            data_map[i, j] = h_val

    print(data_map)
    draw_values(chart_path, data_map, coord_map)


def draw_values(chart_path, data_map, coord_map):
    img = Image.open(chart_path).convert('L')
    draw = ImageDraw.Draw(img)

    ih, jh = data_map.shape
    for i in range(ih):
        for j in range(jh):
            x, y = coord_map[i, j]
            h_val = data_map[i, j]
            draw.text((x-15, y-20), str(int(h_val)))
    plt.imshow(img, cmap='gray')
    plt.show()


if __name__ == "__main__":
    filename = sys.argv[1]  # ex) Matlab8
    main(filename)
