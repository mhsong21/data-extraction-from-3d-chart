import ocr
import axis
import head
import sys
import draw_bottomline
import subprocess
import os


def main(filename):
    chart_name = filename + '.png'
    folder_path = os.path.join(os.path.abspath(sys.path[0]), "../data/")
    chart_path = folder_path + chart_name
    craft_model_path = os.path.join(os.path.abspath(sys.path[0]), "../craft_mlt_25k.pth")
    subprocess.call(["python", "CRAFT-pytorch/test.py", "--test_folder="+folder_path, "--trained_model="+craft_model_path, "--file_name="+chart_name])
    box_path = os.path.join(os.path.abspath(sys.path[0]), '../result/res_' + filename + '.txt')

    axis_points, degrees = axis.axis(folder_path, chart_name)
    print("axis_points")
    print(axis_points)
    print(degrees)
    axis_list = [x[0] + x[1] for x in axis_points]

    result, dbox = ocr.tick_to_value(chart_path, box_path, axis_list)

    zaxis = result[0]
    tick_val = 0
    tick_pixel = dbox[0]
    for i in range(len(zaxis)-1):
        z1 = zaxis[i+1]
        z0 = zaxis[i]
        if z1 is not None and z0 is not None:
            tick_val = (z1 - z0)
            break

    print(tick_pixel, tick_val)
    bottom_line = draw_bottomline.main(filename, axis_points, degrees, dbox)
    print(bottom_line)
    template_coordinate = head.run(filename, axis_points)
    print(template_coordinate)




if __name__ == "__main__":
    filename = sys.argv[1]  # ex) Matlab8
    main(filename)
