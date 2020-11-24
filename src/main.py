import ocr
import axis
import head
import sys


def main(filename):
    folder_path = './data/'
    chart_name = filename + '.png'
    chart_path = folder_path + chart_name
    box_path = './CRAFT-pytorch/result/res_' + filename + '.txt'

    axis_points, degrees = axis.axis(folder_path, chart_name, result_folder_path='./')
    axis_list = [x[0] + x[1] for x in axis_points]

    result = ocr.tick_to_value(chart_path, box_path, axis_list)
    print(result)
    head.run(filename, axis_points)


if __name__ == "__main__":
    filename = sys.argv[1]
    # chart_name = 'Matlab8'
    main(filename)
