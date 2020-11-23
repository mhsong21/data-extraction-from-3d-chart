import ocr
import axis
import head


def main(filename):
    folder_path = './data/'
    chart_name = filename + '.png'
    chart_path = folder_path + chart_name
    box_path = './CRAFT-pytorch/result/res_' + filename + '.txt'

    axis_points = axis.main(folder_path, chart_name, result_folder_path='./')
    axis_list = [x[0] + x[1] for x in axis_points]
    # num_ticks = [4, 6, 12]

    result = ocr.tick_to_value(chart_path, box_path, axis_list)
    print(result)
    head.run(filename)


if __name__ == "__main__":
    # filename = sys.argv[1]
    chart_name = 'Matlab1'
    main(chart_name)
