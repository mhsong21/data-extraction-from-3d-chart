import ocr


def main():
    chart_path = './data/excel_chart_1.png'
    box_path = './CRAFT-pytorch/result/res_excel_chart_1.txt'
    axis_list = [
        [52, 743, 52, 285],
        [52, 743, 1252, 743]
    ]
    num_ticks = [11, 11]

    result = ocr.tick_to_value(chart_path, box_path, axis_list, num_ticks)
    print(result)


if __name__ == "__main__":
    main()
