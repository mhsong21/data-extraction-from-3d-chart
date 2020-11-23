import csv
import numpy as np
import pandas as pd

"""
    gt_f = open(ground_truth_csv_path, 'r', encoding='utf-8')
    gt_reader = csv.reader(gt_f)
    gt_data = []
    for line in gt_reader:
        gt_data.append(line)
    gt_f.close()

    predict_f = open(predict_csv_path, 'r', encoding='utf-8')
    predict_reader = csv.reader(predict_f)
    predict_data = []
    for line in predict_reader:
        predict_data.append(line)
    predict_f.close()
    # predict_df = pd.DataFrame(predict_data)

"""

def evaluate(predict_csv_path, ground_truth_csv_path):
    print("----------Ground truth data----------")
    gt_df = pd.read_csv(ground_truth_csv_path, header=None)
    print(gt_df)
    print("-------------------------------------\n")

    print("-----------Predicted data------------")
    predict_df = pd.read_csv(predict_csv_path, header=None)
    print(predict_df)
    print("-------------------------------------\n")

    gt_npy = gt_df.to_numpy()
    error = (gt_df - predict_df).to_numpy()
    mean_error_rate = np.mean(np.abs(error) / np.abs(gt_npy))
    print("mean_error_rate : ", format(mean_error_rate * 100, ".3f"), "%")



if __name__ == "__main__":
    evaluate("../predict/Matlab16.csv", "../data/ground_truth/Matlab16.csv")