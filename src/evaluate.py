import csv
import numpy as np
import pandas as pd


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