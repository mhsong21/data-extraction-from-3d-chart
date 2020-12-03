import csv
import numpy as np
import pandas as pd
import sys

def evaluate(predict_csv_path, ground_truth_csv_path):
    print("----------Ground truth data----------")
    gt_df = pd.read_csv(ground_truth_csv_path, index_col=0)
    print(gt_df)
    print("-------------------------------------\n")

    print("-----------Predicted data------------")
    predict_df = pd.read_csv(predict_csv_path, index_col=0)
    print(predict_df)
    print("-------------------------------------\n")

    gt_npy = gt_df.to_numpy()
    pred_npy = predict_df.to_numpy()
    error = pred_npy - gt_npy
    print(error)
    mean_error_rate = np.nanmean(np.abs(error) / np.abs(gt_npy))
    print("mean_error_rate : ", format(mean_error_rate * 100, ".3f"), "%")



if __name__ == "__main__":
    filename = sys.argv[1]  # ex) Matlab8
    evaluate("./result/pred_" + filename + ".csv", "./data/ground_truth/" + filename + ".csv")