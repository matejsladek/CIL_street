# -----------------------------------------------------------
# Helper script
# Generates a pandas df from validation_score.txt in
# experiment result folders.
# CIL 2020 - Team NaN
# -----------------------------------------------------------
import os
import numpy as np
import pandas as pd

project_dir = os.path.join(*[os.path.dirname(os.path.abspath(__file__)),"..",".."])


def gen_df_row_data(vs_str,mode)
    res_str = vs_str.split('\n')[2]
    char_to_remove = "[] "
    for c in char_to_remove:
        res_str = res_str.replace(c,"")
    res_np = np.array(res_str.split(","))
    res_np = res_np.astype(np.float)

    row = {}
    if mode=="stl":
        row["loss_t0"] = res_np[0]
        row["acc_t0"] = res_np[1]
        row["kaggle_t0"] = res_np[2]
        row["f1_t0"] = res_np[3]
        row["iou_t0"] = res_np[4]
    elif mode=="mtl_contour_distance":
        raise(Exception)
    elif mode=="mtl_contour":
        row["loss_t0"] = res_np[1]
        row["acc_t0"] = res_np[3]
        row["kaggle_t0"] = res_np[4]
        row["f1_t0"] = res_np[5]
        row["iou_t0"] = res_np[6]
    elif mode=="mtl_distance":
        row["loss_t0"] = res_np[1]
        row["acc_t0"] = res_np[3]
        row["kaggle_t0"] = res_np[4]
        row["f1_t0"] = res_np[5]
        row["iou_t0"] = res_np[6]
    else:
        raise(Exception)
    return(row)


def df_init():
    #All columns to include in pandas dataframe
    df_col_names = ["exp_type","loss_t0","acc_t0","kaggle_t0","f1_t0","iou_t0",
            "exp_path","split"]
    df = pd.DataFrame(columns=df_col_names)
    return(df)


def gen_validation_score_dfs(exp_path_list,mode="stl"):
    df = df_init()
    for i in range(len(exp_path_list)):
        print(exp_path_list[i])
        split_name_list = [s for s in os.listdir(exp_path_list[i]) if "split" in s]
        split_path_list = [os.path.join(exp_path_list[i],s) for s in split_name_list]
        for j in range(len(split_path_list)):
            vsp = os.path.join(split_path_list[j],"validation_score.txt")
            f = open(vsp,"r")
            vs_str = f.read()
            row = gen_df_row_data(vs_str,mode)
            row["exp_type"] = mode
            row["exp_path"] = exp_path_list[i]
            row["split"] = j
            df = df.append(row,ignore_index=True)
    return(df)


if __name__ == "__main__":
    #########################################################################
    ### MANUAL CONFIGURATION of baseline and experiments to generate here ###

    #Specify location of folder containing experiment results
    #and experiments of interest
    exp_dir = "experiments"
    exp_name_list = [
            "mtl_contour_0709_10_51_13",
            "mtl_distance_0710_12_28_03",
            "mtl_distance_0710_12_28_03",
            "stl_0709_21_17_29",
            "stl_0709_21_30_29",
#            "stl_0710_20_21_30"
            ]

    #Metrics to report in final statistical summary
    metrics_of_interest = ["loss_t0","acc_t0","kaggle_t0","f1_t0","iou_t0"]
    #########################################################################


    exp_path_list = [os.path.join(*[project_dir,exp_dir,exp]) for exp in exp_name_list]
    df = gen_validation_score_dfs(exp_path_list)
    stats = df[["exp_type"]+
            metrics_of_interest].groupby(["exp_type"]).agg(["mean","std","count"])
    print(stats)
