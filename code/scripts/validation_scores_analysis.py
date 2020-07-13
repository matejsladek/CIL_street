import os
import numpy as np
import pandas as pd

#project_dir = "../.."
project_dir = "../.."

if __name__ == "__main__":

    col_names = ["exp_type","loss_t0","acc_t0","kaggle_t0","f1_t0","iou_t0","exp_path","split"]
    metrics_of_interest = ["loss_t0","acc_t0","kaggle_t0","f1_t0","iou_t0"] #to report final stats
    df = pd.DataFrame(columns=col_names)

    exp_dir = "experiments"
    exp_name_list = [
            "mtl_contour_0709_10_51_13",
            "mtl_distance_0710_12_28_03",
            "mtl_distance_0710_12_28_03",
            "stl_0709_21_17_29",
            "stl_0709_21_30_29",
#            "stl_0710_20_21_30"
            ]

    exp_path_list = [os.path.join(*[project_dir,exp_dir,exp]) for exp in exp_name_list]

    for i in range(len(exp_path_list)):
        print(exp_path_list[i])
        split_name_list = [s for s in os.listdir(exp_path_list[i]) if "split" in s]
        split_path_list = [os.path.join(exp_path_list[i],s) for s in split_name_list]
       
        for j in range(len(split_path_list)):
            vsp = os.path.join(split_path_list[j],"validation_score.txt")
            f = open(vsp,"r")
            vsp_str = f.read()
            res_str = vsp_str.split('\n')[2]
            char_to_remove = "[] "
            for c in char_to_remove:
                res_str = res_str.replace(c,"")
            res_np = np.array(res_str.split(","))
            res_np = res_np.astype(np.float)

            row = {}
            if "stl" in exp_path_list[i]:
                row["exp_type"] = "stl"
                row["loss_t0"] = res_np[0]
                row["acc_t0"] = res_np[1]
                row["kaggle_t0"] = res_np[2]
                row["f1_t0"] = res_np[3]
                row["iou_t0"] = res_np[4]
            elif "mtl_contour_distance" in exp_path_list[i]:
                raise(Exception)
            elif "mtl_contour" in exp_path_list[i]:
                row["exp_type"] = "mtl_contour"
                row["loss_t0"] = res_np[1]
                row["acc_t0"] = res_np[3]
                row["kaggle_t0"] = res_np[4]
                row["f1_t0"] = res_np[5]
                row["iou_t0"] = res_np[6]
            elif "mtl_distance" in exp_path_list[i]:
                row["exp_type"] = "mtl_distance"
                row["loss_t0"] = res_np[1]
                row["acc_t0"] = res_np[3]
                row["kaggle_t0"] = res_np[4]
                row["f1_t0"] = res_np[5]
                row["iou_t0"] = res_np[6]
            else:
                raise(Exception)
            row["exp_path"] = exp_path_list[i]
            row["split"] = j
            df = df.append(row,ignore_index=True)

    stats = df[["exp_type"]+metrics_of_interest].groupby(["exp_type"]).agg(["mean","std","count"])


