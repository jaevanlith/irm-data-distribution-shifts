import itertools
import os

import numpy as np
from matplotlib import pyplot as plt


def sample_complexity():
    main_path = "C:/Users/boeki/OneDrive/Documenten/TU/BEP/Final Results/Sample Complexity/"
    exps_paths = [""]

    make_plot(main_path, exps_paths, "Number of samples")


def dev_between_envs():
    main_path = "C:/Users/boeki/OneDrive/Documenten/TU/BEP/Final Results/Deviation/"
    exps_paths = ["Hetero/", "Homo/", "Homo z=1/"]

    make_plot(main_path, exps_paths, "Deviation between training environments")


def val_of_envs():
    main_path = "C:/Users/boeki/OneDrive/Documenten/TU/BEP/Final Results/Environment Shift/"
    exps_paths = ["Hetero/", "Homo/"]

    make_plot(main_path, exps_paths, "Adaptation of training environment values")


def make_plot(main_path, exps_paths, x_label):
    weights_paths = ["AVG/", "CAU/", "NON/"]
    shifts = ["CS", "CF", "AC", "HB"]
    methods = ["ERM", "IRM"]

    for exps_path in exps_paths:
        for weight_path in weights_paths:
            path = main_path + exps_path + weight_path
            for s in shifts:
                x_axis = []
                res = init_res(methods)

                f = open(path + "txt/" + s + "-regression.txt", "r")
                f.readline()
                f.readline()

                for line in f:
                    row = line.split()
                    if len(row) != 0:
                        if row[2] == 'between':
                            res = init_res(methods)
                            x_axis = []
                            f.readline()
                        else:
                            if row[0] == "ERM":
                                x_axis.append(float(row[1]))
                            res[row[0]]["mean"].append(float(row[2]))
                            res[row[0]]["std err"].append(float(row[3]))

                # Create plot
                marker = itertools.cycle(('o', 'D', 'x', '^'))
                for m in methods:
                    lb = np.subtract(res[m]["mean"], res[m]["std err"])
                    ub = np.add(res[m]["mean"], res[m]["std err"])
                    plt.plot(x_axis, res[m]["mean"], label=m, marker=next(marker))
                    plt.fill_between(x_axis, ub, lb, alpha=0.25)
                plt.title(s + "-regression")
                plt.xlabel(x_label)
                plt.ylabel("Model estimation error")
                plt.legend(loc="best")
                os.makedirs(path + "figures/", exist_ok=True)
                plt.savefig(path + "figures/" + s + "-regression.png")
                plt.clf()


def init_res(methods):
    res = {}
    for m in methods:
        res[m] = {"mean": [], "std err": []}
    return res

if __name__ == "__main__":
    val_of_envs()
