import os

import numpy as np
from matplotlib import pyplot as plt
import tabulate as tbl


def plot_hetero():
    ERM = [0.00982805, 0.0371659, 0.817322, 0.774902]
    err_ERM = [0.00191034, 0.0103385, 0.00275027, 0.00752794]
    IRM = [0.00264055, 0.00841526, 0.285087, 0.280217]
    err_IRM = [0.000497359, 0.0019088, 0.0158806, 0.0155664]

    plot_bar(IRM, err_IRM, ERM, err_ERM, 'Heteroskedastic Y-noise', "./results/hetero/", "heteroskedastic")
    make_table(IRM, err_IRM, ERM, err_ERM, "./results/hetero/", "heteroskedastic")


def plot_homo():
    ERM = [0.000159935, 0.00123384, 0.00960937, 0.0184941]
    err_ERM = [1.80751e-05, 0.000328161, 0.000206782, 0.00160594]
    IRM = [0.0012127, 0.00279001, 0.00299109, 0.0120483]
    err_IRM = [0.00017278, 0.000629739, 0.000619408, 0.00136667]

    plot_bar(IRM, err_IRM, ERM, err_ERM, 'Homoskedastic Y-noise', "./results/hetero/", "homoskedastic")
    make_table(IRM, err_IRM, ERM, err_ERM, "./results/hetero/", "homoskedastic")


def plot_homo_z_one():
    ERM = [0.000780744, 0.0300967, 0.25422, 0.282096]
    err_ERM = [0.000169966, 0.00999697, 0.00147476, 0.0139388]
    IRM = [0.00181727, 0.00514537, 0.256834, 0.257263]
    err_IRM = [0.000347409, 0.00163086, 0.00292934, 0.00366199]

    plot_bar(IRM, err_IRM, ERM, err_ERM, 'Homoskedastic Y-noise (constant X2)', "./results/hetero/", "homoskedastic z=1")
    make_table(IRM, err_IRM, ERM, err_ERM, "./results/hetero/", "homoskedastic z=1")

def plot_not_scrambled():
    IRM = [0.00264055, 0.00841526, 0.285087, 0.280217]
    ERM = [0.00982805, 0.0371659, 0.817322, 0.774902]

    plot_bar(IRM, ERM, 'Not scrambled', "./results/scramble/", "not_scramble")


def plot_scrambled():
    IRM = [0.94698, 1.0902, 0.798489, 0.84677]
    ERM = [0.952655, 1.12906, 0.912988, 0.896329]

    plot_bar(IRM, ERM, 'Scrambled', "./results/scramble/", "scramble")


def plot_bar(IRM, err_IRM, ERM, err_ERM, title, repo, fileName):
    # set width of bar
    barWidth = 0.25

    # Set position of bar on X axis
    br1 = np.arange(4)
    br2 = [x + barWidth for x in br1]

    # Make the plot
    plt.bar(br1, IRM, yerr=err_IRM, width=barWidth, label='IRM')
    plt.bar(br2, ERM, yerr=err_ERM, width=barWidth, label='ERM')

    # Adding Xticks
    plt.title(title, fontsize=18)
    plt.xlabel('Data distribution shift', fontsize=14)
    plt.ylabel('Model estimation error', fontsize=14)
    plt.xticks([r + (barWidth / 2) for r in range(4)],
               ['CS', 'CF', 'AC', 'HB'])

    plt.tight_layout()
    plt.legend()
    os.makedirs(repo, exist_ok=True)
    plt.savefig(repo + fileName + ".png")
    plt.clf()


def make_table(IRM, err_IRM, ERM, err_ERM, repo, fileName):
    shifts = ["CS", "CF", "AC", "HB"]
    rows = []

    for i in range(4):
        rows.append(["ERM", shifts[i],
                     "{:.4f}".format(round(float(ERM[i]), 4)) + " \pm " + "{:.4f}".format(round(float(err_ERM[i]), 4))])
        rows.append(["IRM", shifts[i],
                     "{:.4f}".format(round(float(IRM[i]), 4)) + " \pm " + "{:.4f}".format(round(float(err_IRM[i]), 4))])

    table = tbl.tabulate(rows, tablefmt='latex_raw',
                         headers=["Method", "Shift", "Model estimation error"])

    # Write table to file
    os.makedirs(repo, exist_ok=True)
    f = open(repo + fileName + ".tex", "w")
    f.write(table)
    f.close()

if __name__ == "__main__":
    plot_homo_z_one()
    plot_homo()
