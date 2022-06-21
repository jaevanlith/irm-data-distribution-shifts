from sem import DataModel
import os
import matplotlib.pyplot as plt
import itertools


def make_plot_3d(shift, hetero):
    environment_devs = [0.2, 2.0, 5.0]
    sem = DataModel(2,
                    shift=shift,
                    ones=1,
                    scramble=0,
                    hetero=hetero,
                    confounder_on_x=0)

    env_list = [float(e) for e in environment_devs]
    environments = [sem.instantiate_training_environments(100, e) for e in env_list]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    marker = itertools.cycle(('D', 'o', 'x'))

    for i in range(len(environments)):
        xs = environments[i][0]
        ys = environments[i][1]
        zs = environments[i][2]
        ax.scatter(xs, zs, ys, marker=next(marker), label=environment_devs[i])

        ax.set_xlabel('X1', fontsize=14, fontweight='bold')
        ax.set_ylabel('X2', fontsize=14, fontweight='bold')
        ax.set_zlabel('Y', fontsize=14, fontweight='bold')

    plt.legend()
    plt.title(shift + ' data samples')
    plt.tight_layout()
    dir_str = 'hetero'
    if not hetero:
        dir_str = 'homo'

    os.makedirs("./results/data_plot/" + dir_str + "_3D/", exist_ok=True)
    plt.savefig("./results/data_plot/" + dir_str + "_3D/" + shift + ".png")


def make_plot_2d(shift, hetero):
    environment_devs = [0.2, 2.0, 5.0]
    sem = DataModel(2,
                    shift=shift,
                    ones=1,
                    scramble=0,
                    hetero=hetero,
                    confounder_on_x=0)

    env_list = [float(e) for e in environment_devs]
    environments = [sem.instantiate_training_environments(70, e) for e in env_list]

    plt.figure()

    marker = itertools.cycle(('D', 'o', 'x'))

    for i in range(len(environments)):
        xs = environments[i][0]
        zs = environments[i][2]
        plt.scatter(xs, zs, marker=next(marker), label=environment_devs[i])

        plt.xlabel('X1', fontsize=14, fontweight='bold')
        plt.ylabel('X2', fontsize=14, fontweight='bold')

    plt.legend()
    plt.title(shift + ' data samples')
    plt.tight_layout()
    dir_str = 'hetero'
    if not hetero:
        dir_str = 'homo'

    os.makedirs("./results/data_plot/" + dir_str + "_2D/", exist_ok=True)
    plt.savefig("./results/data_plot/" + dir_str + "_2D/" + shift + ".png")


if __name__ == '__main__':
    shifts = ['CS', 'CF', 'AC', 'HB']
    for s in shifts:
        make_plot_2d(s, 1)
        make_plot_2d(s, 0)
        make_plot_3d(s, 1)
        make_plot_3d(s, 0)