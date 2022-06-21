# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import itertools
import os
import random

from sem import DataModel
from models import *

import argparse
import torch
import numpy
import matplotlib.pyplot as plt
import tabulate as tbl
from scipy import stats


def pretty(vector):
    vlist = vector.view(-1).tolist()
    return "[" + ", ".join("{:+.3f}".format(vi) for vi in vlist) + "]"


def errors(w, w_hat):
    w = w.view(-1)
    w_hat = w_hat.view(-1)

    i_causal = torch.where(w != 0)[0].view(-1)
    i_noncausal = torch.where(w == 0)[0].view(-1)

    if len(i_causal):
        error_causal = (w[i_causal] - w_hat[i_causal]).pow(2).mean()
        error_causal = error_causal.item()
    else:
        error_causal = 0

    if len(i_noncausal):
        error_noncausal = (w[i_noncausal] - w_hat[i_noncausal]).pow(2).mean()
        error_noncausal = error_noncausal.item()
    else:
        error_noncausal = 0

    return error_causal, error_noncausal


def run_experiment(args):
    if args["seed"] >= 0:
        torch.manual_seed(args["seed"])
        numpy.random.seed(args["seed"])
        torch.set_num_threads(1)

    if args["setup_sem"] == "chain":
        setup_str = "shift={}_hetero={}_scramble={}".format(
            args["shift"],
            args["setup_hetero"],
            args["setup_scramble"])
    elif args["setup_sem"] == "icp":
        setup_str = "sem_icp"
    else:
        raise NotImplementedError

    all_methods = {
        "ERM": EmpiricalRiskMinimizer,
        "IRM": InvariantRiskMinimization,
        "ICP": InvariantCausalPrediction,
        "REX": REXv21,
    }

    if args["methods"] == "all":
        methods = all_methods
    else:
        methods = {m: all_methods[m] for m in args["methods"].split(',')}

    all_sems = []
    all_environments = []
    results = {}

    for key in all_methods.keys():
        results[key] = {
            "errs_causal": [],
            "errs_noncausal": []
        }

    for rep_i in range(args["n_reps"]):
        if args["setup_sem"] == "chain":
            sem = DataModel(dim=args["dim"],
                            shift=args["shift"],
                            dim_x=args["dim_x"],
                            dim_z=args["dim_z"],
                            ones=args["setup_ones"],
                            scramble=args["setup_scramble"],
                            hetero=args["setup_hetero"],
                            confounder_on_x=args["setup_confounder_on_x"])

            env_list = [float(e) for e in args["env_list"].split(",")]
            environments = [sem(args["n_samples"], e) for e in env_list]
        else:
            raise NotImplementedError

        all_sems.append(sem)
        all_environments.append(environments)

    i = 1
    for sem, environments in zip(all_sems, all_environments):
        print("Repetition: " + str(i))
        i += 1

        sem_solution, sem_scramble = sem.solution()

        for method_name, method_constructor in methods.items():
            print("Running " + method_name + "...")
            method = method_constructor(environments, args)

            method_solution = sem_scramble @ method.solution()

            err_causal, err_noncausal = errors(sem_solution, method_solution)

            results[method_name]["errs_causal"].append(err_causal)
            results[method_name]["errs_noncausal"].append(err_noncausal)

    return results


def sample_complexity(shifts, methods):
    """
    Experiment on all shifts to yield model estimation error vs number of samples.
    """
    print("Running sample complexity experiment...")

    n_samples = [50, 200, 500, 1000, 1500, 2000]

    res = initResultsArray(methods)

    # Loop over shifts
    for s in shifts:
        print("--  SHIFT: " + s)
        for n in n_samples:
            print("- # SAMPLES: " + str(n))
            settings = getSettings(n_samples=n, n_reps=2, n_iterations=1, shift=s, methods=",".join(methods))

            save_results(methods, res, settings, s)

        export_results("sample_complexity", res, methods, n_samples, s, "heteroskedastic",
                       "Number of samples")


def deviation_training_environments(shifts, methods):
    """
    Experiment on all shifts to yield model estimation error vs mutual deviation of training envs.
    """
    print("Running training environments experiment...")

    first_env = 0.2
    devs = [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0, 15.0]
    hs = [0, 1]

    # Hetero- or homoskedastic Y-noise
    for h in hs:
        res = initResultsArray(methods)
        # Loop over shifts
        for s in shifts:
            print("--  SHIFT: " + s)

            for dev in devs:
                env_list = [first_env, first_env + dev, first_env + 2*dev]
                print("- ENVIRONMENTS: " + str(env_list))

                settings = getSettings(n_reps=2, n_iterations=1, setup_hetero=0,
                                       shift=s, methods=",".join(methods), env_list=",".join(map(str, env_list)))
                save_results(methods, res, settings, s)

            y_noise = "homoskedastic"
            if h:
                y_noise = "heteroskedastic"

            export_results("deviation_training_environments", res, methods, devs, s, y_noise,
                           "Adaptation of training environment values")


def values_training_environments(shifts, methods):
    """
    Experiment on all shifts to yield model estimation error vs adaptation of values of training environments.
    """
    print("Running training environments shift experiment...")

    adaptations = [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0, 15.0]
    hs = [0, 1]

    # Hetero- or homoskedastic Y-noise
    for h in hs:
        res = initResultsArray(methods)

        # Loop over shifts
        for s in shifts:
            print("--  SHIFT: " + s)

            for adaptation in adaptations:
                env_list = [0.2 + adaptation, 2.0 + adaptation, 5.0 + adaptation]
                print("- ENVIRONMENTS: " + str(env_list))

                settings = getSettings(n_reps=2, n_iterations=1, setup_hetero=h,
                                       shift=s, methods=",".join(methods), env_list=",".join(map(str, env_list)))

                save_results(methods, res, settings, s)

            y_noise = "homoskedastic"
            if h:
                y_noise = "heteroskedastic"

            export_results("values_training_environments", res, methods, adaptations, s, y_noise,
                           "Adaptation of training environment values")


def y_noise(shifts, methods):
    """
    Experiment on hetero- and homoskedastic Y-noise to yield model estimation error vs data distribution shift.
    """
    hs = [0, 1, 2]

    for h in hs:
        res = initResultsArray(methods)
        IRM = []
        err_IRM = []
        ERM = []
        err_ERM = []

        # Loop over shifts
        for s in shifts:
            print("--  SHIFT: " + s)
            settings = getSettings(n_reps=2, n_iterations=1, shift=s, methods=",".join(methods), setup_hetero=h)

            save_results(methods, res, settings, s)

            IRM.append(res["AVG"]["IRM"][s]["mean"][0])
            err_IRM.append(res["AVG"]["IRM"][s]["std dev"][0])
            ERM.append(res["AVG"]["ERM"][s]["mean"][0])
            err_ERM.append(res["AVG"]["ERM"][s]["std dev"][0])

        y_noise = "Homoskedastic Y-noise"
        if h == 1:
            y_noise = "Heteroskedastic Y-noise"
        if h == 2:
            y_noise = "Homoskedastic Y-noise (constant X2)"

        plot_bar(IRM, err_IRM, ERM, err_ERM, y_noise, "./results/y_noise/figures/", y_noise)

        # Loop over avg, cau and non
        for t in res.keys():
            # Create table
            rows = []
            for s in shifts:
                for m in methods:
                    rows.append([m, s, res[t][m][s]["mean"][0], res[t][m][s]["std dev"][0]])
            table = tbl.tabulate(rows, headers=["Method", "Data distribution shift",
                                                "Model estimation error", "Standard error of measurement"])

            # Write table to file
            os.makedirs("./results/y_noise/txt/", exist_ok=True)
            f = open("./results/y_noise/txt/" + y_noise + ".txt", "a")
            f.write(table + "\n\n")
            f.close()


def getSettings(dim=10, dim_x=0, dim_z=0, n_samples=1000, n_reps=10, skip_reps=0, seed=0, print_vectors=1,
                n_iterations=100000, lr=0.001, verbose=0, methods="ERM,ICP,IRM", alpha=0.05,
                env_list=".2,2.,5.", setup_sem="chain", setup_ones=1, setup_hetero=1, setup_scramble=0,
                setup_confounder_on_x=0, shift="AC"):
    settings = {
        'dim': dim,
        'dim_x': dim_x,
        'dim_z': dim_z,
        'n_samples': n_samples,
        'n_reps': n_reps,
        'skip_reps': skip_reps,
        'seed': seed,
        'print_vectors': print_vectors,
        'n_iterations': n_iterations,
        'lr': lr,
        'verbose': verbose,
        'methods': methods,
        'alpha': alpha,
        'env_list': env_list,
        'setup_sem': setup_sem,
        'setup_ones': setup_ones,
        'setup_hetero': setup_hetero,
        'setup_scramble': setup_scramble,
        'setup_confounder_on_x': setup_confounder_on_x,
        'shift': shift
    }

    return settings


def initResultsArray(methods):
    '''
    Initialize array where the results are stored.
    '''
    # First = mean, second mean of causal, third mean of noncausal
    res = {"AVG": {}, "CAU": {}, "NON": {}}

    # Create dictonaries for the shifts
    for t in res.keys():
        for m in methods:
            res[t][m] = {"CS": {"mean": [], "std dev": []},
                            "CF": {"mean": [], "std dev": []},
                            "AC": {"mean": [], "std dev": []},
                            "HB": {"mean": [], "std dev": []}}

    return res


def save_results(methods, res, settings, s):
    results = run_experiment(settings)

    for m in methods:
        res["AVG"][m][s]["mean"].append(numpy.mean(results[m]["errs_causal"] + results[m]["errs_noncausal"]))
        res["AVG"][m][s]["std dev"].append(stats.sem(results[m]["errs_causal"] + results[m]["errs_noncausal"]))

        res["CAU"][m][s]["mean"].append(numpy.mean(results[m]["errs_causal"]))
        res["CAU"][m][s]["std dev"].append(stats.sem(results[m]["errs_causal"]))

        res["NON"][m][s]["mean"].append(numpy.mean(results[m]["errs_noncausal"]))
        res["NON"][m][s]["std dev"].append(stats.sem(results[m]["errs_noncausal"]))


def export_results(exp_path, res, methods, x_axis, s, y_noise, x_label):
    # Loop over avg, cau and non
    for t in res.keys():
        path = "./results/" + exp_path + "/" + y_noise + "/" + t + "/"
        # Create plot
        marker = itertools.cycle(('o', 'D', 'x', '^'))
        for m in methods:
            lb = np.subtract(res[t][m][s]["mean"], res[t][m][s]["std dev"])
            ub = np.add(res[t][m][s]["mean"], res[t][m][s]["std dev"])
            plt.plot(x_axis, res[t][m][s]["mean"], label=m, marker=next(marker))
            plt.fill_between(x_axis, ub, lb, alpha=0.25)

        plt.title(s + "-regression " + y_noise + " Y-noise")
        plt.xlabel(x_label)
        plt.ylabel("Model estimation error")
        plt.legend(loc="best")
        os.makedirs(path + "figures/", exist_ok=True)
        plt.savefig(path + "figures/" + s + "-regression.png")
        plt.clf()

        # Create table
        rows = []
        for i in range(len(x_axis)):
            for m in methods:
                rows.append([m, x_axis[i], res[t][m][s]["mean"][i], res[t][m][s]["std dev"][i]])
        table = tbl.tabulate(rows, headers=["Method", "Deviation between training environments",
                                            "Model estimation error", "Standard error of measurement"])

        # Write table to file
        os.makedirs(path + "txt/", exist_ok=True)
        f = open(path + "txt/" + s + "-regression.txt", "a")
        f.write(table + "\n\n")
        f.close()


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Invariant regression')
    parser.add_argument('--experiment', type=str, default="SampleComplexity")
    args = dict(vars(parser.parse_args()))

    # The shifts and methods for all experiments
    shifts = ["CS", "CF", "AC", "HB"]
    all_methods = ["ERM", "IRM", "ICP", "REX"]
    methods = all_methods[0:2]  # ERM and IRM

    if args['experiment'] == "sample_complexity":
        sample_complexity(shifts, methods)
    elif args['experiment'] == "deviation_training_environments":
        deviation_training_environments(shifts, methods)
    elif args['experiment'] == "values_training_environments":
        values_training_environments(shifts, methods)
    elif args['experiment'] == "y_noise":
        y_noise(shifts, methods)
    else:
        print('Experiment not available')
