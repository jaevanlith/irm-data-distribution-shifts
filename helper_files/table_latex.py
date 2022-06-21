import os

import tabulate as tbl


def convert_to_latex(main_path, exps_paths, shifts):
    for exps_path in exps_paths:
        for s in shifts:
            path = main_path + exps_path + "AVG/txt/"
            rows = []
            f = open(path + s + "-regression.txt", "r")
            f.readline()
            f.readline()

            for line in f:
                row = line.split()
                if len(row) != 0:
                    if row[2] == 'between':
                        rows = []
                        f.readline()
                    else:
                        row[2] = "{:.4f}".format(round(float(row[2]), 4)) + " $\pm$ " + "{:.4f}".format(
                            round(float(row[3]), 4))
                        rows.append(row[0:3])

            read_errs(rows, s, "CAU", main_path, exps_path)
            read_errs(rows, s, "NON", main_path, exps_path)

            table = tbl.tabulate(rows, tablefmt='latex_raw',
                                 headers=["Method", "Number of samples", "MER Average", "MER Causal", "MER Non-causal"])
            # Write table to file
            os.makedirs(path + "tex/", exist_ok=True)
            f = open(path + "tex/" + s + "-regression.tex", "w")
            f.write(table)
            f.close()


def read_errs(rows, s, w, main_path, exps_path):
    path = main_path + exps_path + w + "/txt/"
    f = open(path + s + "-regression.txt", "r")
    f.readline()
    f.readline()

    i = 0
    for line in f:
        row = line.split()
        if len(row) != 0:
            if row[2] == 'between':
                rows = []
                f.readline()
            else:
                row[2] = "{:.4f}".format(round(float(row[2]), 4)) + " $\pm$ " + "{:.4f}".format(round(float(row[3]), 4))
                if i < len(rows):
                    rows[i].append(row[2])
                    i += 1


if __name__ == "__main__":
    main_paths = ["Deviation/", "Environment Shift/", "Sample Complexity/"]
    exps_paths = ["Hetero/", "Homo/"]
    shifts = ["CS", "CF", "AC", "HB"]

    for main_path in main_paths:
        if main_path == "Sample Complexity/":
            exps_paths = [""]

        main_path = "C:/Users/boeki/OneDrive/Documenten/TU/BEP/Final Results/" + main_path
        convert_to_latex(main_path, exps_paths, shifts)
