"""
A first iteration of performing a linear combination to compute the fraction of proposed
states.

Author: Joseph D. Yesselman

"""
import re
import sys
import logging
import os
import click
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FixedLocator, FixedFormatter
# logging #############################################################################

APP_LOGGER_NAME = "DMS-LINEAR-COMBO"


def setup_applevel_logger(logger_name=APP_LOGGER_NAME, is_debug=False, file_name=None):
    """
    Set up the logger for the app
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG if is_debug else logging.INFO)

    formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")

    # pylint: disable=C0103
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(sh)

    if file_name:
        # pylint: disable=C0103
        fh = logging.FileHandler(file_name)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def get_logger(module_name):
    """
    Get the logger for the module
    """
    return logging.getLogger(APP_LOGGER_NAME).getChild(module_name)


# global variables ####################################################################

log = get_logger("")


# subselect data ######################################################################


def str_to_range(x):
    """
    Convert a string representation of a range of numbers to a list of integers.

    Given a string representation of a range of numbers, this function returns a
    list of integers corresponding to the numbers in the range. The string can
    contain single numbers separated by commas, and ranges of numbers separated by
    a hyphen.

    :param x: A string representation of a range of numbers.
    :return: A list of integers corresponding to the numbers in the range.
    """
    return sum(
        (
            i if len(i) == 1 else list(range(i[0], i[1] + 1))
            for i in (
            [int(j) for j in i if j] for i in re.findall(r"(\d+),?(?:-(\d+))?", x)
        )
        ),
        [],
    )


def subselect_data(data, includes=None):
    if includes is None:
        return data
    for i in range(len(data)):
        if i not in includes:
            data[i] = 0
    return data


# util functions ######################################################################


def get_data_from_file(path, subselect_res, is_dc221=False):
    """
    Get the data from a file
    """
    df = pd.read_csv(path, sep="\t")
    data = np.array(df["Mismatches"])
    includes = str_to_range(subselect_res)
    if len(includes) == 0:
        includes = None
    data = subselect_data(data, includes)
    if is_dc221:
        data = np.insert(data, 11, 0)
    # set the mutation positions to 0
    data[14] = 0  # mod
    data[11] = 0
    return data


# plotting functions ##################################################################


def plot_data(seq, data_wt, data_c224a, data_dc221):
    log.info("plotting starting data -> results/mutation_comparison_plots.ps")
    fig, ax = plt.subplots(3, 1, figsize=(10, 5))
    ax[0].set_title("WT")
    ax[0].step(range(len(seq)), data_wt, where='mid', color='black', linewidth=0.5, label="wild-type", alpha=1)
    ax[0].set_xticks(range(0, len(seq), 1))
    ax[0].set_xticklabels(seq)
    ax[1].set_title("C224A")
    ax[1].step(range(len(seq)), data_c224a, where='mid', color='magenta', linewidth=0.5, label="mut", alpha=1)
    ax[1].set_xticks(range(0, len(seq), 1))
    ax[1].set_xticklabels(seq)
    ax[2].set_title("dC221")
    ax[2].step(range(len(seq)), data_dc221, where='mid', color='green', linewidth=0.5, label="mut", alpha=1)
    ax[2].set_xticks(range(0, len(seq), 1))
    ax[2].set_xticklabels(seq)
    fig.subplots_adjust(hspace=0.5)
    fig.savefig("results/mutation_comparison_plots.ps")


def calculate_statistics(best_data, data_wt):
    # Calculate chi-square
    chi_sq = np.sum((data_wt - best_data) ** 2)

    # Calculate BIC
    n = len(data_wt)
    k = 2  # number of parameters (percent c224a, percent dc221)
    bic = n * np.log(chi_sq / n) + k * np.log(n)

    # Calculate AIC
    aic = n * np.log(chi_sq / n) + 2 * k

    # Calculate R2
    mean_data_wt = np.mean(data_wt)
    SS_res = np.sum((data_wt - best_data) ** 2)
    SS_tot = np.sum((data_wt - mean_data_wt) ** 2)
    r2 = 1 - SS_res / SS_tot

    return chi_sq, bic, aic, r2

def plot_best_fit(best_percent, best_data, seq, data_wt):
    chi_sq, bic, aic, r2 = calculate_statistics(best_data, data_wt)

    log.info("plotting best fit -> results/linear_combo_fit.ps")
    fig, ax = plt.subplots(2, 1, figsize=(10, 5))
    ax[0].set_title(f"c224a % = {best_percent}, dc221 % = {100 - best_percent} ")
    ax[0].step(range(len(seq)), data_wt, where='mid', color='black', linewidth=0.5, label="wild-type", alpha=1)
    ax[0].step(range(len(seq)), best_data, where='mid', color='cornflowerblue', linewidth=0.5, label="linear-comb", alpha=1)
    ax[0].set_xticks(range(0, len(seq), 1))
    ax[0].set_xticklabels(seq)
    ax[0].set_ylim([-0.00170, 0.07])
    yticks = [i / 100 for i in range(0, 8, 1)]
    yticklabels = ['{:.2f}'.format(i) for i in yticks]
    ax[0].yaxis.set_major_locator(FixedLocator(yticks))
    ax[0].yaxis.set_major_formatter(FixedFormatter(yticklabels))
    ax[0].legend()
    residual = data_wt - best_data
    chi_square = np.sum(residual ** 2)
    ax[1].step(range(len(seq)), residual, where='mid', color='orange', linewidth=0.5, label="residuals", alpha=1)
    ax[1].set_xticks(range(0, len(seq), 1))
    ax[1].set_xticklabels(seq)
    ax[1].set_ylim([-0.03, 0.03])
    ax[1].text(0.05, 0.8, f"Chi-square = {chi_square:}", transform=ax[1].transAxes)
    ax[1].text(0.05, 0.5, f"R2 = {r2:}", transform=ax[1].transAxes)
    fig.subplots_adjust(hspace=0.2)
    fig.savefig("results/linear_combo_fit.ps")

# main  ##############################################################################


@click.command()
@click.option("--wt", help="path to wild-type data assumed tsv format", required=True)
@click.option("--c224a", help="path to c224a data assumed tsv format", required=True)
@click.option("--dc221", help="path to dc221 data assumed tsv format", required=True)
@click.option("--subselect-res", default="", help="subselect data")
def main(wt, c224a, dc221, subselect_res):
    """
    A simple script to compute the fraction of each state of the 7SK SL3 system
    """
    # setup logging
    setup_applevel_logger()
    log.info(f"wild-type data path: {wt}")
    log.info(f"c224a data path: {c224a}")
    log.info(f"dc221 data path: {dc221}")
    log.info("results will be saved in results/")
    os.makedirs("results", exist_ok=True)
    # the wild-type sequence
    seq = (
        "CCCUGCUAGAACCUCCAAACAAGCUCUCAAGGUCCAUUUGUAGGAGAACGUAGGG"
    )
    # setup data from tsv files
    data_wt = get_data_from_file(wt, subselect_res)
    data_c224a = get_data_from_file(c224a, subselect_res)
    data_dc221 = get_data_from_file(dc221, subselect_res, True)

    # weight the wt data and mutant data by the same factor using the wt data
    ratio_c224a = np.mean(data_wt) / np.mean(data_c224a)
    ratio_dc221 = np.mean(data_wt) / np.mean(data_dc221)

    data_c224a_weighted = data_c224a * ratio_c224a
    data_dc221_weighted = data_dc221 * ratio_dc221

    # compute the best linear fit and mean absolute error of the fit
    best_diff = 10000
    best_mae = 10000
    best_data = None
    best_percent = 0
    best_mae_diff = 10000
    for i in range(0, 56, 1):
        percent = i / 100
        data = percent * data_c224a_weighted + (1 - percent) * data_dc221_weighted
        diff = np.sum(np.abs(data_wt - data))
        mae = np.mean(np.abs(data_wt - data))
        if diff < best_diff:
            best_diff = diff
            best_mae_diff = mae
            best_percent = i
            best_data = data
            best_mae = mae
        elif diff == best_diff and mae < best_mae_diff:
            best_mae_diff = mae
            best_percent = i
            best_data = data
            best_mae = mae
    log.info(f"best fit: c224a % = {best_percent}, dc221 % = {100 - best_percent} ")
    log.info(f"sum of residuals: {best_diff}")
    log.info(f"mean absolute error for best fit: {best_mae}")
    # output data in tsv files
    log.info("linear combination fit -> results/linear_combo_profile.tsv")
    f = open("results/linear_combo_profile.tsv", "w")
    f.write("Position\tMismatches\n")
    for i, d in enumerate(data):
        f.write(f"{i + 1}\t{d}\n")
    f.close()
    log.info("residuals -> results/residuals.tsv")
    f = open("results/residuals.tsv", "w")
    f.write("Position\tMismatches\n")
    for i, d in enumerate(data):
        f.write(f"{i + 1}\t{data_wt[i] - d}\n")
    f.close()
    # store mutation comparison in csv file
    f = open("results/pre-processed.csv", "w")
    f.write("Position,data_wt,data_c224a,data_dc221\n")
    # Write the data
    for i in range(len(data_wt)):
        f.write(f"{i + 1},{data_wt[i]},{data_c224a_weighted[i]},{data_dc221_weighted[i]}\n")
    # Close the file
    f.close()

    plot_data(seq, data_wt, data_c224a_weighted, data_dc221_weighted)
    plot_best_fit(best_percent, best_data, seq, data_wt)


# pylint: disable=no-value-for-parameter
if __name__ == "__main__":
    main()
