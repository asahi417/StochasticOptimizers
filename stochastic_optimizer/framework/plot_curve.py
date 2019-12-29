import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def plot_curve(yy,
               train_size,
               split,
               rounds,
               path: str,
               fill_ci: bool = True,
               ylabel=None,
               sign: str = '.-'):
    """Plot learning curve

     Parameters
    ----------
    yy: dict of error {algorithm_name:[error]}
    data_size: full datasize used in learning
    split: splitting number
    rounds: independent trial number
    data_path: where to save the figure

     Return
    ----------
    fig: figure instance
    """

    fig = plt.figure()
    xx = np.linspace(0, train_size, split)
    for k in yy.keys():
        m = np.array(yy[k]["mean"])
        if sign is not None:
            plt.plot(xx, m, sign, label=k)
        else:
            plt.plot(xx, m, label=k)
        if fill_ci:
            std = yy[k]["std"]
            if np.max(std) != 0:
                sem = np.array(std) / rounds ** 0.5
                ci = stats.t.interval(0.95, train_size-1, loc=m, scale=sem)
                plt.fill_between(xx, ci[0], ci[1], alpha=0.1, color="r")
    plt.xlim(xmin=0, xmax=train_size)
    plt.xlabel("Iteration")
    plt.grid(True)
    if ylabel is None:
        plt.ylabel("Misclassification Rate")
    else:
        plt.ylabel(ylabel)

    plt.rcParams['font.size'] = 14
    plt.legend(loc="best", fontsize=10)
    plt.savefig("%s/error.eps" % path, bbox_inches="tight")
    plt.savefig("%s/error.pdf" % path, bbox_inches="tight", transparent=True)

    return fig

