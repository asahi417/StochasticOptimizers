import numpy as np
import matplotlib.pyplot as plt

from scipy import stats


class PlotCurves(object):
    """Plot learning curve

     Parameters
    ----------
    yy: dict of error {algorithm_name:[error]}
    data_size: full datasize used in learning
    split: splitting number
    rounds: independent trial number
    data_path: where to save the figure
    """

    def __init__(self, split, data_order, data_size, rounds, test_size,
                 data_classes=1, remove=[]):
        self.split = split
        self.data_order = data_order
        self.data_size = data_size
        self.data_classes = data_classes
        self.rounds = rounds
        self.remove = remove
        self.test_size = test_size

    def sparsity(self, yy, fill_ci=True, sign='.-'):
        bias = self.data_classes*self.data_order
        fig = plt.figure()
        train_size = np.ceil(self.data_size*(1-self.test_size)).astype(int)
        xx = np.linspace(0, train_size, self.split)
        for k in yy.keys():
            if k in self.remove:
                continue
            m = 1-np.array(yy[k]["mean"])/bias
            m = m*100
            if sign is not None:
                plt.plot(xx, m, sign, label=k)
            else:
                plt.plot(xx, m, label=k)
            if fill_ci:
                std = yy[k]["std"]
                if np.max(std) != 0:
                    sem = np.array(std)/(bias*(self.rounds)**0.5)
                    ci = stats.t.interval(0.95, train_size-1, loc=m, scale=sem)
                    plt.fill_between(xx, ci[0], ci[1], alpha=0.1, color="r")
        plt.xlim(xmin=0, xmax=train_size)
        plt.xlabel("Iteration")
        plt.grid(True)
        plt.ylim(ymin=0, ymax=100)
        return fig

    def error(self, yy, fill_ci=True, ylabel=None, sign='.-'):
        fig = plt.figure()
        train_size = np.ceil(self.data_size*(1-self.test_size)).astype(int)
        xx = np.linspace(0, train_size, self.split)
        for k in yy.keys():
            if k in self.remove:
                continue
            m = np.array(yy[k]["mean"])
            if sign is not None:
                plt.plot(xx, m, sign, label=k)
            else:
                plt.plot(xx, m, label=k)
            if fill_ci:
                std = yy[k]["std"]
                if np.max(std) != 0:
                    sem = np.array(std)/(self.rounds)**0.5
                    ci = stats.t.interval(0.95, train_size-1, loc=m, scale=sem)
                    plt.fill_between(xx, ci[0], ci[1], alpha=0.1, color="r")
        plt.xlim(xmin=0, xmax=train_size)
        plt.xlabel("Iteration")
        plt.grid(True)
        if ylabel is None:
            plt.ylabel("Error Rate")
        else:
            plt.ylabel(ylabel)
        return fig
