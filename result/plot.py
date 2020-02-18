import matplotlib as mpl
import numpy as np
import tensorflow as tf
mpl.use('Agg')

from matplotlib import pyplot as plt

mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["font.family"] = "DejaVu Serif"


def main(unused_argv):
    models = ['single', 'ewc', 'multi', 'meta']
    colors = {'single': 'C0',
              'ewc': 'C1',
              'multi': 'C2',
              'meta': 'C3'}

    evoplot = {}
    n_task = 10
    path = 'result/'
    for i in range(n_task):
        evoplot[i] = {}
        for model in models:
            data = np.load(path + model + '.npy')
            evoplot[i][model] = data[:, i]

    for i in range(n_task):
        plt.figure(figsize=(10, 8))
        for model in models:
            if model in evoplot[i]:
                x = np.arange(len(evoplot[i][model]))
                x = (x - x.min()) / (x.max() - x.min()) * n_task
                plt.plot(x, evoplot[i][model], label=model, color=colors[model], lw=1)
                plt.xticks(range(0, n_task + 1, 2))

        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.ylabel('Accuracy', fontsize=16)
        plt.xlabel('tasks', fontsize=16)
        plt.title("Task %d's accuracy" % int(i + 1), fontsize=16)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig('figure/evo' + str(i) + '.pdf', bbox_inches='tight')


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
