import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.size'] = 18 
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Computer Modern']

def plot(triang, list_fields, list_labels,
         vmin=None, vmax=None, filename=None,
         figsize=None):
    if vmin is not None and vmax is not None:
        levels = np.linspace(vmin-0.01*np.abs(vmin), vmax+0.01*np.abs(vmax), 20)
    elif vmin is None or vmax is None:
        levels = 20

    if figsize is None:
        figsize = (5*len(list_fields), 3)
    if len(list_fields) > 1:
        fig, axs = plt.subplots(1, len(list_fields),
                figsize=figsize)
        for field, label, ax in zip(list_fields, list_labels, axs):

            a0 = ax.tricontourf(triang, field,
                    levels=levels, cmap='viridis')
            ax.set_title(label)
        fig.colorbar(a0, ax=axs.ravel().tolist())
        #fig.tight_layout()
        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()
    else:
        fig, ax = plt.subplots(1, 1,
                figsize=(5, 3))
        a0 = ax.tricontourf(triang, list_fields[0],
                    levels=levels, cmap='viridis')
        ax.set_title(list_labels[0])
        fig.colorbar(a0, ax=ax)
        fig.tight_layout()
        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()


def compute_exact_correction(pod, snaps):
    return snaps - pod.expand(pod.reduce(snaps))


