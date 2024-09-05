import matplotlib.pyplot as plt

def plot(triang, list_fields, list_labels, filename=None):
    if len(list_fields) > 1:
        fig, axs = plt.subplots(1, len(list_fields),
                figsize=(5*len(list_fields), 3))
        for field, label, ax in zip(list_fields, list_labels, axs):
            a0 = ax.tricontourf(triang, field,
                    levels=16, cmap='viridis')
            ax.set_title(label)
            fig.colorbar(a0, ax=ax)
        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()
    else:
        fig, ax = plt.subplots(1, 1,
                figsize=(5, 3))
        a0 = ax.tricontourf(triang, list_fields[0],
                    levels=16, cmap='viridis')
        ax.set_title(list_labels[0])
        fig.colorbar(a0, ax=ax)
        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()


def compute_exact_correction(pod, pod_big, snaps):
    return pod_big.expand(pod_big.reduce(snaps)) - pod.expand(pod.reduce(snaps))


