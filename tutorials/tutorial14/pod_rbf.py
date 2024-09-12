from math import sqrt
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
import torch
from pina.solvers import ReducedOrderModelSolver as ROMSolver
from pina.model.layers import PODBlock, RBFBlock
from pina.model import FeedForward
from pina.geometry import CartesianDomain
from pina.problem import AbstractProblem, ParametricProblem
from pina import Condition, LabelTensor
from pina.callbacks import MetricTracker
from smithers.dataset import NavierStokesDataset, LidCavity
from utils import plot
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.size'] = 18 
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Computer Modern']
map = plt.get_cmap('Set1')
matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler('color', map.colors)

torch.manual_seed(20)

def err(snap, snap_pred):
    # relative errors
    errs = torch.linalg.norm(snap - snap_pred, dim=1)/torch.linalg.norm(snap, dim=1)
    err = float(torch.mean(errs))
    return err

class PODRBF(torch.nn.Module):
    """
    Non-intrusive ROM using POD as reduction and RBF as approximation.
    """
    def __init__(self, pod_rank, rbf_kernel,**kwargs):
        super().__init__()
        self.pod = PODBlock(pod_rank)
        self.rbf = RBFBlock(kernel=rbf_kernel,**kwargs)

    def fit(self, params, snaps):
        self.pod.fit(snaps)
        self.rbf.fit(params, self.pod.reduce(snaps))
        self.snapshots = snaps
        self.params = params

    def forward(self, param_test):
        snaps_pred_test = self.pod.expand(self.rbf(param_test))
        return snaps_pred_test

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Reduced Order Model Example')
    parser.add_argument('--reddim', type=int, default=3, help='Reduced dimension')
    parser.add_argument('--field', type=str, default='mag(v)', help='Field to reduce')

    args = parser.parse_args()
    field = args.field
    reddim = args.reddim

    # Import dataset
    #data = NavierStokesDataset()
    data = LidCavity()
    snapshots = data.snapshots[field]
    params = data.params
    Ndof = snapshots.shape[1]
    Nparams = params.shape[1]

    # Divide dataset into training and testing
    params_train, params_test, snapshots_train, snapshots_test = train_test_split(
            params, snapshots, test_size=0.2, shuffle=True, random_state=42)

    # From numpy to LabelTensor
    params_train = LabelTensor(torch.tensor(params_train, dtype=torch.float32),
            labels=['mu'])
    params_test = LabelTensor(torch.tensor(params_test, dtype=torch.float32),
            labels=['mu'])
    snapshots_train = LabelTensor(torch.tensor(snapshots_train, dtype=torch.float32),
            labels=[f's{i}' for i in range(snapshots_train.shape[1])])
    snapshots_test = LabelTensor(torch.tensor(snapshots_test, dtype=torch.float32),
            labels=[f's{i}' for i in range(snapshots_test.shape[1])])

    # Define ROM problem with only data
    class SnapshotProblem(ParametricProblem):
        input_variables = [f'mu']
        output_variables = [f's{i}' for i in range(Ndof)]
        parameter_domain = CartesianDomain({'mu':[0, 100]})
        conditions = {'data': Condition(input_points=params_train,
            output_points=snapshots_train)}

    problem = SnapshotProblem()
    #print(snapshots_train.shape, snapshots_test.shape)

    # POD model
    rom = PODBlock(reddim)
    rom.fit(snapshots_train)
    #values = rom.values.detach().numpy()
    #energy = np.cumsum(values)/values.sum()
    #maxmode = 200
    #fig,[ax1,ax2] = plt.subplots(1,2)
    #ax1.semilogy(list(range(maxmode)),values[:maxmode],'o',color='steelblue')
    #ax1.set_xlabel('Rank')
    #ax1.set_ylabel('Singular values')
    ##fig, ax = plt.subplots()
    #ax2.plot(list(range(maxmode)), energy[:maxmode],'o',color='steelblue')
    #ax2.set_xlabel('Rank')
    #ax2.set_ylabel('Energy')
    #fig.tight_layout()
    #plt.show()
    #print(np.argmin(np.abs(energy-0.999)))
    #exit()
    #print(rom.basis.size())
    #red = rom.reduce(snapshots_train)
    #print(f'mean = {torch.mean(red,dim=0)}\nstd = {torch.std(red,dim=0)}')

    #plot the modes
    #modes = rom.basis.T
    #list_fields = [modes.detach().numpy()[:, i].reshape(-1)
    #        for i in range(reddim)]
    #list_labels = [f'Mode {i}' for i in range(reddim)]
    #plot(data.triang,list_fields, list_labels,filename='img/modes_pod')

    #fig = plt.figure()
    #ax = fig.add_subplot(projection='3d')
    #scatter = ax.scatter(red[:,0],red[:,1],red[:,2],c=params_train)
    #ax.set_title("POD coefficients")
    #fig.colorbar(scatter)
    #fig.savefig('img/red_scatter.png')



    predicted_snaps_train = rom.expand(rom.reduce(snapshots_train))
    predicted_snaps_test = rom.expand(rom.reduce(snapshots_test))

    error_train = err(snapshots_train, predicted_snaps_train)
    error_test = err(snapshots_test, predicted_snaps_test)
    #print('POD model')
    #print('Train relative error:', error_train)
    #print('Test relative error:', error_test)

    # POD-RBF model
    rom_rbf = PODRBF(pod_rank=reddim, rbf_kernel='inverse_multiquadric',epsilon=100.)
    rom_rbf.fit(params_train, snapshots_train)
    predicted_snaps_test_rbf = rom_rbf(params_test)
    predicted_snaps_train_rbf = rom_rbf(params_train)

    error_train_rbf = err(snapshots_train, predicted_snaps_train_rbf)
    error_test_rbf = err(snapshots_test, predicted_snaps_test_rbf)
    print(reddim,error_train,error_test,error_train_rbf,error_test_rbf)
    #print('POD-RBF')
    #print('Train relative error:', error_train_rbf)
    #print('Test relative error:', error_test_rbf)

'''
    # Plot the results
    fig, axs = plt.subplots(1, 3, figsize=(15, 3))
    ind_test = 0
    snap = snapshots_train[ind_test].detach().numpy().reshape(-1)
    pred_snap = predicted_snaps_train[ind_test].detach().numpy().reshape(-1)
    a0 = axs[0].tricontourf(data.triang, snap, levels=16,
            cmap='viridis')
    axs[0].set_title('Truth (mu test={})'.format(params_train[ind_test].detach().numpy()[0]))
    a1 = axs[1].tricontourf(data.triang, pred_snap, levels=16,
            cmap='viridis')
    axs[1].set_title('Prediction (mu test={})'.format(params_train[ind_test].detach().numpy()[0]))
    a2 = axs[2].tricontourf(data.triang, snap - pred_snap, levels=16,
            cmap='viridis')
    axs[2].set_title('Error')
    fig.colorbar(a0, ax=axs[0])
    fig.colorbar(a1, ax=axs[1])
    fig.colorbar(a2, ax=axs[2])
    plt.show()
'''
