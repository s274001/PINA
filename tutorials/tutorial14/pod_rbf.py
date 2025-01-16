from math import sqrt
import time
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
matplotlib.rcParams['font.serif'] = ['Computer Modern Roman']
map = plt.get_cmap('Set1')
matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler('color', map.colors)

# torch.manual_seed(20)

def err(snap, snap_pred):
    # relative errors
    errs = torch.linalg.norm(snap - snap_pred, dim=1)/torch.linalg.norm(snap, dim=1)
    err_med = float(torch.median(errs))
    err = float(torch.mean(errs))
    return err, err_med

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
    # data = NavierStokesDataset()
    data = LidCavity()
    snapshots = data.snapshots[field]
    params = data.params
    Ndof = snapshots.shape[1]
    Nparams = params.shape[1]

    # Divide dataset into training and testing
    params_train, params_test, snapshots_train, snapshots_test = train_test_split(
            params, snapshots, test_size=0.20, shuffle=True, random_state=42)

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
    print(snapshots_train.shape, snapshots_test.shape)

    # POD model
    rom = PODBlock(reddim)
    rom.fit(snapshots_train)
    #print(rom.reduce(snapshots_train[:3]))
    #exit()
    ##plot singular values and energy
    #values = rom.values.detach().numpy()
    #values *= values
    #energy = np.cumsum(values)/values.sum()
    #maxmode = 200
    #fig,[ax1,ax2] = plt.subplots(1,2)
    #ax1.semilogy(list(range(maxmode)),values[:maxmode],'o',color='steelblue')
    #ax1.set_xlabel('Rank')
    #ax1.set_ylabel('Singular values')
    #ax1.grid()
    ##fig, ax = plt.subplots()
    #ax2.plot(list(range(maxmode)), energy[:maxmode],'o',color='steelblue')
    #ax2.set_xlabel('Rank')
    #ax2.set_ylabel('Energy')
    #fig.tight_layout()
    #ax2.grid()
    #plt.show()
    #print(energy[:140])
    #print(np.argmin(np.abs(energy-0.99)))
    #exit()
    #print(rom.basis.size())
    #red = rom.reduce(snapshots_train)
    #print(f'mean = {torch.mean(red,dim=0)}\nstd = {torch.std(red,dim=0)}')

    #plot the modes
    modes = rom.basis.T
    print(modes.shape)
    vmin = modes.min()
    vmax = modes.max()
    list_fields = [modes.detach().numpy()[:,i].reshape(-1) for i in range(reddim)]
    list_labels = [f'Mode {i+1}' for i in range(reddim)]
    plot(data.triang,list_fields,list_labels,
         filename = 'img/pod_modes_cavity',
         vmin = vmin,
         vmax = vmax)
    # for i in range(reddim):
    #    modes = rom.basis.T
    #    list_fields = [modes.detach().numpy()[:, i].reshape(-1)]
    #    #list_fields = [modes.detach().numpy()[:, i].reshape(-1)
    #    #        for i in range(reddim)]
    #    list_labels = [f'Mode {i+1}']# for i in range(reddim)]
    #    plot(data.triang,list_fields, list_labels)#,filename=f'img/modes{i+1}_pod')
    #exit()

    #plot the coefficients wrt the first 3 modes
    #snapshots= LabelTensor(torch.tensor(snapshots, dtype=torch.float32),
    #        labels=[f's{i}' for i in range(snapshots.shape[1])])
    #params= LabelTensor(torch.tensor(params, dtype=torch.float32),
    #        labels=['mu'])
    #rom.fit(snapshots)
    #rom_rbf = PODRBF(pod_rank=reddim, rbf_kernel='inverse_multiquadric',epsilon=100.)#,smoothing=1e-1)
    #rom_rbf.fit(params_train, snapshots_train)
    #t= 0
    #for i in range(100):
    #    t1 = time.time()
    #    rom_rbf.rbf(params_train)
    #    t2 = time.time()
    #    t += (t2-t1)
    #print(t/(100.*240))
    #exit()
    #fig = plt.figure()
    #ax = fig.add_subplot(projection='3d')
    #ax = fig.add_subplot()
    #red = rom.reduce(snapshots)
    #scatter = ax.scatter(red[:,0],red[:,1],red[:,2],marker='.',c=params*0.1*1e5,alpha=0.8)
    #scatter = ax.scatter(red[:,0],red[:,1],marker='.',c=params*0.1*1e5,alpha=0.8)
    #scatter = ax.scatter(params, red[:,0],marker='.',c=params*0.1*1e5,alpha=0.8)
    #red = rom.reduce(snapshots_test)
    #ax.scatter(red[:,0],red[:,1],red[:,2],marker='o',c='r',alpha=1,label='Test real')
    #ax.scatter(red[:,0],red[:,1],marker='o',c='r',alpha=0.6,label='Test real')
    #ax.scatter(params_test,red[:,0],marker='o',c='r',alpha=0.6,label='Test real')
    #red = rom_rbf.rbf(torch.linspace(0.1,1,1000).unsqueeze(1))
    #ax.plot(red[:,0],red[:,1],-red[:,2],ls='--',marker='',c='k',alpha=0.4)
    #ax.plot(red[:,0],red[:,1],ls='--',marker='',c='k',alpha=0.5)
    #ax.plot(torch.linspace(0.1,1,1000).unsqueeze(1),red[:,0],ls='--',marker='',c='k',alpha=0.7)
    #red = rom_rbf.rbf(params_test)
    #ax.scatter(red[:,0],red[:,1],-red[:,2],marker='^',c='orange',alpha=1,label='Test prediction')
    #ax.scatter(red[:,0],red[:,1],marker='^',c='orange',alpha=0.6,label='Test prediction')
    #ax.scatter(params_test,red[:,0],marker='^',c='orange',alpha=0.6,label='Test prediction')
    #ax.set_title("POD coefficients")
    #ax.legend()
    #cbar = fig.colorbar(scatter)
    #cbar.ax.set_title('Re')
    #ax.set_xlabel(r'$a_1$')
    #ax.set_ylabel(r'$a_2$')
    #fig.tight_layout()
    #for ang in range(0,360,5):
    #    ax.view_init(azim=ang)
    #    fig.savefig(f'img/movie/pod_proj{ang:03}.png')
    #plt.show()
    #exit()


    #ranks = list(range(1,10)) + list(range(10,121,10))
    ranks = [3]
    #errors_train_pod = np.zeros((len(ranks),240))
    #errors_train_rbf = np.zeros((len(ranks),240))
    #norms_train = torch.linalg.norm(snapshots_train,dim=1)
    #errors_test_pod = np.zeros((len(ranks),60))
    #errors_test_rbf = np.zeros((len(ranks),60))
    #norms_test = torch.linalg.norm(snapshots_test,dim=1)
    for i,reddim in enumerate(ranks):
        print(f'Iteration {i}, rank {reddim}')
        rom = PODBlock(reddim)
        rom.fit(snapshots_train)

        predicted_snaps_train = rom.expand(rom.reduce(snapshots_train))
        predicted_snaps_test = rom.expand(rom.reduce(snapshots_test))

        error_train, error_train_median = err(snapshots_train, predicted_snaps_train)
        error_test, error_test_median = err(snapshots_test, predicted_snaps_test)
        print('POD model')
        print('Train relative error:', error_train)
        print('Test relative error:', error_test)

        # POD-RBF model
        rom_rbf = PODRBF(pod_rank=reddim, rbf_kernel='inverse_multiquadric',epsilon=100.)#,smoothing=1e-1)
        rom_rbf.fit(params_train, snapshots_train)
        predicted_snaps_test_rbf = rom_rbf(params_test)
        predicted_snaps_train_rbf = rom_rbf(params_train)

        #errors_train_pod[i] = (torch.linalg.norm(predicted_snaps_train- snapshots_train,dim=1)/norms_train).detach().numpy()
        #errors_train_rbf[i] = (torch.linalg.norm(predicted_snaps_train_rbf - snapshots_train,dim=1)/norms_train).detach().numpy()
        #errors_test_pod[i] = (torch.linalg.norm(predicted_snaps_test- snapshots_test,dim=1)/norms_test).detach().numpy()
        #errors_test_rbf[i] = (torch.linalg.norm(predicted_snaps_test_rbf - snapshots_test,dim=1)/norms_test).detach().numpy()

        error_train_rbf, error_train_rbf_median = err(snapshots_train, predicted_snaps_train_rbf)
        error_test_rbf, error_test_rbf_median = err(snapshots_test, predicted_snaps_test_rbf)
        print('mean:')
        print(reddim,error_train,error_test,error_train_rbf,error_test_rbf)
        print('median:')
        print(reddim,error_train_median,error_test_median,error_train_rbf_median,error_test_rbf_median)
        print('POD-RBF')
        print('Train relative error:', error_train_rbf)
        print('Test relative error:', error_test_rbf)
    #np.save('rbf_errors_train',errors_train_rbf)
    #np.save('rbf_errors_test',errors_test_rbf)
    #np.save('pod_errors_train',errors_train_pod)
    #np.save('pod_errors_test',errors_test_pod)


    # Plot the results
    # fig, axs = plt.subplots(1, 3, figsize=(15, 3))
    # ind_test = 0
    # snap = snapshots_train[ind_test].detach().numpy().reshape(-1)
    # pred_snap = predicted_snaps_train[ind_test].detach().numpy().reshape(-1)
    # a0 = axs[0].tricontourf(data.triang, snap, levels=16,
    #         cmap='viridis')
    # axs[0].set_title('Truth (mu test={})'.format(params_train[ind_test].detach().numpy()[0]))
    # a1 = axs[1].tricontourf(data.triang, pred_snap, levels=16,
    #         cmap='viridis')
    # axs[1].set_title('Prediction (mu test={})'.format(params_train[ind_test].detach().numpy()[0]))
    # a2 = axs[2].tricontourf(data.triang, snap - pred_snap, levels=16,
    #         cmap='viridis')
    # axs[2].set_title('Error')
    # fig.colorbar(a0, ax=axs[0])
    # fig.colorbar(a1, ax=axs[1])
    # fig.colorbar(a2, ax=axs[2])
    # plt.show()

