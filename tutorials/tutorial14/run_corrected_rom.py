import numpy as np
import argparse
from sklearn.model_selection import train_test_split
import torch
from pina.solvers import ReducedOrderModelSolver as ROMSolver
from pina.solvers import SupervisedSolver
from pina.model.layers import PODBlock, RBFLayer
from pina.model import FeedForward, DeepONet
from pina.geometry import CartesianDomain
from pina.problem import AbstractProblem, ParametricProblem
from pina.callbacks import MetricTracker
from pina import Condition, LabelTensor, Trainer, Plotter
from smithers.dataset import NavierStokesDataset
import matplotlib.pyplot as plt
from pod_rbf import err, PODRBF
from scaler import Scaler
from closure_models import  LinearCorrNet, DeepONetCorrNet
from corrected_rom import CorrectedROM
from sklearn.preprocessing import MinMaxScaler
import os

torch.manual_seed(20)

def plot(list_fields, list_labels, filename=None):
    fig, axs = plt.subplots(1, len(list_fields),
            figsize=(5*len(list_fields), 3))
    for field, label, ax in zip(list_fields, list_labels, axs):
        a0 = ax.tricontourf(data.triang, field,
                levels=16, cmap='viridis')
        ax.set_title(label)
        fig.colorbar(a0, ax=ax)
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()

def compute_exact_correction(pod, pod_big, snaps):
    return pod_big.expand(pod_big.reduce(snaps)) - pod.expand(pod.reduce(snaps))

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Reduced Order Model Example')
    parser.add_argument('--reddim', type=int, default=3, help='Reduced dimension')
    parser.add_argument('--bigdim', type=int, default=15, help='Bigger dimension')
    parser.add_argument('--field', type=str, default='mag(v)', help='Field to reduce')
    parser.add_argument('--load', help='Directory to save or load file', type=str)

    args = parser.parse_args()
    field = args.field
    reddim = args.reddim
    bigdim = args.bigdim


    # Import dataset
    data = NavierStokesDataset()
    snapshots = data.snapshots[field]
    coords = data.pts_coordinates
    params = data.params
    scaler_params = MinMaxScaler()
    params = scaler_params.fit_transform(params)
    Ndof = snapshots.shape[1]
    Nparams = params.shape[1]

    # Divide dataset into training and testing
    params_train, params_test, snapshots_train, snapshots_test = train_test_split(
            params, snapshots, test_size=0.2, shuffle=False, random_state=42)

    # From numpy to LabelTensor
    params_train = LabelTensor(torch.tensor(params_train, dtype=torch.float32),
            labels=['mu'])
    params_test = LabelTensor(torch.tensor(params_test, dtype=torch.float32),
            labels=['mu'])
    snapshots_train = LabelTensor(torch.tensor(snapshots_train, dtype=torch.float32),
            labels=[f's{i}' for i in range(snapshots_train.shape[1])])
    snapshots_test = LabelTensor(torch.tensor(snapshots_test, dtype=torch.float32),
            labels=[f's{i}' for i in range(snapshots_test.shape[1])])

    # define the exact correction term
    pod = PODBlock(reddim)
    pod.fit(snapshots_train)
    modes = pod.basis

    pod_big = PODBlock(bigdim)
    pod_big.fit(snapshots_train)

    # Plot the modes
    modes = pod.basis.T
    list_fields = [modes.detach().numpy()[:, i].reshape(-1)
            for i in range(modes.shape[1])]
    list_labels = [f'Mode {i}' for i in range(modes.shape[1])]
    #plot(list_fields, list_labels)

    # Compact the modes and the coords into a single tensor
    coords = torch.tensor(coords.repeat(reddim, 1).T)
    coords = LabelTensor(coords, ['x', 'y'])
    modes = LabelTensor(modes.reshape(-1, 1), 'modes')
    input_deeponet = coords.append(modes)

    # Compute space-dependent exact correction terms
    exact_correction = compute_exact_correction(pod, pod_big, snapshots_train)
    exact_correction = LabelTensor(exact_correction,
            [f's{i}' for i in range(Ndof)])

    # Define ROM problem with only data (parameters and snapshots)
    class SnapshotProblem(ParametricProblem):
        input_variables = ['mu']
        output_variables = snapshots_train.labels
        #output_variables += exact_correction.labels
        parameter_domain = CartesianDomain({'mu':[0, 100]})
        conditions = {#'data': Condition(input_points=params_train,
            #output_points=snapshots_train),
            'correction': Condition(input_points=params_train,
                output_points=exact_correction)}

    problem = SnapshotProblem()

    # Strategies for correction term
    # 1. POD + Linear
    #ann_corr = LinearCorrNet(pod, scaler=Scaler()) #linear model with least squares, not working

    # 2. POD + DeepONet modes@coeffs
    ann_corr = DeepONetCorrNet(pod, scaler=Scaler())

    #in_pts = problem.conditions["correction"].input_points
    #test = ann_corr(in_pts, ann_corr.coeff_corr)
    #print(in_pts.shape)
    #print(test.size())


    rom = CorrectedROM(problem=problem,
                reduction_network=pod,
                interpolation_network=RBFLayer(),
                correction_network=ann_corr,
                optimizer=torch.optim.Adam,
                optimizer_kwargs={'lr': 3e-3},
                scheduler=torch.optim.lr_scheduler.MultiStepLR,
                scheduler_kwargs={'gamma': 0.5 ,'milestones': list(range(6000,10000,2000))}
                )
    rom.neural_net["reduction_network"].fit(snapshots_train)
    rom.neural_net["interpolation_network"].fit(params_train,
                                                rom.neural_net["reduction_network"].reduce(snapshots_train))

    # Train the ROM to learn the correction term
    epochs = 10000

    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = Trainer(solver=rom, max_epochs=epochs, accelerator='cpu',
            default_root_dir=args.load, callbacks = [MetricTracker()],
                      batch_size=100)
    #params_train.to(device)
    #params_test.to(device)
    #snapshots_train.to(device)
    #snapshots_test.to(device)

    id_ = 90
    if args.load:
        rom = CorrectedROM.load_from_checkpoint(
                checkpoint_path=os.path.join(args.load,
                f'lightning_logs/version_{id_}/checkpoints/epoch={epochs-1}-step={epochs*4}.ckpt'),
                problem=problem, reduction_network=pod,
                interpolation_network=RBFLayer(),
                correction_network=ann_corr)
        rom.neural_net["reduction_network"].fit(snapshots_train)
        rom.neural_net["interpolation_network"].fit(params_train,
                                                rom.neural_net["reduction_network"].reduce(snapshots_train))

        # Plot the modes with the same function
        modes = rom.neural_net["correction_network"].transformed_modes
        modes = modes.detach().numpy()
        list_fields = [modes[:, i] for i in range(modes.shape[1])]
        list_labels = [f'Mode {i}' for i in range(modes.shape[1])]
        plot(list_fields, list_labels, 'img/transformed_modes_10kepochs')


        # Evaluate the ROM on train and test
        predicted_snaps_train = rom(params_train)
        predicted_snaps_test = rom(params_test)
        print(err(snapshots_train, predicted_snaps_train))
        print(err(snapshots_test, predicted_snaps_test))

        # Plot the test results (POD-RBF and POD-RBF corrected)
        ind_test = 0
        snap = snapshots_test[ind_test].detach().numpy().reshape(-1)
        pred_snap = predicted_snaps_test[ind_test].detach().numpy().reshape(-1)
        # POD-RBF (fit and predict)
        pod_rbf = PODRBF(pod_rank=reddim, rbf_kernel='thin_plate_spline')
        pod_rbf.fit(params_train, snapshots_train)
        pred_pod_rbf = pod_rbf(params_test).detach().numpy()[ind_test].reshape(-1)
        list_fields = [snap, pred_snap, pred_pod_rbf,
                snap - pred_snap, snap-pred_pod_rbf]
        list_labels = ['Truth', 'Corrected POD-RBF', 'POD',
                'Error Corrected', 'Error POD']
        plot(list_fields, list_labels, 'img/train_results_10kepochs')

        # Plot test correction (approximated and exact)
        coeff_orig = rom.neural_net["interpolation_network"](params_test)
        corr = ann_corr(params_test,coeff_orig
                ).detach().numpy()[ind_test, :].reshape(-1)
        exact_corr = compute_exact_correction(
                pod, pod_big, snapshots_test)[ind_test].detach().numpy().reshape(-1)
        list_fields = [corr, exact_corr, corr - exact_corr]
        list_labels = ['Approximated Correction', 'Exact Correction', 'Error']
        plot(list_fields, list_labels, 'img/correction_test_10kepochs')

    else:
        trainer.train()
        
        pl = Plotter()
        pl.plot_loss(trainer=trainer,label='mean_loss',filename='img/loss_linear',logy=True)
