import numpy as np
import argparse
from sklearn.model_selection import train_test_split
import torch
from pina.solvers import ReducedOrderModelSolver as ROMSolver
from pina.solvers import SupervisedSolver
from pina.model.layers import PODBlock, RBFBlock
from pina.model import FeedForward, DeepONet
from pina.geometry import CartesianDomain
from pina.problem import AbstractProblem, ParametricProblem
from pina.callbacks import MetricTracker
from pina import Condition, LabelTensor, Trainer, Plotter
from smithers.dataset import NavierStokesDataset, LidCavity
import matplotlib.pyplot as plt
from pod_rbf import err, PODRBF
from scaler import Scaler
from linear_corr import  LinearCorrNet 
from deeponet_corr import DeepONetCorrNet
from corrected_rom import CorrectedROM
from utils import plot, compute_exact_correction
from sklearn.preprocessing import MinMaxScaler
import os
from pytorch_lightning.callbacks import Callback
import logging
import time

logging.basicConfig(filename='deep_log.txt',level=logging.INFO,format='%(message)s')
class LogCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        self.tstart = time.time()
    def on_train_end(self,trainer,pl_module):
        self.tend = time.time()
        logging.info(f'reddim={pl_module.neural_net["reduction_network"].rank}, v_num={pl_module.logger.version}, mean_loss={trainer.callback_metrics['mean_loss'].item()}, train_time={self.tend-self.tstart}')


torch.manual_seed(20)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Reduced Order Model Example')
    parser.add_argument('--reddim', type=int, default=3, help='Reduced dimension')
    parser.add_argument('--bigdim', type=int, default=15, help='Bigger dimension')
    parser.add_argument('--field', type=str, default='mag(v)', help='Field to reduce')
    parser.add_argument('--load', help='Directory to save or load file', type=str)
    parser.add_argument('--outsize', help='Size of DeepONet output', type=int, default=10)
    parser.add_argument('--version', help='Version of the model you wish to load', type=int)

    args = parser.parse_args()
    field = args.field
    reddim = args.reddim
    bigdim = args.bigdim
    outsize = args.outsize


    # Import dataset
    data = LidCavity()
    snapshots = data.snapshots[field]
    coords = data.coordinates
    params = data.params
    scaler_params = MinMaxScaler()
    params = scaler_params.fit_transform(params)
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

    # define the exact correction term
    pod = PODBlock(reddim)
    pod.fit(snapshots_train)
    modes = pod.basis

    rbf = RBFBlock(kernel='inverse_multiquadric',epsilon=100.)
    rbf.fit(params_train, pod.reduce(snapshots_train))

    # Plot the modes
    #modes = pod_big.basis.T
    #list_fields = [modes.detach().numpy()[:, i].reshape(-1)
    #        for i in range(2*reddim)]
    #list_labels = [f'Mode {i}' for i in range(2*reddim)]
    #plot(data.triang,list_fields, list_labels,'img/modes_pod')


    # Coordinates as LabelTensor
    coords = torch.tensor(coords,dtype=torch.float32)
    coords = LabelTensor(coords, ['x', 'y'])

    # Compute space-dependent exact correction terms
    #scal_corr = MinMaxScaler()
    exact_correction = compute_exact_correction(pod, snapshots_train)#.detach().numpy()
    #exact_correction = scal_corr.fit_transform(exact_correction)
    #plot(data.triang,[exact_correction[0] , exact_correction[150], exact_correction[-1]],[f'corr{i}' for i in range(3)],'img/corrections_cavity')
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
    #print(torch.min(problem.conditions["correction"].output_points))
    #print(torch.max(problem.conditions["correction"].output_points))

    # Strategies for correction term
    # 1. POD + Linear
    #ann_corr = LinearCorrNet(pod, scaler=Scaler()) #linear model with least squares, not working

    # 2. POD + DeepONet modes@coeffs
    ann_corr = DeepONetCorrNet(pod,
                               coords=coords,
                               interp=RBFBlock(),
                               scaler=None,
                               mode_output_size=outsize
                               )

    if args.load:
        id_ = args.version
        epochs = 5000
        num_batches = 4
        folder = f'img{id_}'

        rom = CorrectedROM.load_from_checkpoint(
                checkpoint_path=os.path.join(args.load,
                f'lightning_logs/version_{id_}/checkpoints/epoch={epochs-1}-step={epochs*num_batches}.ckpt'),
                problem=problem, reduction_network=pod,
                interpolation_network=rbf,
                correction_network=ann_corr)
        # fit the pod and RBF on train data
        #rom.neural_net["reduction_network"].fit(snapshots_train)
        #rom.neural_net["interpolation_network"].fit(params_train,
        #                                        rom.neural_net["reduction_network"].reduce(snapshots_train))
        rom.eval()

        #print(torch.min(rom.problem.conditions["correction"].output_points))
        #print(torch.max(rom.problem.conditions["correction"].output_points))



        # Evaluate the ROM on train and test
        predicted_snaps_train = rom(params_train)
        #exact_corr = compute_exact_correction(pod, snapshots_test)
        #exact_corr = LabelTensor(exact_corr,
        #    [f's{i}' for i in range(Ndof)])
        #rom.neural_net["correction_network"].scaler = Scaler(axis=1)
        #rom.neural_net["correction_network"].scaler.fit(exact_corr) 
        predicted_snaps_test = rom(params_test)
        print(params_test[10])
        np.save('deep_prediction_cavity_10',predicted_snaps_test[10].detach().numpy())
        train_error, train_error_median = err(snapshots_train, predicted_snaps_train)
        test_error, test_error_median = err(snapshots_test, predicted_snaps_test)
        #print('Train error = ', err(snapshots_train, predicted_snaps_train))
        #print('Test error = ', err(snapshots_test, predicted_snaps_test))
        print(train_error, test_error)
        print(train_error_median, test_error_median)
        exit()

        # Plot the modes with the same function
        modes = rom.neural_net["correction_network"].transformed_modes()
        modes = modes.detach().numpy()
        list_fields = [modes[:, i] for i in range(modes.shape[1])]
        list_labels = [f'Mode {i}' for i in range(modes.shape[1])]
        #plot(data.triang,list_fields, list_labels, folder+f'/transformed_modes_{epochs}epochs')

        # Plot the test results (POD-RBF and POD-RBF corrected)
        ind_test = 2
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
        plot(data.triang,list_fields, list_labels, folder+f'/train_results_{epochs}epochs')

        # Plot test correction (approximated and exact)
        coeff_orig = rom.neural_net["interpolation_network"](params_test)
        corr_scaler = rom.neural_net["correction_network"].scaler
        # scale the predicted correction back to original scale
        corr = ann_corr(params_test,coeff_orig)
        #if corr_scaler is not None:
        #    corr = corr_scaler.inverse_transform(corr)
        corr = corr.detach().numpy()[ind_test, :].reshape(-1)
        exact_corr = compute_exact_correction(pod, snapshots_test)
        if corr_scaler is not None:
            exact_corr = corr_scaler.transform(exact_corr)
        exact_corr = exact_corr[ind_test].detach().numpy().reshape(-1)
        list_fields = [corr, exact_corr, corr - exact_corr]
        list_labels = ['Approximated Correction', 'Exact Correction', 'Error']
        plot(data.triang,list_fields, list_labels, folder+f'/correction_test_{epochs}epochs')

    else:
        rom = CorrectedROM(problem=problem,
                    reduction_network=pod,
                    interpolation_network=rbf,
                    correction_network=ann_corr,
                    optimizer=torch.optim.AdamW,
                    optimizer_kwargs={'lr': 7e-3},
                    #scheduler=torch.optim.lr_scheduler.MultiStepLR,
                    #scheduler_kwargs={'gamma': 0.5 ,'milestones': [250, 500, 750, 1000, 1250, 1500, 1750] }
                    )
        # Fit the pod and RBF on train data
        #rom.neural_net["reduction_network"].fit(snapshots_train)
        #rom.neural_net["interpolation_network"].fit(params_train,
        #                                            rom.neural_net["reduction_network"].reduce(snapshots_train))

        # Train the ROM to learn the correction term
        epochs = 5000

        trainer = Trainer(solver=rom, max_epochs=epochs, accelerator='cpu',
                default_root_dir=args.load, callbacks = [MetricTracker(),LogCallback()],
                          batch_size=60)
        trainer.train()
        
        #pl = Plotter()
        #pl.plot_loss(trainer=trainer,label='mean_loss',filename='img/loss',logy=True)
