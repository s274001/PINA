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
from linear_corr import LinearCorrNet
from deeponet_corr import DeepONetCorrNet
from quadratic_corr import QuadraticCorrNet
from corrected_rom import CorrectedROM
from tutorials.tutorial14.setup_backstep import BackstepProblem
from utils import plot, compute_exact_correction
from sklearn.preprocessing import MinMaxScaler
import os
from pytorch_lightning.callbacks import Callback
from scipy.stats import gaussian_kde


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Reduced Order Model Example')
    parser.add_argument('--reddim', type=int, default=3, help='Reduced dimension')
    parser.add_argument('--train', type=int, default=10, help='Train set size')
    parser.add_argument('--field', type=str, default='mag(v)', help='Field to reduce')
    parser.add_argument('--load', help='Directory to save or load file', type=str)
    parser.add_argument('--version', help='Model version', type=int)

    args = parser.parse_args()
    field = args.field
    reddim = args.reddim



    backstep = BackstepProblem('mag(v)',reddim=reddim,subset=None,train_size=args.train, device='cpu')
    problem = backstep.problem

    # 2. POD + DeepONet modes@coeffs
    ann_corr = QuadraticCorrNet(pod=backstep.pod,
                                coeffs=backstep.pod.reduce(backstep.snapshots_train),
                                interp=backstep.rbf,
                                )

    if args.load:
        id_ = args.version
        epochs = 2
        num_batches = 1

        rom = CorrectedROM.load_from_checkpoint(
                checkpoint_path=os.path.join(args.load,
                f'lightning_logs/version_{id_}/checkpoints/epoch={epochs-1}-step={epochs*num_batches}.ckpt'),
                problem=problem, reduction_network=backstep.pod,
                interpolation_network=backstep.rbf,
                correction_network=ann_corr)
        rom.eval()

        # Plot the operator columns
        # modes = rom.neural_net["correction_network"].operator.T
        # modes = modes.detach().numpy()
        # list_fields = [modes[:, i] for i in range(modes.shape[1])]
        # list_labels = [f'Mode {i}' for i in range(modes.shape[1])]
        # vmin = min([field.min() for field in list_fields])
        # vmax = max([field.max() for field in list_fields])
        # plot(data.triang,list_fields, list_labels, 
        #      filename='img/quadratic_operator_columns',
        #      vmin = vmin,
        #      vmax = vmax)
        # # fig = plt.gcf()
        # # poss = []
        # # for i,ax in enumerate(fig.axes):
        # #         pos = ax.get_position()
        # #         ax.set_position([pos.x0, pos.y0 + 0.4, pos.width, pos.height/2])
        # #         poss.append(pos)
        # fig,axs = plt.subplots(1,6)
        # axs = axs.ravel()
        # x = np.linspace(-1.5,1.5,200)
        # for i,field in enumerate(list_fields):
        #         ax = axs[i]
        #         # ax = fig.add_subplot(1,6,1+i)
        #         # pos = poss[i]
        #         # ax = fig.add_axes([pos.x0, pos.y0, pos.width , 0.3])
        #         dens = gaussian_kde(field)
        #         ax.plot(x,dens(x),'k')
        #         ax.fill_between(x,np.zeros_like(x),dens(x),color='grey',alpha=0.6)
        #         ax.set_title(f'Mode {i}')
        #         ax.set_xlabel('$|v|$')
        # axs[0].set_ylabel('Density')
        # fig.tight_layout()
        # plt.show()

        params_train = backstep.params_train
        params_test = backstep.params_test
        snapshots_train = backstep.snapshots_train
        snapshots_test = backstep.snapshots_test
        # Evaluate the ROM on train and test
        predicted_snaps_train = rom(params_train)
        predicted_snaps_test = rom(params_test)
        train_error = err(snapshots_train, predicted_snaps_train)
        test_error = err(snapshots_test, predicted_snaps_test)
        # train_error = (torch.linalg.norm(predicted_snaps_train-snapshots_train,dim=-1)/torch.linalg.norm(snapshots_train,dim=-1)).tensor.cpu().detach().numpy()
        # test_error = (torch.linalg.norm(predicted_snaps_test-snapshots_test,dim=-1)/torch.linalg.norm(snapshots_test,dim=-1)).tensor.cpu().detach().numpy()
        #
        # np.save('./full_data/backstep/ls_train_error',train_error)
        # np.save('./full_data/backstep/ls_test_error',test_error)
        #print('Train error = ', err(snapshots_train, predicted_snaps_train))
        #print('Test error = ', err(snapshots_test, predicted_snaps_test))
        print(train_error,test_error)
        exit()

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
        plot(data.triang,list_fields, list_labels, filename='img/train_results_quadratic')

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
        vmin = min([field.min() for field in list_fields])
        vmax = max([field.max() for field in list_fields])
        plot(data.triang,list_fields, list_labels, 
             filename='img/correction_test_quadratic',
             vmin = vmin,
             vmax = vmax)

    else:
        rom = CorrectedROM(problem=problem,
                    reduction_network=backstep.pod,
                    interpolation_network=backstep.rbf,
                    correction_network=ann_corr,
                    optimizer=torch.optim.Adam,
                    optimizer_kwargs={'lr': 1e-2},
                    )

        # Train the ROM to learn the correction term
        epochs = 2

        trainer = Trainer(solver=rom, max_epochs=epochs, accelerator='cpu',
                default_root_dir=args.load)#, callbacks = [MetricTracker()],
                #,batch_size=80)
        trainer.train()
        #pl = Plotter()
        #pl.plot_loss(trainer=trainer,label='mean_loss',filename='img/loss',logy=True)
