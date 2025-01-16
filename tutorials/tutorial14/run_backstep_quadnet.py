from matplotlib.tri import Triangulation
import numpy as np
import argparse
import torch
from pina.model.layers import PODBlock, RBFBlock
from pina.callbacks import MetricTracker
from pina import Condition, LabelTensor, Trainer, Plotter
from pina.loss import LpLoss
import matplotlib.pyplot as plt
from pod_rbf import err, PODRBF
from corrected_rom import CorrectedROM
from utils import plot, compute_exact_correction
from quadnet import QuadNet
import os
import sys
from pytorch_lightning.callbacks import Callback, EarlyStopping
import logging
import time
from setup_backstep import BackstepProblem
from scipy.stats import gaussian_kde

logging.basicConfig(filename='backstep_test_log.txt',level=logging.INFO,format='%(message)s')
class LogCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        self.tstart = time.time()
    def on_train_end(self,trainer,pl_module):
        self.tend = time.time()
        logging.info(f'reddim={pl_module.neural_net["reduction_network"].rank}, v_num={pl_module.logger.version}, loss_corr={trainer.callback_metrics['loss_corr'].item()}, train_time={self.tend-self.tstart}, epochs={trainer.current_epoch}')

#torch.manual_seed(20)

def main(arguments=None):
    # Parse arguments
    parser = argparse.ArgumentParser(description='Reduced Order Model Example')
    parser.add_argument('--reddim', type=int, default=3, help='Reduced dimension')
    parser.add_argument('--train', type=int, default=10, help='Train set size')
    parser.add_argument('--field', type=str, default='mag(v)', help='Field to reduce')
    parser.add_argument('--load', help='Directory to save or load file', type=str)
    parser.add_argument('--version', help='Model version', type=int)
    parser.add_argument('--epochs', help='Number of training epochs', type=int,default=1000)

    args = parser.parse_args(arguments)
    field = args.field
    reddim = args.reddim
    epochs = args.epochs

    backstep = BackstepProblem(field,reddim,subset=None, train_size=args.train,device='gpu')
    # logging.info(f'pts={backstep.subset_size}')
    # backstep.gpu()
    # print(backstep.indices)
    # if not args.load:
    #     backstep_ref = BackstepProblem(field,reddim)
    #     fig,ax = plt.subplots(figsize=(15,6))
    #     ax.scatter(backstep.coords['x'].cpu().detach().numpy(),backstep.coords['y'].cpu().detach().numpy(), c='k')
    #     ax.triplot(backstep_ref.data.triang,c='grey',zorder=-1)   
    #     fig.savefig('img/quadnet_training_points')
    #     # plt.show()
    #     del fig,ax,backstep_ref
    #     plt.gcf().clear()


    problem = backstep.problem



    # 2. POD + DeepONet modes@coeffs
    corr_net = QuadNet(backstep.modes, backstep.coords)
    # coef = backstep.pod.reduce(backstep.snapshots_train)

    if not args.load:
        match reddim:
            case 3:
                if args.train < 100:
                    lr = 3e-4
                else:
                    lr = 1e-3
            case 5:
                if args.train < 150:
                    lr = 1e-4
                else:
                    lr = 6e-4
            case 7:
                if args.train < 30:
                    lr = 7e-5
                elif 30 <= args.train <= 200:
                    lr = 1e-4
                else:
                    lr = 5e-4
            case 9:
                if args.train < 30:
                    lr = 1e-5
                elif 30 <= args.train < 60:
                    lr = 5e-5
                else:
                    lr = 1e-4
            case _:
                lr = 1e-4

        lr = 1e-2
        rom = CorrectedROM(problem = problem,
                           reduction_network = backstep.pod,
                           interpolation_network = backstep.rbf,
                           correction_network = corr_net,
                           optimizer = torch.optim.Adam,
                           optimizer_kwargs = {'lr':lr,'weight_decay':1e-3},
                           loss = LpLoss(relative=True),
                           scheduler=torch.optim.lr_scheduler.MultiStepLR,
                           scheduler_kwargs={'milestones':[1000,2000,3000,4000],
                                             'gamma':0.9}
                           )

        # epochs = 20000

        if args.train > 10:
            num_batches = 4
        else:
            num_batches = 1

        early = EarlyStopping(monitor='loss_corr',patience=5000,stopping_threshold=1e-2, check_on_train_epoch_end=True)
        trainer = Trainer(solver=rom, 
                          max_epochs=epochs, 
                          accelerator='gpu',
                          default_root_dir=args.load, 
                          callbacks = [MetricTracker(),LogCallback(),early],
                          batch_size=args.train//num_batches,
                          )
        trainer.train()
        
        pl = Plotter()
        pl.plot_loss(trainer=trainer,label='loss_corr',logy=True,filename=f'quadnet/backstep/loss_{reddim}_{args.train}')
        #plt.show()
        params_test = backstep.params_test
        snapshots_test = backstep.snapshots_test
        # print(params_test.device)
        # print(snapshots_test.device)
        rom.cuda()
        rom.eval()
        predicted_snaps_test = rom(params_test)
        test_error = err(snapshots_test, predicted_snaps_test)
        logging.info(f'r={reddim}, train_size={args.train}, lr={lr}, test_mean={test_error[0]}\n------------------')

    else:
        id_ = args.version
        # epochs = 20000
        num_batches = 5

        data = backstep.data
        params_train = backstep.params_train
        params_test = backstep.params_test
        snapshots_train = backstep.snapshots_train
        snapshots_test = backstep.snapshots_test
        pod = backstep.pod

        rom = CorrectedROM.load_from_checkpoint(
                checkpoint_path=os.path.join(args.load,
                f'lightning_logs/version_{id_}/checkpoints/epoch={epochs-1}-step={epochs*num_batches}.ckpt'),
                problem = problem,
               reduction_network = backstep.pod,
               interpolation_network = backstep.rbf,
               correction_network = corr_net)
        # fit the pod and RBF on train data
        # rom.neural_net["reduction_network"].fit(snapshots_train)
        # rom.neural_net["interpolation_network"].fit(params_train,
        #                                        rom.neural_net["reduction_network"].reduce(snapshots_train))
        rom.eval()
        #print(torch.min(rom.problem.conditions["correction"].output_points))
        #print(torch.max(rom.problem.conditions["correction"].output_points))
        print(rom.modes@corr_net.C())

        c = corr_net.C().tensor
        list_labels = [f'{i}' for i in range(0,reddim*(reddim+1)//2)]
        list_fields = [c[:,i].detach().numpy().reshape(-1) for i in range(0,reddim*(reddim+1)//2)]
        # c = torch.reshape(corr_net.C().tensor,(-1,3,3))
        # list_labels = [f'{i},{j}' for i in range(1,reddim+1) for j in range(i,reddim+1)]
        # list_fields = [c[:,i,j].detach().numpy().reshape(-1) for i in range(0,reddim) for j in range(i,reddim)]
        # plot(data.triang,list_fields,list_labels,filename='img/c_entries_net_u')
        # list_labels = [f'{j},{i}' for i in range(1,reddim+1) for j in range(i,reddim+1)]
        # list_fields = [c[:,j,i].detach().numpy().reshape(-1) for i in range(0,reddim) for j in range(i,reddim)]
        vmin = min([field.min() for field in list_fields])
        vmax = max([field.max() for field in list_fields])
        plot(data.triang,list_fields,list_labels,
                 filename='img/c_entries_net',
                 vmin = vmin,
                 vmax = vmax)
        fig,axs = plt.subplots(1,6)
        axs = axs.ravel()
        x = np.linspace(-1.5,1.5,200)
        for i,field in enumerate(list_fields):
                ax = axs[i]
                # ax = fig.add_subplot(1,6,1+i)
                # pos = poss[i]
                # ax = fig.add_axes([pos.x0, pos.y0, pos.width , 0.3])
                dens = gaussian_kde(field)
                ax.plot(x,dens(x),'k')
                ax.fill_between(x,np.zeros_like(x),dens(x),color='grey',alpha=0.6)
                ax.set_title(f'Column {i}')
                ax.set_xlabel('$|v|$')
        axs[0].set_ylabel('Density')
        fig.tight_layout()
        plt.show()


        # Plot the modes with the same function
        #modes = rom.neural_net["correction_network"].modes
        #modes = modes.detach().numpy()
        #list_fields = [modes[:, i] for i in range(modes.shape[1])]
        #list_labels = [f'Mode {i}' for i in range(modes.shape[1])]
        ##plot(data.triang,list_fields, list_labels, filename='img/transformed_modes_10kepochs')
        #for i in range(reddim):
        #    list_fields = [modes[:, i].reshape(-1)]
        #    #list_fields = [modes.detach().numpy()[:, i].reshape(-1)
        #    #        for i in range(reddim)]
        #    list_labels = [f'Corrected mode {i+1}']# for i in range(reddim)]
        #    plot(data.triang,list_fields, list_labels,filename=f'img/modes{i+1}_deep_sparse')
        ##exit()


        # Evaluate the ROM on train and test
        predicted_snaps_train = rom(params_train)
        predicted_snaps_test = rom(params_test)
        print(params_test[10])
        np.save('deep_prediction_10',predicted_snaps_test[10].detach().numpy())
        train_error = err(snapshots_train, predicted_snaps_train)
        test_error = err(snapshots_test, predicted_snaps_test)
        #print('Train error = ', err(snapshots_train, predicted_snaps_train))
        #print('Test error = ', err(snapshots_test, predicted_snaps_test))
        print(train_error,test_error)
        #exit()

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
        plot(data.triang,list_fields, list_labels, filename='img/quadnet_compare')

        # Plot test correction (approximated and exact)
        coeff_orig = rom.neural_net["interpolation_network"](params_test)
        corr_scaler = rom.neural_net["correction_network"].scaler
        # scale the predicted correction back to original scale
        corr = corr_net(params_test,coeff_orig)
        #if corr_scaler is not None:
        #    corr = corr_scaler.inverse_transform(corr)
        corr = corr.detach().numpy()[ind_test, :].reshape(-1)
        exact_corr = compute_exact_correction(pod, snapshots_test)
        if corr_scaler is not None:
            exact_corr = corr_scaler.transform(exact_corr)
        exact_corr = exact_corr[ind_test].detach().numpy().reshape(-1)
        list_fields = [corr, exact_corr, corr - exact_corr]
        list_labels = ['Approximated Correction', 'Exact Correction', 'Error']
        plot(data.triang,list_fields, list_labels, filename='img/quadnet_correction')

if __name__ == "__main__":
    dimensions = [7,9]
    sizes = [10,20,30,40,50,100,150,200]

    # for r in dimensions:
    #     for size in sizes:
    #         args = ['--epochs', '50000', '--reddim', f'{r}', '--train', f'{size}']
    #
    #         main(args)
    main(sys.argv[1:])
