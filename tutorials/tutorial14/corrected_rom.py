import torch
from pina.solvers import SupervisedSolver

class CorrectedROM(SupervisedSolver):
    """
    Non-intrusive ROM using POD as reduction and RBF as approximation,
    and with additional correction/closure term that aim to reintroduce the
    contribution of the neglected modes.
    """
    def __init__(self,
            problem,
            reduction_network,
            interpolation_network,
            correction_network,
            loss=torch.nn.MSELoss(),
            optimizer=torch.optim.Adam,
            optimizer_kwargs={'lr':1e-3},
            scheduler=torch.optim.lr_scheduler.ConstantLR,
            scheduler_kwargs={"factor":1, "total_iters":0},
            ):
        model = torch.nn.ModuleDict({
            "reduction_network": reduction_network,
            "interpolation_network": interpolation_network,
            "correction_network": correction_network,
            })
        super().__init__(
                model=model,
                problem=problem,
                loss=loss,
                optimizer=optimizer,
                optimizer_kwargs=optimizer_kwargs,
                scheduler=scheduler,
                scheduler_kwargs=scheduler_kwargs,
                )

        # if reduction and interpolation has fit method, fit them
        #if hasattr(reduction_network, "fit"):
        #    reduction_network.fit(problem.conditions["data"].output_points)
        #if hasattr(interpolation_network, "fit"):
        #    interpolation_network.fit(problem.conditions["data"].input_points,
        #            reduction_network.reduce(problem.conditions["data"].output_points))
        if hasattr(correction_network, "fit"):
            correction_network.fit(problem.conditions["correction"].input_points,
                   problem.conditions["correction"].output_points)

        self.modes = reduction_network.basis


    def forward(self, input_params):
        '''
        Compute the ROM solution, by summing the POD term and the correction term.
        '''
        reduction_network = self.neural_net["reduction_network"]
        interpolation_network = self.neural_net["interpolation_network"]
        correction_network = self.neural_net["correction_network"]

        # the coefficients wrt the original POD basis
        coeff = interpolation_network(input_params)

        pod_term = reduction_network.expand(coeff)
        correction_term = correction_network(input_params,coeff)

        return pod_term + correction_term

    def loss_data(self, input_pts, output_pts):
        interpolation_network = self.neural_net["interpolation_network"]
        correction_network = self.neural_net["correction_network"]

        coeff_orig = interpolation_network(input_pts)

        approx_correction = correction_network(input_pts, coeff_orig)

        #exact_correction = self.problem.conditions["correction"].output_points
        exact_correction = output_pts
        
        loss_correction = self.loss(approx_correction, exact_correction)

        #modes_corr = correction_network.transformed_modes
        #loss_orthog = torch.norm(torch.matmul(modes_corr.T,modes_corr)-torch.eye(correction_network.reduced_dim))
        
        return loss_correction #+ loss_orthog


    @property
    def neural_net(self):
        return self._neural_net.torchmodel


