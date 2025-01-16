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
            #problem.conditions["correction"].output_points = 
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
        if correction_network.scaler is not None:
            correction_term = correction_network.scaler.inverse_transform(correction_term)

        return pod_term + correction_term

    def _forward_no_interp(self, input_params, snaps):
        '''
        Compute the ROM solution, by summing the POD term and the correction term.
        '''
        reduction_network = self.neural_net["reduction_network"]
        correction_network = self.neural_net["correction_network"]

        # project snapshots
        coeff = reduction_network.reduce(snaps)

        # POD term is just expand(reduce(snapshots)), we do not do interpolation
        pod_term = reduction_network.expand(coeff)
        correction_term = correction_network(input_params,coeff)
        if correction_network.scaler is not None:
            correction_term = correction_network.scaler.inverse_transform(correction_term)

        return pod_term + correction_term

    def loss_data(self, input_pts, output_pts):
        interpolation_network = self.neural_net["interpolation_network"]
        correction_network = self.neural_net["correction_network"]

        coeff_orig = interpolation_network(input_pts)

        approx_correction = correction_network(input_pts, coeff_orig)

        exact_correction = output_pts
        
        loss_correction = self.loss(approx_correction, exact_correction)

        # orthogonal components loss
        V = correction_network.modes
        Vbar = correction_network.C(input_pts)
        # Vbar = correction_network.operator.T
        loss_orthog = torch.norm(V.T@Vbar)

        # importance of correction over orthonormalisation
        beta = 0.001
        self.log("loss_orthon", float(loss_orthog), prog_bar=True, logger=True)
        self.log("loss_corr", float(loss_correction), prog_bar=True, logger=True)

        # norm1 = torch.linalg.norm(approx_correction-exact_correction,dim=-1).mean()
        # norm2 = torch.linalg.norm(exact_correction,dim=-1).mean()
        # other_term = torch.nn.functional.relu(norm1 - norm2)

        #return beta * loss_correction #+ (1-beta) * loss_orthog
        return  loss_correction + beta * loss_orthog #+ other_term


    @property
    def neural_net(self):
        return self._neural_net.torchmodel


