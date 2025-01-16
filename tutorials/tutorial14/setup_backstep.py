import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from smithers.dataset import NavierStokesDataset
from pina.model.layers import PODBlock, RBFBlock
from pina.problem import ParametricProblem
from pina.geometry import CartesianDomain
from pina import Condition, LabelTensor
from utils import compute_exact_correction
import numpy as np

class BackstepProblem():
    def __init__(self,field,reddim,subset=None,train_size=100,test_size=100,device='cpu'):
        self.field = field
        self.reddim = reddim
        self.train_size = train_size
        self.test_size = test_size
        self.device = device
        self._load_data()
        self._train_test_split()
        if self.device == 'gpu':
            self.gpu() 
        self._fit_pod()
        if self.device == 'gpu':
            self.gpu()
        self._fit_rbf()
        self._compute_corrections()
        if subset is not None:
            self.subset_size = subset
            self._extract_subset()
        self._define_problem()
        if self.device == 'gpu':
            self.gpu()

    def _load_data(self):
        self.data = NavierStokesDataset()

        self.snapshots = self.data.snapshots[self.field]

        coords = self.data.pts_coordinates
        # Coordinates as LabelTensor
        coords = torch.tensor(coords.T,dtype=torch.float32)
        self.coords = LabelTensor(coords, ['x', 'y'])

        params = self.data.params
        self.scaler_params = MinMaxScaler()
        self.params = self.scaler_params.fit_transform(params)

        self.Ndof = self.snapshots.shape[1]
        self.Nparams = self.params.shape[1]

    def _train_test_split(self,seed=42):
        # Divide dataset into training and testing
        params_train, params_test, snapshots_train, snapshots_test = train_test_split(
                self.params, self.snapshots, test_size=self.test_size,train_size=self.train_size, shuffle=True, random_state=seed)

        # From numpy to LabelTensor
        self.params_train = LabelTensor(torch.tensor(params_train, dtype=torch.float32),
                labels=['mu'])
        self.params_test = LabelTensor(torch.tensor(params_test, dtype=torch.float32),
                labels=['mu'])
        self.snapshots_train = LabelTensor(torch.tensor(snapshots_train, dtype=torch.float32),
                labels=[f's{i}' for i in range(snapshots_train.shape[1])])
        self.snapshots_test = LabelTensor(torch.tensor(snapshots_test, dtype=torch.float32),
                labels=[f's{i}' for i in range(snapshots_test.shape[1])])

    def _fit_pod(self):
        self.pod = PODBlock(self.reddim)
        self.pod.fit(self.snapshots_train)
        self.modes = self.pod.basis.T
        self.modes = LabelTensor(self.modes, [f'{i}' for i in range(self.reddim)])

    def _fit_rbf(self):
        self.rbf = RBFBlock(kernel='linear')
        self.rbf.fit(self.params_train, self.pod.reduce(self.snapshots_train))

    def _compute_corrections(self):
        # Compute space-dependent exact correction terms
        exact_correction = compute_exact_correction(self.pod, self.snapshots_train)

        self.exact_correction = LabelTensor(exact_correction,
                [f's{i}' for i in range(self.Ndof)])

    def _define_problem(self):
        # Define ROM problem with only data (parameters and snapshots)
        class SnapshotProblem(ParametricProblem):
            input_variables = ['mu']
            output_variables = self.snapshots_train.labels
            #output_variables += exact_correction.labels
            parameter_domain = CartesianDomain({'mu':[0, 100]})
            conditions = {'correction': Condition(
                                input_points=self.params_train,
                                output_points=self.exact_correction)
                          }

        self.problem = SnapshotProblem()

    def _extract_subset(self):
        N = int(self.subset_size*self.Ndof)
        values_ = self.exact_correction#.detach().numpy()
        # avg = np.zeros_like(values_[0])
        # for i in range(self.snapshots_train.shape[0]):
        #     values = values_[i]
        #
        #     # Compute gradients
        #     node_gradients = compute_gradients_matplotlib(self.data.triang, values)
        #     avg += np.linalg.norm(node_gradients,axis=-1)
        # avg /= (i+1)
        avg = torch.mean(torch.abs(values_),dim=0)
       # avg = torch.from_numpy(avg)
        p = torch.exp(-torch.mean(avg)/(avg+1e-6))
        p /= p.sum()

        indices = p.multinomial(N,replacement=False)

        #indices = torch.randperm(self.Ndof)[:N]
        self.indices = indices.tensor
        self.Ndof = N
        self.snapshots = self.snapshots[:,indices.tensor.cpu()]

        self.snapshots_train = self.snapshots_train.tensor[:,indices]
        self.snapshots_train = LabelTensor(self.snapshots_train, [f's{i}' for i in range(self.snapshots_train.shape[1])])

        self.snapshots_test = self.snapshots_test.tensor[:,indices]
        self.snapshots_test= LabelTensor(self.snapshots_test, [f's{i}' for i in range(self.snapshots_test.shape[1])])

        self.coords = self.coords[indices,:]
        self.modes = self.modes[indices,:]
        self.modes = LabelTensor(self.modes, [f'{i}' for i in range(self.reddim)])

        self.exact_correction = self.exact_correction.tensor[:,indices]
        self.exact_correction = LabelTensor(self.exact_correction, [f's{i}' for i in range(self.exact_correction.shape[1])])
        # print(self.snapshots.shape)
        # print(self.snapshots_train.shape)
        # print(self.snapshots_test.shape)
        # print(self.coords.shape)
        # print(self.modes.shape)
        # print(self.exact_correction.shape)

    def gpu(self):
        for arg in dir(self):
            if 'cuda' in dir(getattr(self, arg)):
                # print(f'Moving {arg}')
                setattr(self,arg,getattr(self,arg).cuda())
