import torch
from pina import LabelTensor

class Scaler():
    def __init__(self, feature_range=(0., 1.), axis=0):
        self.min_range, self.max_range = feature_range[0], feature_range[1]
        self.axis = axis

    def fit(self, data):
        if isinstance(data, LabelTensor):
            self.labels = data.labels
        else:
            self.labels = []*data.shape[self.axis]
        self.min_correction = torch.min(data, dim=self.axis, keepdim=True).values
        self.max_correction = torch.max(data, dim=self.axis, keepdim=True).values

    def transform(self, data):
        data_std = (data - self.min_correction)/(
                self.max_correction - self.min_correction)
        data_scaled = data_std*(self.max_range - self.min_range) + self.min_range
        data_scaled = self._avoid_nan_in_scaling(data_scaled)
        return LabelTensor(data_scaled, labels=self.labels)

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def _avoid_nan_in_scaling(self, data):
        data = torch.where(torch.isnan(data), torch.zeros_like(data), data)
        return data

    def inverse_transform(self, data):
        data = data*(self.max_correction - self.min_correction) - self.min_range
        data = data/(self.max_range - self.min_range) + self.min_correction
        return LabelTensor(data, labels=self.labels)



