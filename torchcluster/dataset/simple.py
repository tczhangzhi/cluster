import torch
from .base import Dataset

class SimpleDataset(Dataset):
    r"""Simple dataset class
        Args:
            n_clusters (int) - how many clusters in result.
            device (string) - device of tensors.
            feature (int) - the dim of each data point.
            sigma (float) - factor of clustering difficulty, the bigger the easier.
    """
    def __init__(self, n_clusters, device='cpu', feature=10, sigma=10):
        super(SimpleDataset, self).__init__()
        self.n_clusters = n_clusters
        self.device = device
        self.feature = feature
        self.sigma = sigma
    
    def __call__(self, n):
        """
        Args:
            n (int): the number of data point.
        Returns:
            tensor: a matrix of n by m, where n is the number of data point and m is the dim of each data point.
        """
        idx_n = n // self.n_clusters
        return torch.cat([(torch.randn(idx_n, self.feature) + idx * self.sigma).to(self.device) for idx in range(self.n_clusters)])