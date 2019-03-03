import torch
from .base import Dataset

class SimpleDataset(Dataset):
    """We use this as a simple dataset to test clustering algorithm.
    """
    def __init__(self, n_clusters, device='cpu', feature=10, sigma=10):
        """Simple dataset factory's config.

        Args:
           n_clusters (int) - How many clusters in result.

        Kwargs:
           device (string) - Device of tensors.

           feature (int) - The dim of each data point.

           sigma (float) - Factor of clustering difficulty, the bigger the easier.

        """
        super(SimpleDataset, self).__init__()
        self.n_clusters = n_clusters
        self.device = device
        self.feature = feature
        self.sigma = sigma
    
    def __call__(self, n):
        """Generate dataset.

        Args:
           n (int) - the number of data point.
        
        """
        idx_n = n // self.n_clusters
        return torch.cat([(torch.randn(idx_n, self.feature) + idx * self.sigma).to(self.device) for idx in range(self.n_clusters)])