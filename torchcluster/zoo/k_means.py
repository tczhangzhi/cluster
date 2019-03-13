import torch
import numpy as np
from .base import Cluster
from ..helper.distance import setwise_distance

class KMeans(Cluster):
    """K-Means algorithm
    """
    def __init__(self, n_clusters, tol=1e-4):
        """Spectrum clustering factory's config.

        Args:
           n_clusters (int) - How many clusters in result.

        Kwargs:
           tol (float) - stop to update when shift is smaller than tol

        """
        super(KMeans, self).__init__()
        self.n_clusters = n_clusters
        self.tol = tol

    def __call__(self, x):
        """Clustering.

        Args:
           x (Tensor) - Data points of number n by feature dim m.
        
        """
        idx = np.random.choice(len(x), self.n_clusters)
        state = x[idx]

        while True:
            pre_state = state.clone()
            dis = setwise_distance(x, state).squeeze()
            result = torch.argmin(dis, dim=1)
            
            for i in range(self.n_clusters):
                idx = torch.nonzero(result == i).squeeze()
                items = torch.index_select(x, 0, idx)
                if items.size(0):
                    state[i] = items.mean(dim=0)
                else:
                    state[i] = pre_state[i].clone()
                
            shift = torch.pairwise_distance(pre_state, state)
            total = torch.pow(torch.sum(shift), 2.0)

            if total < self.tol:
                return result, state