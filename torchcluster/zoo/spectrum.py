import torch
import numpy as np
from .base import Cluster
from .k_means import KMeans
from ..helper.distance import setwise_distance

class SpectrumClustering(Cluster):
    """Spectrum clustering algorithm.
    """

    def __init__(self, n_clusters=None, cluster=None, threshold=2, k=2, eps=1e-05):
        """Spectrum clustering factory's config.

        Kwargs:
           n_clusters (int) - how many clusters in result. You do not need it if giving a cluster

           cluster (Cluster) - clustering method after spectrum transformation

           threshold (int) - threshold of dropping out an edge 

           k (int) - the number of selected feature

           eps (float) – a value added to the denominator for numerical stability.

        """

        super(SpectrumClustering, self).__init__()
        if cluster is None:
            cluster = KMeans(n_clusters)
        self.cluster = cluster
        self.threshold = threshold
        self.k = k
        self.eps = eps

    def __call__(self, x):
        """Clustering.

        Args:
           x (Tensor) - Data points of number n by feature dim m.
        
        """
        adj = (setwise_distance(x) < self.threshold).float()
        diag = adj.sum(1).diag()
        laplican = diag - adj

        inv_diag = torch.diag(torch.pow(torch.diag(diag), -0.5))
        sym_laplican = inv_diag.mm(laplican).mm(inv_diag)

        e, v = sym_laplican.eig(eigenvectors=True)
        _, idx = torch.topk(e[:, 0], self.k, largest=False)
        select_v = v[:, idx]
        norm_v = select_v.div(select_v.norm(p=2, dim=1, keepdim=True).expand_as(select_v) + self.eps)
        
        return self.cluster(norm_v)