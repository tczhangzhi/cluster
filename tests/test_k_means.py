import pytest
import torch
from torchcluster.dataset.simple import SimpleDataset
from torchcluster.zoo.spectrum import KMeans

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_k_means():
    X, y = SimpleDataset(2, feature=2, sigma=2, device=device)(100)
    cluster = KMeans(2)

    result, _ = cluster(X)
    print('acc:', (result == y).sum().item() / result.size(0))
    assert(True)