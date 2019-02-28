import pytest
import torch
from torchcluster.dataset.simple import SimpleDataset
from torchcluster.zoo.spectrum import SpectrumClustering

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_spectrum():
    dataset = SimpleDataset(2, feature=2, sigma=2, device=device)(100)
    cluster = SpectrumClustering(2)

    result, _ = cluster(dataset)
    print(result)
    assert(True)