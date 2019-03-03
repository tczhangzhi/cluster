torchcluster
============

`Documentation <https://torchcluster.readthedocs.io/en/latest/>`__ \|

Torchcluster is a python package for cluster analysis. The speed of the
clustering algorithm has been effectively improved with the Pytorch
backend. We are also working on test datasets and visualization tools.
Related work is coming in the next release.

System requirements
-------------------

torchcluster should work on

-  all Linux distributions no earlier than Ubuntu 16.04
-  macOS X
-  Windows 10

torchcluster also requires Python 3.5 or later. Python 2 support is
coming.

Right now, torchcluster works on `PyTorch <https://pytorch.org/>`__
0.4.1.

Installation
------------

Using pip
~~~~~~~~~

::

    pip install torchcluster

Using anaconda
~~~~~~~~~~~~~~

::

    conda install -c tczhangzhi torchcluster

How torchcluster looks like
---------------------------

Define a dataset generator and generate a dataset:

::

    from torchcluster.dataset.simple import SimpleDataset

    dataset_factory = SimpleDataset(2, feature=2, sigma=2, device=device)
    dataset = dataset_factory(100)

Configuring a clustering algorithm and get your result:

::

    from torchcluster.zoo.spectrum import SpectrumClustering

    cluster = SpectrumClustering(2)
    result, _ = cluster(dataset)

You can also cluster your own data sets. The dataset should be a tensor
of n by m, where n is the number of data points in the dataset and m is
the dimension of each data point:

::

    dataset = torch.cat([torch.randn(500,2) + torch.Tensor([-2,-3]), torch.randn(500,2) + torch.Tensor([2,1])])

Use spectral clustering to get the following results:

::

    tensor([0, 0, ..., 1, 1])

License
-------

`MIT <http://opensource.org/licenses/MIT>`__

Copyright (c) 2019-present, Zhang Zhi