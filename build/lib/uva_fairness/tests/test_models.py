from uva_fairness import models
from torch import nn


def test_BaseGCN():
    result = models.base_GCN()
    assert isinstance(result, nn.Module)
