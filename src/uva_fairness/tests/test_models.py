from uva_fairness import models

def test_BaseGCN():
    result = models.base_GCN()
    assert result == "This is the base GCN"
