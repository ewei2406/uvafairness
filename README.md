# UVA Fairness library

UVA Fairness library implementation

## Installation
```
pip install -i https://test.pypi.org/simple/ uva-fairness-ewei2406==0.0.1
```

## Usage
Base GCN (using Torch)
```
from uva_fairness import models
import torch

simple_model = models.BaseGCN(input_dim=64, output_dim=10)
X = torch.rand(1, 8, 8, device=device)
logits = simple_model(X)
```

## Development
Clone the repository:
```
git clone https://github.com/ewei2406/uvafairness.git
```
Modify contents in ```src/uva_fairness```

## Testing
Requires ```pytest >= 6.2.5```
```
pytest uva_fairness
```

![137540723-827a7f2e-7cfb-48a2-8955-bf3c69b7ba91](https://user-images.githubusercontent.com/34495421/138992587-74391783-6643-4201-a0c3-8581e3ec55ce.png)
