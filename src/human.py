import random

import torch
from datasets.datasets import B2RDataset

data = B2RDataset(root_dir="data/train/070502", transform=None)

for i in range(7):
    id = random.randint(0, len(data)-1)
    sample = data[id]
    label = sample["label"]
    label = label * torch.tensor([data.x_max, data.y_max, data.theta_max])
    print(label)
    # get keyboard input
    input("Press Enter to continue...")