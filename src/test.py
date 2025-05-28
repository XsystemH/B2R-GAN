import torch
from datasets.datasets import B2RDataset
from models.discriminator import Discriminator
from models.generator_unet import UNetGenerator
from trainers.gan_trainer import GANTrainer
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image

from utils.color_mask import yellow_mask

data = B2RDataset(root_dir="data/train/070502", transform=None)
sample = data[9]
print(sample["label"])
print(sample["image"].shape)

imgs = sample["image"]
null_image = sample["null_image"]
mask = yellow_mask(imgs, r_min=0.75, r_max=1.25, b_max=0.25)
masked = imgs * mask
masked = to_pil_image(masked)
plt.imshow(masked)
plt.axis('off')
plt.show()

added = null_image * 0.8 + imgs * mask
added = to_pil_image(added)
plt.imshow(added)
plt.axis('off')
plt.show()