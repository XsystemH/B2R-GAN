import os, sys
import time
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "src")))

import argparse
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.transforms.functional import to_pil_image
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from datasets.datasets import B2RDataset
from models.discriminator import Discriminator
from models.generator_unet import UNetGenerator
from trainers.gan_trainer import GANTrainer

parser = argparse.ArgumentParser(description="Train a GAN model")
parser.add_argument("--data_dir", type=str, default="data/train/070502", help="Directory containing training data")
parser.add_argument("--batch_size", type=int, default=14, help="Batch size for training")
parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs for training")
parser.add_argument("--G_lr", type=float, default=0.002, help="Learning rate for the generator's optimizer")
parser.add_argument("--D_lr", type=float, default=0.002, help="Learning rate for the discriminator's optimizer")
parser.add_argument("--num_filters", type=int, default=16, help="Number of filters in the generator and discriminator")
parser.add_argument("--cold_start", type=int, default=1, help="Whether to cold start the generator and discriminator")

args = parser.parse_args()

data = B2RDataset(root_dir=args.data_dir, transform=None)

# randomly devide to train and test
torch.manual_seed(42)
test_indices = torch.randperm(len(data))[:int(len(data) * 0.1)]
test_dataset = Subset(data, test_indices)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

train_indices = [i for i in range(len(data)) if i not in test_indices]
train_dataset = Subset(data, train_indices)
data_loader = DataLoader(train_dataset, batch_size=14, shuffle=True)

gen = UNetGenerator(in_channels=3, out_channels=3, num_filters=args.num_filters)
dis = Discriminator(in_channels=3, out_channels=3, num_filters=args.num_filters)
trainer = GANTrainer(generator=gen, discriminator=dis, dataloader=data_loader, config={"G_lr": args.G_lr, "D_lr": args.D_lr, "showlog": 100})

print(args.cold_start)

# Cold start the generator and discriminator
if args.cold_start == 1:
    print("Cold starting the generator...")
    trainer.train_G(1000)
    print("Cold starting the discriminator...")
    trainer.train_D(1000)
    
    colddir = "outputs/cold_start"
    if not os.path.exists(colddir):
        os.makedirs(colddir)
    print("Saving cold start models...")
    trainer.save_model(colddir)
else:
    print("Loading pre-trained models...")
    trainer.load_model("outputs/cold_start")

# Train together
print("Training together...")
data_loader = DataLoader(data, batch_size=args.batch_size, shuffle=True)
trainer = GANTrainer(generator=gen, discriminator=dis, dataloader=data_loader, config={"G_lr": args.G_lr, "D_lr": args.D_lr, "showlog": 100})
for epoch in range(args.num_epochs):
    for _ in range(900):
        # Unsupervised training
        for batch_idx, batch in enumerate(data_loader):
            null_images = batch["null_image"].to(trainer.device)
            # Random Labels in [0, 1]
            labels = torch.rand(null_images.shape[0], 3, 1, 1).to(trainer.device)

            B, C, H, W = images.shape

            loss_G = trainer.train_G_step_un(inputs, labels)
            images = trainer.G.forward(null_images, labels)
            loss_D = trainer.train_D_step(images, labels)
    
    for _ in range(100):
        # Supervised training
        for batch_idx, batch in enumerate(data_loader):
            null_images = batch["null_image"].to(trainer.device)
            images = batch["image"].to(trainer.device)
            labels = batch["label"].to(trainer.device)

            B, C, H, W = images.shape

            loss_G = trainer.train_G_step(null_images, labels, images)
            loss_D = trainer.train_D_step(images, labels)
            
    trainer.save_model(f"outputs/epoch_{epoch}")
    ##################
    sample = data[9]
    imgs = sample["image"].unsqueeze(0).to(trainer.device)
    labels = sample["label"].unsqueeze(0).to(trainer.device)
    null_image = sample["null_image"].unsqueeze(0).to(trainer.device)
    inputs = torch.cat((null_image, labels.view(1, 3, 1, 1).expand(1, 3, 480, 640)), dim=1)

    ref = trainer.G.forward(inputs)
    print(ref.shape)
    ref = ref.squeeze(0)
    ref = to_pil_image(ref)
    plt.imshow(ref)
    plt.axis('off')
    plt.show()

    pred = trainer.D.forward(imgs)
    print(pred)
    print(labels)
    ##################