import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class B2RDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform or transforms.ToTensor()
        with open(os.path.join(root_dir, 'labels.json'), 'r') as f:
            self.labels = json.load(f)
        path = os.path.join(root_dir, 'Null.jpg')
        self.null_image = Image.open(path).convert("RGB")
        if self.transform:
            self.null_image = self.transform(self.null_image)
        
        self.x_max = max(label["x"] for label in self.labels["labels"])
        self.y_max = max(label["y"] for label in self.labels["labels"])
        self.theta_max = 90.0
    
    def __len__(self):
        return self.labels["num"]
    
    def __getitem__(self, idx):
        sample = {}
        
        label = self.labels["labels"][idx]
        label_combined = [
            label["x"]/self.x_max,
            label["y"]/self.y_max,
            label["theta"]/self.theta_max,
        ]
        sample["label"] = torch.tensor(label_combined, dtype=torch.float32)
        
        img_path = os.path.join(self.root_dir, f"{idx:03d}.jpg")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        sample['image'] = image
        
        sample["null_image"] = self.null_image
        
        return sample