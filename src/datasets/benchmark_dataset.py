import os
import json
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class BenchmarkDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform or transforms.ToTensor()
        with open(os.path.join(root_dir, f'{split}_annotations.json'), 'r') as f:
            self.annotations = json.load(f)
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img_front = Image.open(ann['front_path']).convert('RGB')
        img_side = Image.open(ann['side_path']).convert('RGB')
        # TODO: The real structure of BenchmarkDataset is needed
        sample = {}
        return sample