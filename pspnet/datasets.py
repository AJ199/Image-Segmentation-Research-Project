# datasets.py
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import torchvision.transforms as T

class SimpleSegDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transforms=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.ids = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.transforms = transforms
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, idx):
        name = self.ids[idx]
        img = Image.open(os.path.join(self.image_dir, name)).convert('RGB')
        mask = Image.open(os.path.join(self.mask_dir, name)).convert('L')
        if self.transforms:
            img = self.transforms(img)
        mask = np.array(mask, dtype=np.int64)
        return img, mask
