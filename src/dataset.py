from torch.utils.data import Dataset
import os
import cv2
import torch
from typing import Optional, Tuple

class DoomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = os.path.join(self.image_dir, self.images[index])
        image = cv2.imread(img_path)
        aug_image = image.copy()
        if self.transform is not None:
            aug_image = self.transform(image = aug_image)["image"]
        image = torch.from_numpy(image).permute((2, 0, 1))
        return image, aug_image