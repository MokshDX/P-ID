import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import json
from torchvision import transforms

class CocoDetectionDataset(Dataset):
    def __init__(self, root, annFile, transform=None):
        """
        Initialize the dataset with root directory and COCO-format annotation file.

        Args:
            root (str): Path to the directory containing images.
            annFile (str): Path to COCO-format JSON annotation file.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.root = root
        self.transform = transform  # Store transform for later use

        # Load COCO annotations
        with open(annFile, 'r') as f:
            self.coco = json.load(f)

        # Map image IDs to file names
        self.imgs = {img['id']: img['file_name'] for img in self.coco['images']}

        # Map image IDs to their annotations
        self.annotations = {}
        for ann in self.coco['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)

        self.ids = list(self.imgs.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]
        img_path = os.path.join(self.root, self.imgs[img_id])
        image = Image.open(img_path).convert('RGB')

        anns = self.annotations.get(img_id, [])

        boxes = []
        labels = []

        for ann in anns:
            bbox = ann['bbox']  # COCO format: [x, y, width, height]
            x, y, w, h = bbox
            boxes.append([x, y, x + w, y + h])  # Convert to [x1, y1, x2, y2]
            labels.append(ann['category_id'])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {'boxes': boxes, 'labels': labels}

        if self.transform:
            image = self.transform(image)  # Apply transform here

        return image, target
