from torchvision import transforms  # Add missing import

class PIDSymbolDataset(Dataset):
    def __init__(self, data_dir, mode='small', transform=None, symbol_size_threshold=700):
        # Always include ToTensor first
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Converts PIL Image -> Tensor
            *([transform] if transform else [])  # Add custom transforms after
        ])
        
    def __getitem__(self, idx):
        image = Image.open(sample['image_path']).convert("RGB")
        # Convert to tensor BEFORE returning
        if self.transform:
            image = self.transform(image)  # Now a tensor [C,H,W]
        return image, target  # image is now a tensor
