import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class Data(Dataset):
    def __init__(self, img_dir, max_size=5000):
        self.img_dir = img_dir
        self.img_names = [img_name for img_name in os.listdir(img_dir) if img_name.endswith(('.png', '.jpg', '.jpeg'))]
        if len(self.img_names) > max_size:
            self.img_names = self.img_names[:max_size]
        self.transform = self.default_transform()

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image

    @staticmethod
    def default_transform():
        return transforms.Compose([
            transforms.Resize((720, 720)),
            transforms.ToTensor(),
        ])
    
    def set_transform(self, transform):
        self.transform = transform

    def get_dataloader(self, batch_size=32, shuffle=True, num_workers=0):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)