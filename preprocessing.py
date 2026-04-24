import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from torchvision.datasets import ImageFolder

source_root = Path('/content/drive/MyDrive/AML_Dataset/Warp-C/train_crops')
target_root = Path('/content/Warp-C-Processed/train_crops')
target_root.mkdir(parents=True, exist_ok=True)

processing_pipeline = transforms.Compose([
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0))
])

all_images = list(source_root.rglob('*.png')) + list(source_root.rglob('*.jpg'))
print(f"Found {len(all_images)} images. Starting processing...")

for img_path in tqdm(all_images, desc="Processing Images"):
    try:
        class_name = img_path.parent.name
        new_class_path = target_root / class_name
        new_class_path.mkdir(exist_ok=True)

        with Image.open(img_path).convert('RGB') as img:
            processed_img = processing_pipeline(img)
            processed_img.save(new_class_path / f"aug_{img_path.name}")

    except Exception as e:
        print(f"Skipping {img_path.name}: {e}")

processed_dataset = ImageFolder(root=str(target_root))
targets = np.array(processed_dataset.targets)
class_counts = np.bincount(targets)
weights = len(targets) / (len(class_counts) * class_counts)
weights_tensor = torch.FloatTensor(weights).to('cuda' if torch.cuda.is_available() else 'cpu')

