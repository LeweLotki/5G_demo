import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import DataLoader, TensorDataset
import torch
from load_data import load_images_and_labels, transform_images

def main():
    img_dir = '../data/Humans'  # Ensure this path is correct
    images, labels = load_images_and_labels(img_dir, max_size=20)  # Small number for testing
    
    # Transform images
    images = transform_images(images)
    
    # Convert lists to tensors
    images_tensor = torch.stack(images)
    labels_tensor = torch.tensor(labels)
    
    # Create a TensorDataset and DataLoader
    dataset = TensorDataset(images_tensor, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Debug: Print the length of the dataset
    print(f"Dataset length: {len(dataset)}")

    for batch_idx, (images, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}:")
        print(f"Images tensor shape: {images.shape}")
        print(f"Labels tensor shape: {labels.shape}")
        print(f"Labels: {labels}")

        fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
        for j in range(len(images)):
            image = images[j]
            label = labels[j].item()  # Ensure label is a scalar
            print(f"Label type: {type(label)}, Label value: {label}")  # Debug print
            image = to_pil_image(image)  # Convert tensor to PIL Image

            axes[j].imshow(image)
            axes[j].set_title(f'Label: {label}')
            axes[j].axis('off')

        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    main()
