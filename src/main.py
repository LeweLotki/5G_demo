import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
import torch
from dataloader import TransformedData


def main():
    img_dir = '../data/Humans'
    data = TransformedData(img_dir, max_size=20)
    dataloader = data.get_dataloader()
    
    show_images(dataloader=dataloader)

def show_images(dataloader):

    for batch_idx, (images, labels) in enumerate(dataloader):
        # print(f'debug: {type(data)}')
        print(f"Batch {batch_idx + 1}:")
        print(f"Images tensor shape: {images.shape}")
        print(f"Labels tensor shape: {labels.shape}")
        print(f"Labels: {labels}")

        fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
        for j, _ in enumerate(images):
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
