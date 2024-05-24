from dataloader import Data, TransformedData
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
import torch

import matplotlib
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image

def main():
    matplotlib.use('TkAgg')
    img_dir = '../data/Humans'  # Update this path as needed
    dataset = TransformedData(img_dir=img_dir)
    dataloader = dataset.get_dataloader(batch_size=32, shuffle=True)

    show_image_pairs(dataloader, num_pairs=5)



def show_image_pairs(dataloader, num_pairs=5):
    dataloader_iter = iter(dataloader)

    for i in range(num_pairs):
        print(f"Fetching batch {i+1}/{num_pairs}")
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            print("No more batches available.")
            break

        print(f"Batch type: {type(batch)}")
        print(f"Batch shape: {batch.shape}")

        if not isinstance(batch, torch.Tensor):
            print("Batch is not a tensor.")
            continue

        # Assuming batch size is 32, process the first two images for demonstration
        images = batch[:2]
        labels = [0, 1]  # Dummy labels, update according to your dataset

        if len(images) < 2:
            print("Not enough images in the batch to display pairs.")
            continue

        print("Preparing to display images...")

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        for j in range(2):
            image = images[j]
            label = labels[j]
            print(f"Displaying image {j+1} with label {label}")
            image = to_pil_image(image)
            axes[j].imshow(image)
            axes[j].set_title(f'Label: {label}')
            axes[j].axis('off')

        plt.tight_layout()
        plt.show()

        print("Images displayed.")


if __name__ == '__main__':
    main()
