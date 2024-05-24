from data import Data, TransformedData
import matplotlib.pyplot as plt

def main():
    
    img_dir = '../data/Humans'  # Update this path as needed
    dataset = TransformedData(img_dir=img_dir)
    dataloader = dataset.get_dataloader(batch_size=32, shuffle=True)

    show_image_pairs(dataloader, num_pairs=5)

import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image

def show_image_pairs(dataloader, num_pairs=5):
    fig, axes = plt.subplots(num_pairs, 2, figsize=(10, 5 * num_pairs))
    dataloader_iter = iter(dataloader)

    for i in range(num_pairs):
        images = next(dataloader_iter)
        if len(images) < 2:
            break

        image1, image2 = images[:2]

        # Convert tensors back to PIL images
        image1 = to_pil_image(image1)
        image2 = to_pil_image(image2)

        axes[i, 0].imshow(image1)
        axes[i, 1].imshow(image2)

        axes[i, 0].axis('off')
        axes[i, 1].axis('off')

    plt.show()

if __name__ == '__main__':
    main()
