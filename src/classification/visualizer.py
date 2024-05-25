# visualizer.py
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torchvision.transforms.functional import to_pil_image

class Visualizer:
    def __init__(self):
        self.losses = []
        self.accuracies = []

    def update_metrics(self, outputs, labels, loss):
        self.losses.append(loss)
        _, preds = torch.max(outputs, 1)
        accuracy = (preds == labels).float().mean()
        self.accuracies.append(accuracy.item())

    def plot_metrics(self):
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(self.losses, label='Loss')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.accuracies, label='Accuracy')
        plt.xlabel('Batch')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def calculate_final_metrics(self, model, dataloader):
        model.eval()
        all_outputs = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in dataloader:
                outputs = model(inputs)
                all_outputs.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy().flatten())  # Ensure labels are flattened to 1D

        all_outputs = np.array(all_outputs)
        all_labels = np.array(all_labels)

        # Calculate metrics
        preds = np.argmax(all_outputs, axis=1)
        accuracy = accuracy_score(all_labels, preds)
        precision = precision_score(all_labels, preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, preds, average='weighted', zero_division=0)
        
        # Check if both classes are present in the labels
        if len(np.unique(all_labels)) > 1:
            auc = roc_auc_score(all_labels, all_outputs[:, 1], average='weighted')
        else:
            auc = float('nan')
            print("Warning: Only one class present in y_true. ROC AUC score is not defined.")

        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1 Score: {f1:.4f}')
        print(f'AUC: {auc:.4f}')

    def show_images_with_labels(self, dataloader, model, num_batches=5):
        model.eval()
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(dataloader):
                if batch_idx >= num_batches:
                    break

                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                batch_size = len(images)
                fig, axes = plt.subplots(1, batch_size, figsize=(15, 5))
                
                if batch_size == 1:
                    axes = [axes]

                for j in range(batch_size):
                    image = to_pil_image(images[j])
                    label = predicted[j].item()
                    actual_label = labels[j].item() if labels.dim() > 0 else labels.item()  # Handle 0-dim tensor
                    axes[j].imshow(image)
                    axes[j].set_title(f'Pred: {label}, Act: {actual_label}')
                    axes[j].axis('off')

                plt.tight_layout()
                plt.show()
