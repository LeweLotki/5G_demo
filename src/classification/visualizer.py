import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, f1_score
from torchvision.transforms.functional import to_pil_image

class Visualizer:
    def __init__(self):
        self.train_losses = []
        self.train_accuracies = []

    def update_metrics(self, outputs, labels, loss):
        _, preds = torch.max(outputs, 1)
        accuracy = accuracy_score(labels.cpu(), preds.cpu())
        self.train_losses.append(loss)
        self.train_accuracies.append(accuracy)

    def plot_metrics(self):
        epochs = range(1, len(self.train_losses) + 1)

        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color=color)
        ax1.plot(epochs, self.train_losses, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Accuracy', color=color)
        ax2.plot(epochs, self.train_accuracies, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()
        plt.show()

    def calculate_final_metrics(self, model, dataloader):
        model.eval()
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for images, labels in dataloader:
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)

        print(f'Accuracy: {accuracy:.4f}')
        print(f'AUC: {auc:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'F1 Score: {f1:.4f}')

    def show_images_with_labels(self, dataloader, model, num_batches=5):
        model.eval()
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(dataloader):
                if batch_idx >= num_batches:
                    break

                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
                for j, image in enumerate(images):
                    image = to_pil_image(image)
                    pred_label = predicted[j].item()
                    true_label = labels[j].item()
                    axes[j].imshow(image)
                    axes[j].set_title(f'True: {true_label} Pred: {pred_label}')
                    axes[j].axis('off')

                plt.tight_layout()
                plt.show()
