import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, f1_score

class Evaluator:
    def __init__(self, model, batch_size):
        self.model = model
        self.batch_size = batch_size

    def evaluate(self, test_dataset):
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        all_labels = []
        all_preds = []
        self.model.eval()
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self.model(inputs)
                preds = torch.sigmoid(outputs).squeeze().cpu().numpy()
                labels = labels.cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels)
        
        # Convert lists to numpy arrays for sklearn metrics
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        
        # Binarize predictions based on a threshold of 0.5
        binary_preds = (all_preds >= 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, binary_preds)
        auc = roc_auc_score(all_labels, all_preds)
        precision = precision_score(all_labels, binary_preds)
        f1 = f1_score(all_labels, binary_preds)
        
        return accuracy, auc, precision, f1
