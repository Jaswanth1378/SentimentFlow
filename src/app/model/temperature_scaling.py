import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple
import numpy as np

class TemperatureScaling:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.temperature = nn.Parameter(torch.ones(1).to(device))
    
    def _compute_ece(self, probs, labels, n_bins=15):
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = predictions == labels
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = np.mean(in_bin)
            
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(accuracies[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def calibrate(self, val_loader: DataLoader, max_iter: int = 50, lr: float = 0.01) -> float:
        self.model.eval()
        nll_criterion = nn.CrossEntropyLoss()
        
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(**inputs)
                all_logits.append(outputs.logits)
                all_labels.append(labels)
        
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        def eval_loss():
            loss = nll_criterion(all_logits / self.temperature, all_labels)
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)
        
        optimal_temp = self.temperature.item()
        print(f"Optimal temperature: {optimal_temp:.4f}")
        
        calibrated_probs = torch.softmax(all_logits / self.temperature, dim=1).cpu().numpy()
        ece_before = self._compute_ece(
            torch.softmax(all_logits, dim=1).cpu().numpy(),
            all_labels.cpu().numpy()
        )
        ece_after = self._compute_ece(calibrated_probs, all_labels.cpu().numpy())
        
        print(f"ECE before calibration: {ece_before:.4f}")
        print(f"ECE after calibration: {ece_after:.4f}")
        
        return optimal_temp

def calibrate_model(model, val_loader, device='cpu') -> float:
    scaler = TemperatureScaling(model, device)
    return scaler.calibrate(val_loader)
