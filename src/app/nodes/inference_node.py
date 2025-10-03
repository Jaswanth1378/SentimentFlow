import torch
from typing import Dict, Any, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

class InferenceNode:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        self.label_map = {0: "negative", 1: "positive"}
        self.temperature = 1.0
    
    def set_temperature(self, temperature: float):
        self.temperature = temperature
    
    def run(self, text: str) -> Dict[str, Any]:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            scaled_logits = logits / self.temperature
            probs = torch.softmax(scaled_logits, dim=-1).cpu().numpy()[0]
        
        label_idx = int(probs.argmax())
        confidence = float(probs[label_idx])
        pred_label = self.label_map.get(label_idx, f"label_{label_idx}")
        
        probs_dict = {self.label_map.get(i, f"label_{i}"): float(probs[i]) 
                      for i in range(len(probs))}
        
        return {
            "label": pred_label,
            "label_idx": label_idx,
            "probs": probs_dict,
            "confidence": confidence,
            "text": text
        }
