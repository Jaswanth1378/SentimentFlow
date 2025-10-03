from typing import Dict, Any
from src.app.config import Config

class ConfidenceCheckNode:
    def __init__(
        self,
        threshold_accept: float = None,
        threshold_clarify: float = None
    ):
        self.threshold_accept = threshold_accept or Config.CONFIDENCE_THRESHOLDS["accept"]
        self.threshold_clarify = threshold_clarify or Config.CONFIDENCE_THRESHOLDS["clarify"]
    
    def run(self, inference_output: Dict[str, Any]) -> Dict[str, Any]:
        confidence = inference_output["confidence"]
        
        if confidence >= self.threshold_accept:
            action = "accept"
            status = "HIGH"
        elif confidence >= self.threshold_clarify:
            action = "ask_clarify"
            status = "MEDIUM"
        else:
            action = "escalate"
            status = "LOW"
        
        return {
            "action": action,
            "status": status,
            "confidence": confidence,
            "threshold_accept": self.threshold_accept,
            "threshold_clarify": self.threshold_clarify,
            **inference_output
        }
