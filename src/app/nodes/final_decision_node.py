from typing import Dict, Any
import uuid
from src.app.logger import logger_instance

class FinalDecisionNode:
    def __init__(self):
        self.logger = logger_instance
    
    def run(self, fallback_output: Dict[str, Any]) -> Dict[str, Any]:
        request_id = str(uuid.uuid4())
        
        log_entry = self.logger.log_inference(
            request_id=request_id,
            input_text=fallback_output["text"],
            pred_label=fallback_output["label"],
            probs=fallback_output["probs"],
            confidence=fallback_output["confidence"],
            confidence_status=fallback_output["status"],
            fallback_activated=fallback_output.get("fallback_activated", False),
            fallback_strategy=fallback_output.get("fallback_strategy"),
            fallback_question=fallback_output.get("fallback_question"),
            user_response=fallback_output.get("user_response"),
            final_label=fallback_output.get("final_label"),
            final_decision_via=fallback_output.get("final_decision_via")
        )
        
        return {
            "request_id": request_id,
            "final_label": fallback_output.get("final_label", fallback_output["label"]),
            "confidence": fallback_output["confidence"],
            "decision_via": fallback_output.get("final_decision_via", "direct_prediction"),
            "log_entry": log_entry
        }
