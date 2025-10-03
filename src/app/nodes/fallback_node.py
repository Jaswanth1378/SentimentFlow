from typing import Dict, Any, Optional, Callable
from transformers import pipeline
from src.app.config import Config

class FallbackNode:
    def __init__(
        self,
        zero_shot_model: str = None,
        zero_shot_labels: list = None,
        user_input_callback: Optional[Callable] = None
    ):
        self.zero_shot_model_name = zero_shot_model or Config.ZERO_SHOT_MODEL
        self.zero_shot_labels = zero_shot_labels or Config.ZERO_SHOT_LABELS
        self.user_input_callback = user_input_callback
        
        self.zero_shot_pipeline = None
    
    def _init_zero_shot(self):
        if self.zero_shot_pipeline is None:
            self.zero_shot_pipeline = pipeline(
                "zero-shot-classification",
                model=self.zero_shot_model_name,
                device=-1
            )
    
    def run(
        self,
        confidence_output: Dict[str, Any],
        interactive: bool = True
    ) -> Dict[str, Any]:
        action = confidence_output["action"]
        text = confidence_output["text"]
        pred_label = confidence_output["label"]
        
        fallback_result = {
            "fallback_activated": True,
            "fallback_strategy": None,
            "fallback_question": None,
            "user_response": None,
            "final_label": pred_label,
            "final_decision_via": "fallback",
            **confidence_output
        }
        
        if action == "ask_clarify" and interactive and self.user_input_callback:
            fallback_result["fallback_strategy"] = "clarification"
            
            opposite_label = "negative" if pred_label == "positive" else "positive"
            question = f"The model predicted '{pred_label}' with {confidence_output['confidence']:.1%} confidence. Was this a {opposite_label} review? (yes/no)"
            fallback_result["fallback_question"] = question
            
            user_response = self.user_input_callback(question)
            fallback_result["user_response"] = user_response
            
            if user_response and user_response.lower() in ['yes', 'y']:
                fallback_result["final_label"] = opposite_label
                fallback_result["final_decision_via"] = "user_clarification"
            else:
                fallback_result["final_label"] = pred_label
                fallback_result["final_decision_via"] = "user_confirmed"
        
        elif action == "escalate" or (action == "ask_clarify" and not interactive):
            self._init_zero_shot()
            fallback_result["fallback_strategy"] = "zero_shot_backup"
            
            zero_shot_result = self.zero_shot_pipeline(
                text,
                candidate_labels=self.zero_shot_labels,
                multi_label=False
            )
            
            backup_label = zero_shot_result['labels'][0]
            backup_confidence = zero_shot_result['scores'][0]
            
            fallback_result["backup_model"] = {
                "label": backup_label,
                "confidence": backup_confidence,
                "all_scores": dict(zip(zero_shot_result['labels'], zero_shot_result['scores']))
            }
            
            if action == "escalate":
                fallback_result["final_label"] = backup_label
                fallback_result["final_decision_via"] = "backup_model_escalation"
            else:
                fallback_result["final_label"] = backup_label
                fallback_result["final_decision_via"] = "backup_model_fallback"
        
        else:
            fallback_result["fallback_activated"] = False
            fallback_result["final_decision_via"] = "direct_prediction"
        
        return fallback_result
