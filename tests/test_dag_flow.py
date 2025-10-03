import pytest
from unittest.mock import Mock, patch, MagicMock
from src.app.dag import SelfHealingDAG

class TestDAGIntegration:
    @patch('src.app.dag.InferenceNode')
    @patch('src.app.dag.ConfidenceCheckNode')
    @patch('src.app.dag.FallbackNode')
    @patch('src.app.dag.FinalDecisionNode')
    def test_high_confidence_flow(self, mock_final, mock_fallback, mock_confidence, mock_inference):
        mock_inference_instance = MagicMock()
        mock_inference_instance.run.return_value = {
            "label": "positive",
            "probs": {"positive": 0.95, "negative": 0.05},
            "confidence": 0.95,
            "text": "Amazing movie!"
        }
        mock_inference.return_value = mock_inference_instance
        
        mock_confidence_instance = MagicMock()
        mock_confidence_instance.run.return_value = {
            "action": "accept",
            "status": "HIGH",
            "label": "positive",
            "confidence": 0.95
        }
        mock_confidence.return_value = mock_confidence_instance
        
        mock_final_instance = MagicMock()
        mock_final_instance.run.return_value = {
            "request_id": "test-123",
            "final_label": "positive",
            "confidence": 0.95,
            "decision_via": "direct_prediction"
        }
        mock_final.return_value = mock_final_instance
        
        dag = SelfHealingDAG(model_path="fake-path", interactive=False)
        result = dag.run("Amazing movie!")
        
        assert result["final_label"] == "positive"
        mock_fallback_instance = mock_fallback.return_value
        mock_fallback_instance.run.assert_not_called()
    
    @patch('src.app.dag.InferenceNode')
    @patch('src.app.dag.ConfidenceCheckNode')
    @patch('src.app.dag.FallbackNode')
    @patch('src.app.dag.FinalDecisionNode')
    def test_low_confidence_fallback_flow(self, mock_final, mock_fallback, mock_confidence, mock_inference):
        mock_inference_instance = MagicMock()
        mock_inference_instance.run.return_value = {
            "label": "positive",
            "probs": {"positive": 0.55, "negative": 0.45},
            "confidence": 0.55,
            "text": "Unclear review"
        }
        mock_inference.return_value = mock_inference_instance
        
        mock_confidence_instance = MagicMock()
        mock_confidence_instance.run.return_value = {
            "action": "ask_clarify",
            "status": "MEDIUM",
            "label": "positive",
            "confidence": 0.55,
            "text": "Unclear review"
        }
        mock_confidence.return_value = mock_confidence_instance
        
        mock_fallback_instance = MagicMock()
        mock_fallback_instance.run.return_value = {
            "fallback_activated": True,
            "fallback_strategy": "clarification",
            "final_label": "negative",
            "final_decision_via": "user_clarification",
            "label": "positive",
            "confidence": 0.55,
            "status": "MEDIUM"
        }
        mock_fallback.return_value = mock_fallback_instance
        
        mock_final_instance = MagicMock()
        mock_final_instance.run.return_value = {
            "request_id": "test-456",
            "final_label": "negative",
            "confidence": 0.55,
            "decision_via": "user_clarification"
        }
        mock_final.return_value = mock_final_instance
        
        dag = SelfHealingDAG(model_path="fake-path", interactive=True)
        result = dag.run("Unclear review")
        
        mock_fallback_instance.run.assert_called_once()
        assert result["decision_via"] == "user_clarification"
