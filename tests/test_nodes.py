import pytest
import torch
from unittest.mock import Mock, MagicMock, patch
from src.app.nodes.inference_node import InferenceNode
from src.app.nodes.confidence_node import ConfidenceCheckNode
from src.app.nodes.fallback_node import FallbackNode
from src.app.nodes.final_decision_node import FinalDecisionNode

class TestInferenceNode:
    @patch('src.app.nodes.inference_node.AutoTokenizer')
    @patch('src.app.nodes.inference_node.AutoModelForSequenceClassification')
    def test_inference_output_structure(self, mock_model_class, mock_tokenizer_class):
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_outputs = MagicMock()
        mock_outputs.logits = torch.tensor([[0.2, 0.8]])
        mock_model.return_value = mock_outputs
        mock_model_class.from_pretrained.return_value = mock_model
        
        node = InferenceNode("fake-model-path")
        result = node.run("This is a test text")
        
        assert "label" in result
        assert "probs" in result
        assert "confidence" in result
        assert "text" in result
        assert result["text"] == "This is a test text"

class TestConfidenceCheckNode:
    def test_high_confidence_accept(self):
        node = ConfidenceCheckNode(threshold_accept=0.75, threshold_clarify=0.50)
        
        inference_output = {
            "label": "positive",
            "probs": {"positive": 0.85, "negative": 0.15},
            "confidence": 0.85,
            "text": "Great movie!"
        }
        
        result = node.run(inference_output)
        
        assert result["action"] == "accept"
        assert result["status"] == "HIGH"
    
    def test_medium_confidence_clarify(self):
        node = ConfidenceCheckNode(threshold_accept=0.75, threshold_clarify=0.50)
        
        inference_output = {
            "label": "positive",
            "probs": {"positive": 0.65, "negative": 0.35},
            "confidence": 0.65,
            "text": "Okay movie"
        }
        
        result = node.run(inference_output)
        
        assert result["action"] == "ask_clarify"
        assert result["status"] == "MEDIUM"
    
    def test_low_confidence_escalate(self):
        node = ConfidenceCheckNode(threshold_accept=0.75, threshold_clarify=0.50)
        
        inference_output = {
            "label": "positive",
            "probs": {"positive": 0.45, "negative": 0.55},
            "confidence": 0.45,
            "text": "Confusing movie"
        }
        
        result = node.run(inference_output)
        
        assert result["action"] == "escalate"
        assert result["status"] == "LOW"

class TestFallbackNode:
    def test_clarification_with_user_yes(self):
        user_callback = Mock(return_value="yes")
        node = FallbackNode(user_input_callback=user_callback)
        
        confidence_output = {
            "action": "ask_clarify",
            "text": "Okay movie",
            "label": "positive",
            "confidence": 0.65,
            "status": "MEDIUM",
            "probs": {"positive": 0.65, "negative": 0.35}
        }
        
        result = node.run(confidence_output, interactive=True)
        
        assert result["fallback_activated"] == True
        assert result["fallback_strategy"] == "clarification"
        assert result["final_label"] == "negative"
        assert result["final_decision_via"] == "user_clarification"
        user_callback.assert_called_once()
    
    def test_clarification_with_user_no(self):
        user_callback = Mock(return_value="no")
        node = FallbackNode(user_input_callback=user_callback)
        
        confidence_output = {
            "action": "ask_clarify",
            "text": "Okay movie",
            "label": "positive",
            "confidence": 0.65,
            "status": "MEDIUM",
            "probs": {"positive": 0.65, "negative": 0.35}
        }
        
        result = node.run(confidence_output, interactive=True)
        
        assert result["fallback_activated"] == True
        assert result["final_label"] == "positive"
        assert result["final_decision_via"] == "user_confirmed"

class TestFinalDecisionNode:
    @patch('src.app.nodes.final_decision_node.logger_instance')
    def test_final_decision_logging(self, mock_logger):
        mock_logger.log_inference.return_value = {"logged": True}
        
        node = FinalDecisionNode()
        
        fallback_output = {
            "text": "Test movie",
            "label": "positive",
            "probs": {"positive": 0.85, "negative": 0.15},
            "confidence": 0.85,
            "status": "HIGH",
            "fallback_activated": False,
            "final_label": "positive",
            "final_decision_via": "direct_prediction"
        }
        
        result = node.run(fallback_output)
        
        assert "request_id" in result
        assert "final_label" in result
        assert result["final_label"] == "positive"
        mock_logger.log_inference.assert_called_once()
