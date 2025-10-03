from typing import Dict, Any, Optional, Callable
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
from src.app.nodes.inference_node import InferenceNode
from src.app.nodes.confidence_node import ConfidenceCheckNode
from src.app.nodes.fallback_node import FallbackNode
from src.app.nodes.final_decision_node import FinalDecisionNode

class ClassificationState(TypedDict, total=False):
    text: str
    label: str
    label_idx: int
    probs: Dict[str, float]
    confidence: float
    action: str
    status: str
    threshold_accept: float
    threshold_clarify: float
    fallback_activated: bool
    fallback_strategy: Optional[str]
    fallback_question: Optional[str]
    user_response: Optional[str]
    final_label: str
    final_decision_via: str
    backup_model: Optional[Dict[str, Any]]
    request_id: str
    decision_via: str
    log_entry: Dict[str, Any]

class SelfHealingDAG:
    def __init__(
        self,
        model_path: str,
        user_input_callback: Optional[Callable] = None,
        interactive: bool = True,
        device: str = "cpu"
    ):
        self.inference_node = InferenceNode(model_path, device=device)
        self.confidence_node = ConfidenceCheckNode()
        self.fallback_node = FallbackNode(user_input_callback=user_input_callback)
        self.final_decision_node = FinalDecisionNode()
        self.interactive = interactive
        
        self.graph = self._build_graph()
    
    def _build_graph(self):
        workflow = StateGraph(ClassificationState)
        
        workflow.add_node("inference", self._inference_wrapper)
        workflow.add_node("confidence_check", self._confidence_wrapper)
        workflow.add_node("fallback", self._fallback_wrapper)
        workflow.add_node("final_decision", self._final_decision_wrapper)
        
        workflow.set_entry_point("inference")
        
        workflow.add_edge("inference", "confidence_check")
        
        workflow.add_conditional_edges(
            "confidence_check",
            self._should_use_fallback,
            {
                "fallback": "fallback",
                "final": "final_decision"
            }
        )
        
        workflow.add_edge("fallback", "final_decision")
        workflow.add_edge("final_decision", END)
        
        return workflow.compile()
    
    def _inference_wrapper(self, state: ClassificationState) -> ClassificationState:
        result = self.inference_node.run(state["text"])
        return {**state, **result}
    
    def _confidence_wrapper(self, state: ClassificationState) -> ClassificationState:
        result = self.confidence_node.run(state)
        return {**state, **result}
    
    def _fallback_wrapper(self, state: ClassificationState) -> ClassificationState:
        result = self.fallback_node.run(state, interactive=self.interactive)
        return {**state, **result}
    
    def _final_decision_wrapper(self, state: ClassificationState) -> ClassificationState:
        result = self.final_decision_node.run(state)
        return {**state, **result}
    
    def _should_use_fallback(self, state: ClassificationState) -> str:
        if state["action"] in ["ask_clarify", "escalate"]:
            return "fallback"
        return "final"
    
    def run(self, text: str) -> Dict[str, Any]:
        initial_state = {"text": text}
        final_state = self.graph.invoke(initial_state)
        return final_state
    
    def set_temperature(self, temperature: float):
        self.inference_node.set_temperature(temperature)
